"""
Stage 1: 冻结基模，仅训练 Indexer
改进点：偶数层选择性蒸馏 + 全分布 KL (避免梯度消失)
"""

# Custom Qwen2ForCausalLM (with the modified Indexer-equipped attention).。
from model import Qwen2ForCausalLM

# Custom dataset class that tokenizes and formats the SFT data.
from dataset import SFTDataset

# HuggingFace training utilities: Trainer (training loop), TrainingArguments (hyperparameters), AutoTokenizer, and a basic data collator.。
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator

import torch
# 函数式接口——损失计算里要用到 F.softmax 和 F.kl_div
import torch.nn.functional as F
# 项目配置加载器（模型路径、数据路径、输出目录等）。
from load_config import CONFIG


class DSAWarmupTrainer(Trainer):
    """
    Stage 1 Trainer. Trains only the Indexer (base model frozen).
    Distills on even-numbered layers only, using full-distribution KL.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass, requesting attention outputs (needed to access the Indexer's raw scores per layer).
        # 前向传播，要求返回 attention 相关输出（需要用它拿到每一层 Indexer 的原始打分）。
        outputs = model(**inputs, output_attentions=True)

        all_attentions = outputs.attentions   # list[tuple], 长度 = num_layers

        # Initialize the accumulated KL loss as a zero tensor on the correct device.
        attention_kl_loss = torch.tensor(0.0, device=outputs.loss.device)

        # Counter for how many layers actually get distilled (only even-indexed ones).
        distill_layers = 0

        for layer_idx, attention in enumerate(all_attentions):
            # Even-layer selective distillation
            if layer_idx % 2 != 0:
                continue
            
            # ncrement the count of layers that will actually contribute to the loss.
            distill_layers += 1

            # Unpack this layer's tuple — indices aren't used in warmup (no gather needed yet), only the raw scores.
            # 拆出这一层的元组 —— warmup 阶段不需要用到索引（还不需要 gather），只用原始打分。
            _, raw_attn_weights, indexer_attn_scores = attention

            # Teacher signal — apply softmax to the FULL dense attention matrix (no top-k truncation here).
            # Teacher 信号 —— 对完整的 dense attention 矩阵做 softmax（这里没有 top-k 截断）。
            teacher = F.softmax(raw_attn_weights, dim=-1)   # [B, H, S, S]

            # Sum across all attention heads, collapsing multi-head weights into a single distribution.
            # 在所有注意力头上求和，把多头权重压缩成单一分布。
            teacher = teacher.sum(dim=1, keepdim=True)      # [B, 1, S, S]

            # L1-normalize so the teacher distribution sums to 1 along the key dimension (required for a valid KL target).
            # 做 L1 归一化，让 teacher 分布在 key 维度上总和为 1（这是合法 KL 目标分布的必要条件）。
            teacher = teacher / teacher.norm(p=1, dim=-1, keepdim=True)  # L1 norm

            # Student signal — softmax over the Indexer's own raw scores, also over the FULL sequence (full-distribution KL).
            # Student 信号 —— 对 Indexer 自己的原始打分做 softmax，同样是在完整序列上做（全分布 KL）。
            student = F.softmax(indexer_attn_scores, dim=-1)  # [B, 1, S, S]

            # Clamp away exact zeros to avoid taking log(0) = -inf in the next step.
            # 把精确为零的值夹住，避免下一步取 log(0) 得到 -inf。
            student = student.clamp(min=1e-8)

            # KL divergence between student (log-space) and detached teacher — "detached" means no gradient flows back into the teacher (dense attention) side.
            # 计算 student（对数空间）和 detach 过的 teacher 之间的 KL 散度——"detach" 意味着梯度不会流回 teacher（dense attention）这一侧。
            
            # kl_div(input, target)约定 input 已经是 log(概率) 的格式，target 是概率格式
            # 计算 target * (log(target) - input),也就是 KL(teacher || student)
            # student 和 teacher 都是 [B, 1, S, S] 的格式，计算的结果也是 [B, 1, S, S]
            
            # reduction determines how we compress the result
            #   'none': no reduction, keep the original shape
            #   'sum': sum up all the elements
            #   'mean': sum up all the elements, then divide it by the total number of elements
            #   'batchmean': sum up all the elements, then divide it by the batch size
            #   so we get the average KL loss across samples in the batch
            kl_loss_elementwise = F.kl_div(student.log(), teacher.detach(), reduction="none")
            kl_loss = kl_loss_elementwise.sum(dim=-1).mean()

            # Accumulate this layer's KL loss into the running total.
            # 把这一层的 KL loss 累加进总计。
            attention_kl_loss = attention_kl_loss + kl_loss

        # Average over however many layers were actually distilled (guards against division by zero).
        # 除以实际参与蒸馏的层数取平均（避免除以零）。
        if distill_layers > 0:
            attention_kl_loss = attention_kl_loss / distill_layers

        # Return either (loss, outputs) or just loss, depending on what the Trainer's internals requested.
        # 根据 Trainer 内部的要求，返回 (loss, outputs) 或者只返回 loss。
        return (attention_kl_loss, outputs) if return_outputs else attention_kl_loss


# Only execute the training script when run directly (not when imported).
# 只有直接运行这个脚本时才执行训练（被导入时不会自动跑）。
if __name__ == "__main__":
    # Load the original base model (no Indexer training has happened yet).
    model = Qwen2ForCausalLM.from_pretrained(CONFIG["base_model_path"])

    # Freeze every parameter EXCEPT those belonging to the Indexer — this is Stage 1's defining constraint.
    for name, param in model.named_parameters():
        param.requires_grad = "indexer" in name

    # Count how many parameters will actually receive gradients.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count the total number of parameters in the model, for comparison.
    total_params = sum(p.numel() for p in model.parameters())

    # Print both numbers and the trainable percentage, as a sanity check before training starts.
    print(f"可训练参数: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Load the tokenizer matching the base model.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])

    # Construct the HuggingFace TrainingArguments object with all hyperparameters for Stage 1.
    args = TrainingArguments(
        output_dir=CONFIG["warmup_output_path"],   # 检查点和日志的保存路径
        max_steps=5000,                            # 总共运行步数
        num_train_epochs=1,                        # 总共训练轮数
        do_train=True,                             # 显式开启 Trainer 的训练模式
        per_device_train_batch_size=2,             # 每个设备每次前向/反向传播处理的样本数
        gradient_accumulation_steps=1,             # 不做梯度累积——每一步都更新权重
        logging_steps=50,                          # 每隔多少步记录一次训练指标
        report_to="tensorboard",                   # 把日志发送给 TensorBoard 用于可视化
        save_strategy="steps",                     # 按步数（而不是按 epoch）保存检查点
        save_steps=1000,                           # 每 1000 步保存一次检查点
        save_total_limit=3,                        # 磁盘上最多保留 3 个检查点，旧的会被删除
        bf16=True,                                 # 训练时使用 bfloat16 混合精度
        learning_rate=1e-3,                        # 学习率——相对较大，因为 Indexer 从随机初始化开始
        lr_scheduler_type="cosine",                # 使用余弦学习率调度（训练过程中平滑衰减）
        dataloader_num_workers=8,                  # 数据加载用的并行 worker 进程数
        dataloader_pin_memory=True,                # 固定内存（pinned memory），加快 CPU 到 GPU 的数据传输速度
    )

    data_collator = DefaultDataCollator()   # no special padding logic needed, since SFTDataset already pads.

    # Instantiate the training dataset, tokenizing and truncating/padding to a fixed max length.
    dataset = SFTDataset(CONFIG["train_data_path"], tokenizer=tokenizer, max_seq_len=2048)

    # Instantiate our custom Trainer subclass with the model, arguments, dataset, tokenizer, and collator.
    trainer = DSAWarmupTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training from scratch (not resuming from any existing checkpoint).
    trainer.train(resume_from_checkpoint=False)

    # Save the final model weights to the configured warmup output path.
    trainer.save_model(CONFIG["warmup_model_path"])

    # Save additional trainer state (optimizer state, scheduler state, etc.) for potential resumption.
    trainer.save_state()
