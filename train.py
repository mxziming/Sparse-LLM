"""
Stage 2: 基模 + Indexer 联合训练
改进点：双学习率分组 + 偶数层选择性蒸馏
"""

# the custom Qwen2ForCausalLM
from model import Qwen2ForCausalLM

# Custom dataset class
from dataset import SFTDataset

from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
import torch
import torch.nn.functional as F   # we need softmax, kl_div, and gather operations used in the loss.
from load_config import CONFIG    # Project configuration loader.

# Base model learning rate — small, to protect pretrained weights from being disrupted.
# 基模学习率较小，用来保护预训练权重不被过度扰动。
LR_BASE_MODEL = 1e-5

# Indexer learning rate — larger, since it continues converging from where warmup left off.
# Indexer 学习率较大，因为它要继续从 warmup 阶段的状态收敛。
LR_INDEXER    = 1e-4


class DSAJointTrainer(Trainer):
    """
    Stage 2 Trainer. Dual learning rate via a custom optimizer; even-layer distillation;
    total loss = CE_loss + KL_loss.
    """

    def create_optimizer(self):
        """
        Splits model parameters into two groups with different learning rates:
            indexer.* -> LR_INDEXER (large, keeps converging)
            everything else -> LR_BASE_MODEL (small, fine-tunes to adapt to sparsity)
        """

        indexer_params, base_params = [], []

        for name, param in self.model.named_parameters():
            # Skip any parameter that's frozen (requires_grad=False).
            if not param.requires_grad:
                continue
            # Route the parameter into the Indexer group if "indexer" appears in its name.
            if "indexer" in name:
                indexer_params.append(param)
            # Otherwise, it belongs to the base model group.
            else:
                base_params.append(param)

        # Build the param_groups list expected by PyTorch optimizers — each dict can have its own learning rate.
        param_groups = [
            {"params": base_params,    "lr": LR_BASE_MODEL, "name": "base_model"},
            {"params": indexer_params, "lr": LR_INDEXER,    "name": "indexer"},
        ]

        # Ask the Trainer base class which optimizer class to use and what default kwargs it would normally pass.
        # 询问 Trainer 基类应该用哪个优化器类，以及它通常会传入哪些默认关键字参数。
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # Remove the global "lr" key from the default kwargs — otherwise it would override our per-group learning rates.
        # 从默认参数中移除全局的 "lr" 键, 否则它会覆盖我们为每组单独设置的学习率。
        optimizer_kwargs.pop("lr", None)

        # Instantiate the optimizer with our custom param_groups instead of a single flat parameter list.
        # 用我们自定义的 param_groups（而不是单一的扁平参数列表）来实例化优化器。
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        # Print a confirmation showing how many parameter tensors landed in each group and at what learning rate.
        # 打印一条确认信息，显示每组分到了多少个参数张量，以及对应的学习率。
        print(f"[双学习率] 基模参数 {len(base_params)} 组: lr={LR_BASE_MODEL} | "
              f"Indexer 参数 {len(indexer_params)} 组: lr={LR_INDEXER}")

        # Return the constructed optimizer, as required by the Trainer's internal API contract.
        # 返回构造好的优化器，这是 Trainer 内部 API 约定所要求的。
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass, requesting attention outputs needed for the distillation loss.
        # 前向传播，要求返回蒸馏损失需要用到的 attention 相关输出。
        outputs = model(**inputs, output_attentions=True)
        all_attentions = outputs.attentions

        # The standard language-modeling cross-entropy loss, computed inside the model's forward pass.
        # 标准的语言建模交叉熵损失，在模型前向传播内部计算得出。
        ce_loss = outputs.loss

        # Initialize the accumulated KL loss.
        # 初始化累积的 KL loss。
        attention_kl_loss = torch.tensor(0.0, device=ce_loss.device)

        # Counter for layers actually distilled.
        # 实际参与蒸馏的层数计数。
        distill_layers = 0

        for layer_idx, attention in enumerate(all_attentions):
            # Even-layer selective distillation / 偶数层选择性蒸馏
            if layer_idx % 2 != 0:
                continue
            distill_layers += 1

            # Unpack this layer's tuple
            # 这里 topk_indices 真正被使用，因为联合训练阶段需要用 top-k 来 gather KL loss 所需的子集。
            topk_indices, raw_attn_weights, indexer_attn_scores = attention

            # Teacher signal — but now restricted to only the top-k positions (not the full sequence), via gather.
            # Teacher 信号， 但现在通过 gather 限定在 top-k 位置上（不是完整序列）。
            # [B, heads, S, S] -> [B, heads, S, topk]
            raw_attn_weights_topk = torch.gather(
                raw_attn_weights, -1,
                # Expand the shared indices across all heads so gather's index tensor matches raw_attn_weights' head dimension.
                # 把共享的索引广播到所有头，让 gather 的索引张量和 raw_attn_weights 的头维度对齐。
                # [B, 1, S, topk] -> [B, heads, S, topk]
                topk_indices.expand(-1, raw_attn_weights.shape[1], -1, -1)
            )
            
            # Softmax over just the gathered top-k subset.
            # 在 gather 出来的 top-k 子集上做 softmax。
            teacher = F.softmax(raw_attn_weights_topk, dim=-1)   # [B, H, S, K]
            # Sum across heads.
            # 在头维度上求和。
            teacher = teacher.sum(dim=1, keepdim=True)           # [B, 1, S, K]
            # L1-normalize so the teacher's top-k distribution sums to 1.
            # 做 L1 归一化，让 teacher 的 top-k 分布总和为1。
            teacher = teacher / teacher.norm(p=1, dim=-1, keepdim=True)

            # Student signal — also gathered down to the same top-k indices.
            # Student 信号, 同样 gather 到相同的 top-k 索引上。
            indexer_topk = torch.gather(indexer_attn_scores, -1, topk_indices)
            
            # Softmax over the top-k subset.
            # 在 top-k 子集上做 softmax。
            student = F.softmax(indexer_topk, dim=-1)            # [B, 1, S, K]
            # Clamp to avoid log(0).
            # 夹住避免 log(0)。
            student = student.clamp(min=1e-8)

            # KL divergence between student and detached teacher, restricted to the top-k subset.
            # student 和 detach 过的 teacher 之间的 KL 散度，限定在 top-k 子集上计算。
            kl_loss_elementwise = F.kl_div(student.log(), teacher.detach(), reduction="none")
            kl_loss = kl_loss_elementwise.sum(dim=-1).mean()
            # Accumulate.
            # 累加。
            attention_kl_loss = attention_kl_loss + kl_loss

        # Average over the number of distilled layers.
        # 除以参与蒸馏的层数取平均。
        if distill_layers > 0:
            attention_kl_loss = attention_kl_loss / distill_layers

        # Combine the two loss terms — language modeling quality plus Indexer/sparsity alignment.
        # 把两项损失结合起来——语言建模质量加上 Indexer/稀疏对齐。
        loss = ce_loss + attention_kl_loss

        # Return according to the Trainer's expected interface.
        # 按照 Trainer 期望的接口返回结果。
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: dict, start_time=None):
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group_name = group.get("name", "unnamed")
                logs[f"lr/{group_name}"] = group["lr"]
        super().log(logs, start_time)


if __name__ == "__main__":
    # Load the model from the WARMUP checkpoint — Stage 2 continues from where Stage 1 left off.
    model = Qwen2ForCausalLM.from_pretrained(CONFIG["warmup_model_path"])

    # Count trainable parameters — should now be ALL parameters, since nothing is frozen anymore.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,}")

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])

    # Construct training arguments for Stage 2 (note the much larger max_steps than Stage 1).
    args = TrainingArguments(
        output_dir=CONFIG["train_output_path"],   # 这一阶段检查点的输出目录
        max_steps=30000,                          # 30000 步，比 warmup 长十倍，因为现在需要让整个模型都适应稀疏结构
        do_train=True,
        per_device_train_batch_size=1,            # 每设备 batch size，比 warmup 更小
        gradient_accumulation_steps=1,
        logging_steps=50,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        bf16=True,
        # This global learning_rate is mostly a placeholder for the scheduler's reference point — actual per-group rates come from create_optimizer above.
        # 这里的全局 learning_rate 主要只是给调度器一个参考基准，真正各组的学习率由上面的 create_optimizer 控制。
        learning_rate=LR_BASE_MODEL,
        lr_scheduler_type="cosine",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    # Basic data collator.
    data_collator = DefaultDataCollator()

    # Same dataset construction as warmup (same data, same max sequence length).
    dataset = SFTDataset(CONFIG["train_data_path"], tokenizer=tokenizer, max_seq_len=2048)

    # Instantiate the joint-stage trainer.
    trainer = DSAJointTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training (not resuming from a checkpoint).
    trainer.train(resume_from_checkpoint=False)

    # Save the final jointly-trained model.
    trainer.save_model(CONFIG["train_model_path"])

    # Save additional trainer state.
    trainer.save_state()