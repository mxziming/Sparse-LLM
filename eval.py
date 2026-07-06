"""
Break-even 基准评估 + 多序列长度扫描

  model.py 只改动了 decode 分支（seqlen == 1，配合 KV cache 增量生成），
  所以本版评测 "prefill 一次 + decode 多步" 的自回归过程：
    1. 用一段长度为 seq_len 的上下文做一次 prefill，建立 KV cache
    2. 在这份 KV cache 基础上连续 decode N 步（每步只新增 1 个 token）
    3. 只对 decode 阶段计时 —— 这才是 decode 真正生效、能体现 O(S) -> O(K) 的地方
"""

import time
import random
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from load_config import CONFIG
from model import Qwen2ForCausalLM

NUM_CONTEXTS = 10


# configuration
@dataclass
class EvalConfig:
    # 用于 dense 和 sparse 两种测量的单一训练后模型路径（同一个模型，切换 bypass 开关对比）。
    model_path: str = CONFIG["train_model_path"]
    # 对应分词器的路径。
    tokenizer_path: str = CONFIG["tokenizer_path"]
    # 要扫描的 prefill 长度（即 decode 开始前，KV cache 中已有的历史长度 S_total）。
    seq_lengths: List[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
    # 每个 prefill 长度下，连续 decode 多少步（每步计时一次，取平均）。
    decode_steps: int = 100
    # 预热 decode 步数（不计入计时）。
    warmup_decode_steps: int = 10
    # 固定随机种子，确保 dense 和 sparse 看到完全相同的合成上下文与生成 token。
    seed: int = 42
    # prefill chunk size
    prefill_chunk_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# dense/sparse bypass
def set_sparse_mode(model: Qwen2ForCausalLM, sparse: bool):
    """
    在同一模型上切换稀疏/稠密模式，实现单因子对照。
    sparse=True  -> bypass=False -> indexer 正常 top-k 选择 -> decode 分支走 gather 稀疏 matmul
    sparse=False -> bypass=True  -> decode 分支直接退回对全部历史 key 的 dense matmul
    """
    for layer in model.model.layers:
        layer.self_attn.indexer.bypass = not sparse
        # 关键：bypass 切换时清空 Indexer 自己的 k_cache，避免上一次 prefill 留下的缓存
        # 形状（来自不同 seq_len）污染下一次测量。
        layer.self_attn.indexer.k_cache = None


# baseline
@dataclass
class BenchResult:
    seq_len: int                # prefill 后 KV cache 中的历史长度（decode 开始时的 S_total）
    mode: str                   # "sparse" or "dense"
    avg_decode_latency_ms: float  # 每个 decode step 的平均墙钟耗时（毫秒）
    decode_throughput_tps: float  # decode 阶段吞吐：tokens / 秒（单 token 生成，所以约等于 1000/latency_ms）
    steps: int


def run_prefill_full(model, input_ids, attention_mask, device):
    """
    跑一次 prefill，返回建立好的 past_key_values（KV cache），供后续 decode 复用。
    prefill 走的是 model.py 中 seqlen > 1 的分支，这一步本身不计时
    （我们只关心 decode 阶段的效果，prefill 阶段没有实现 sparse attention）。
    """
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(input_ids.shape[1], device=device),
        )
    return outputs.past_key_values, outputs.logits


def run_prefill_chunked(model, input_ids, attention_mask, device, chunk_size=512):
    for layer in model.model.layers:
        layer.self_attn.chunked_prefill  = True   # 告知 attention 跳过 Indexer
        layer.self_attn.indexer.k_cache  = None   # 清空上一轮残留
 
    cache       = DynamicCache(config=model.config)
    seq_len     = input_ids.shape[1]
    last_logits = None

    with torch.no_grad():
        # we iterate for seq_len / chunk_size (s/c) times
        for start in range(0, seq_len, chunk_size):
            # for each round, 
            #   query length = chunk_size, KV length = end
            end        = min(start + chunk_size, seq_len)
            chunk_ids  = input_ids[:, start:end]
            
            # the attention memory is 
            #   batch_size * num_heads * num_layers * chunk_size * kv_length
            #   O(B * H * L * c * S_cached)

            # the peak attention memory is
            #   batch_size * num_heads * num_layers * chunk_size * seq_len
            #   O(B * H * L * c * S)

            # without chunking, full prefill attention memory is
            #  O(B * H * L * S * S)
            
            # KV Cache:
            #   batch_size * num_heads * num_layers * seq_len * head_dim
            #   O(B * H * L * S * d) = O(B * L * S * D)

            out = model(
                input_ids=chunk_ids,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(start, end, device=device),
            )
            last_logits = out.logits
    
    for layer in model.model.layers:
        layer.self_attn.chunked_prefill  = False
    
    return cache, last_logits


def bench_decode(model, prefill_ids, prefill_mask, device, decode_steps, warmup_steps, vocab_size, seed, chunk_size=512):
    """
    先 prefill 建立 KV cache，再连续 decode `decode_steps` 步，只对 decode 阶段计时。
    每一步用上一步采样得到的 token 作为下一步输入（标准自回归生成），
    用固定种子的伪随机采样保证 sparse/dense 两次运行路径一致、可比。
    """
    model.eval()
    rng = random.Random(seed)

    # Prefill：建立 KV cache（不计时）
    # time complexity: 
    #   batch_size * num_heads * num_layers * seq_len * seq_len * head_dim
    #   O(B * H * L * S * S * d)
    # attention memory:
    #   batch_size * num_heads * num_layers * chunk_size * seq_len
    # KV Cache:
    #   batch_size * num_heads * num_layers * seq_len * head_dim
    past_key_values, last_logits = run_prefill_chunked(model, prefill_ids, prefill_mask, device, chunk_size=chunk_size)
    cache_len_after_prefill = prefill_ids.shape[1]

    # 用 logits 采样下一个 token（固定种子 argmax 或简单取 top-1，保证可复现）
    # time complexity: O(vocab)
    next_token = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

    latencies = []

    with torch.no_grad():
        # 预热 decode 步（不计时）
        for _ in range(warmup_steps):
            # for each decoding step,
            # the time complexity of dense mode is
            #   batch_size * num_heads * num_layers * 1 * seq_len * head_dim
            #   O(B * H * L * 1 * S * d)
            # the time complexity of sparse mode is
            #   indexer scoring: O(B * H * L * 1 * S * d)
            #   top-k selection: O(B * L * 1 * S * logk)
            #   sparse attention: O(B * H * L * 1 * K * d)
            cache_position = torch.tensor([past_key_values.get_seq_length()], device=device)
            outputs = model(
                input_ids=next_token,
                attention_mask=None,   # decode 阶段不传 attention_mask，走 model.py 的 generate 分支
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        # 正式计时的 decode 步 
        for _ in range(decode_steps):
            cache_position = torch.tensor([past_key_values.get_seq_length()], device=device)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # for each decoding step,
            # the time complexity of dense mode is
            #   batch_size * num_heads * num_layers * 1 * seq_len * head_dim
            #   O(B * H * L * 1 * S * d)
            # the time complexity of sparse mode is
            #   indexer scoring: O(B * H * L * 1 * S * d)
            #   top-k selection: O(B * L * 1 * S * logk)
            #   sparse attention: O(B * H * L * 1 * K * d)

            outputs = model(
                input_ids=next_token,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            latencies.append(elapsed)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    total_time = sum(latencies)
    steps = len(latencies)
    avg_latency_ms = 1000 * total_time / steps if steps > 0 else 0.0
    # decode 阶段每步只生成 1 个 token，吞吐 = 1 token / 单步耗时
    throughput = 1.0 / (total_time / steps) if steps > 0 and total_time > 0 else 0.0

    return BenchResult(
        seq_len=cache_len_after_prefill,
        mode="",  # 由调用方填充
        avg_decode_latency_ms=avg_latency_ms,
        decode_throughput_tps=throughput,
        steps=steps,
    )


def build_prefill_inputs(seq_len: int, vocab_size: int, seed: int, device: str):
    """构造固定种子的合成 prefill 输入，sparse/dense 两次运行共享完全相同的上下文。"""
    # time complexity: O(s)
    # memory: O(s)
    rng = random.Random(seed)
    ids = torch.tensor(
        [[rng.randint(0, vocab_size - 1) for _ in range(seq_len)]],
        device=device,
    )
    mask = torch.ones_like(ids)
    return ids, mask


# main pipeline
def main():
    cfg = EvalConfig()
    print(f"device: {cfg.device}")
    print(f"prefill length: {cfg.seq_lengths}")
    print(f"prefill chunk_size: {cfg.prefill_chunk_size}")
    print(f"每个长度 decode 步数: {cfg.decode_steps}（预热 {cfg.warmup_decode_steps} 步）")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    vocab_size = tokenizer.vocab_size

    print(f"加载模型: {cfg.model_path}")
    model = Qwen2ForCausalLM.from_pretrained(cfg.model_path)
    model.to(cfg.device)

    # 防御性检查：确认 Indexer 含 bypass 属性（应用了 model_patch 的 model.py）。
    sample_indexer = model.model.layers[0].self_attn.indexer
    if not hasattr(sample_indexer, "bypass"):
        print("[警告] Indexer 不含 bypass 属性，请确认 model.py 已应用方案A修复。")
        for layer in model.model.layers:
            layer.self_attn.indexer.bypass = False

    index_topk = getattr(sample_indexer, "index_topk", 128)

    results: List[BenchResult] = []
    print("=" * 70)

    for seq_len in cfg.seq_lengths:
        print(f"Prefill Length = {seq_len}\n")

        dense_latencies, sparse_latencies = [], []
        dense_throughputs, sparse_throughputs = [], []

        for ctx_seed in range(NUM_CONTEXTS):
            print(f" Sample {ctx_seed}")
            prefill_ids, prefill_mask = build_prefill_inputs(
                seq_len, vocab_size, cfg.seed + ctx_seed, cfg.device
            )

            # Dense 模式：bypass=True，decode 分支直接对全部历史 key 做 matmul
            set_sparse_mode(model, sparse=False)
            dense_result = bench_decode(
                model, prefill_ids, prefill_mask, cfg.device,
                cfg.decode_steps, cfg.warmup_decode_steps, vocab_size, cfg.seed,
                chunk_size=cfg.prefill_chunk_size
            )
            dense_result.mode = "dense"
            dense_latencies.append(dense_result.avg_decode_latency_ms)
            dense_throughputs.append(dense_result.decode_throughput_tps)
            print(f"  [dense ] decode latency={dense_result.avg_decode_latency_ms:.4f} ms/token  "
                f"throughput={dense_result.decode_throughput_tps:.1f} tok/s")

            # Sparse 模式：bypass=False，decode 分支走 gather 出 K 个 key 的真稀疏 matmul
            set_sparse_mode(model, sparse=True)
            sparse_result = bench_decode(
                model, prefill_ids, prefill_mask, cfg.device,
                cfg.decode_steps, cfg.warmup_decode_steps, vocab_size, cfg.seed,
                chunk_size=cfg.prefill_chunk_size
            )
            sparse_result.mode = "sparse"
            sparse_latencies.append(sparse_result.avg_decode_latency_ms)
            sparse_throughputs.append(sparse_result.decode_throughput_tps)
            print(f"  [sparse] decode latency={sparse_result.avg_decode_latency_ms:.4f} ms/token  "
                f"throughput={sparse_result.decode_throughput_tps:.1f} tok/s")

        avg_dense = BenchResult(
            seq_len=seq_len,
            mode="dense",
            avg_decode_latency_ms=sum(dense_latencies) / NUM_CONTEXTS,
            decode_throughput_tps=sum(dense_throughputs) / NUM_CONTEXTS,
            steps=cfg.decode_steps,
        )
        avg_sparse = BenchResult(
            seq_len=seq_len,
            mode="sparse",
            avg_decode_latency_ms=sum(sparse_latencies) / NUM_CONTEXTS,
            decode_throughput_tps=sum(sparse_throughputs) / NUM_CONTEXTS,
            steps=cfg.decode_steps,
        )

        results.append(avg_dense)
        results.append(avg_sparse)
        
        speedup = avg_sparse.decode_throughput_tps / avg_dense.decode_throughput_tps \
            if avg_dense.decode_throughput_tps > 0 else float("nan")
        print(f"\n  decode 阶段 average speedup (sparse/dense) over {NUM_CONTEXTS}: {speedup:.3f}x   "
              f"(理论 K/S = {index_topk}/{seq_len} = {index_topk/seq_len:.4f})")
        print("-" * 70)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'prefill_len':>11}  {'dense ms/tok':>13}  {'sparse ms/tok':>14}  "
          f"{'speedup':>8}  {'理论K/S':>8}")
    print("-" * 70)
    for i in range(0, len(results), 2):
        d, s = results[i], results[i + 1]
        speedup = s.decode_throughput_tps / d.decode_throughput_tps if d.decode_throughput_tps > 0 else float("nan")
        theoretical_ratio = index_topk / d.seq_len
        print(f"{d.seq_len:>11}  {d.avg_decode_latency_ms:>13.4f}  {s.avg_decode_latency_ms:>14.4f}  "
              f"{speedup:>7.3f}x  {theoretical_ratio:>7.4f}")
    print("=" * 70)
    print("speedup > 1 表示稀疏 decode 比稠密 decode 更快（净加速）。")
    print(f"理论K/S列：index_topk={index_topk}，K/S 越小代表理论上稀疏的计算量优势越大。")
    print("预期趋势：随着 prefill_len(即 S_total) 增大，gather 出的 K 个 key 相对全部")
    print("历史 key 的占比越来越小，speedup 应随 prefill_len 增长而持续提升，并在")
    print("prefill_len 远大于 index_topk 时趋于稳定的净加速（speedup > 1）。")


if __name__ == "__main__":
    main()