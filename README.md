# NimbusInfer-rs

[![ci](https://github.com/EAnge1o/nimbus-infer-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/EAnge1o/nimbus-infer-rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-blue.svg)](https://www.rust-lang.org/)
[![Dependency](https://img.shields.io/badge/crates-0-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NimbusInfer-rs** is the Rust sibling of [nimbus-infer](https://github.com/EAnge1o/nimbus-infer): a from-scratch LLM inference engine with **zero external crates** -- byte-level BPE tokenizer, KV cache, GPT-2 **and** LLaMA architectures, hand-written AVX2/FMA kernels, int8/int4 weight quantization, even its own JSON parser. No PyTorch, no candle, no rayon, no serde, no tokenizers.

> 一个大模型推理引擎的每个环节都可以被清晰地构建、测试和讲解。Python 版证明了这一点；Rust 版进一步回答：**在不借用任何现成库的前提下，手写的内核能离 OpenBLAS 有多近，量化能为内存受限的解码换来多少吞吐？**

它直接读取 nimbus-infer 的 checkpoint（`config.json` + `weights.safetensors` + `tokenizer/`），同一份模型文件在两个引擎里跑出**逐 token 一致**的贪心生成结果。

## 基准结果

三个真实 checkpoint，同一台机器，同一协议（128 token prefill + 64 token 贪心 decode，3 轮取最优）：

| 模型 | 引擎 | 量化 | 权重内存 | prefill tok/s | decode tok/s |
|---|---|---|---:|---:|---:|
| gpt2-124M | Python/NumPy | fp32 | 474.7MB | **349.9** | 32.8 |
| gpt2-124M | Rust | fp32 | 471.2MB | 273.6 | 35.1 |
| gpt2-124M | Rust | int8 | **118.3MB** | 267.5 | 57.7 |
| gpt2-124M | Rust | int4 | **62.6MB** | 244.6 | **60.8** |
| distilgpt2-82M | Python/NumPy | fp32 | 312.5MB | **640.1** | 58.1 |
| distilgpt2-82M | Rust | fp32 | 309.2MB | 594.0 | 63.3 |
| distilgpt2-82M | Rust | int8 | **77.7MB** | 541.7 | **92.5** |
| distilgpt2-82M | Rust | int4 | **41.1MB** | 487.7 | 89.6 |
| SmolLM2-135M (LLaMA) | Python/NumPy | fp32 | 513.1MB | **268.0** | 28.0 |
| SmolLM2-135M (LLaMA) | Rust | fp32 | 513.0MB | 120.3 | 35.4 |
| SmolLM2-135M (LLaMA) | Rust | int8 | **129.0MB** | 115.4 | **46.0** |
| SmolLM2-135M (LLaMA) | Rust | int4 | **71.3MB** | 104.3 | 42.6 |

**环境**：Intel i7-12700H（6P+8E，20 线程）、16GB DDR4、Windows 11、rustc stable (gnu)、numpy 2.2（OpenBLAS 后端）。每次运行有 ±10% 波动，表内为单次实测。

**读法**（这些数字诚实地告诉我们三件事）：

1. **解码是内存带宽问题，量化直接兑现为吞吐**。gpt2 的 decode 从 fp32 的 35.1 提升到 int8 的 57.7 / int4 的 60.8 tok/s（+73%），代价只是 4-bit 精度。int4 权重只有 fp32 的 1/7.5（62.6MB vs 471MB）。
2. **手写 fp32 内核在 decode 上与 OpenBLAS 打平或反超**（35.1 vs 32.8；distilgpt2 上 63.3 vs 58.1）-- 一个只有 3000 行、零依赖的实现。
3. **prefill 是 OpenBLAS 的主场**。它的 GEMM 有几十年的分块/寄存器调优，我们的逐点积 GEMM 在 prefill 上落后 15%~55%（模型越"瘦"差距越大）。这个差距是本仓库文档化的已知边界，不是隐藏的失败 -- 修复方向写在 [Roadmap](#roadmap) 里。

kernel 微基准（单次 50257×768 GEMV，等价 gpt2 的 lm_head）：

| kernel | GFLOPS | 说明 |
|---|---:|---|
| fp32 (AVX2+FMA) | 19.4 | 8-lane FMA，运行时检测 |
| int8 | 22.5 | 权重按字节流式反量化 |
| int4 | **29.2** | 半字节解包在寄存器内完成 |

## 跨引擎一致性（本项目最看重的性质）

`benchmarks/cross_engine.py` 用 Python 引擎生成随机小模型 + 真实权重模型，逐项验证两个引擎：

```
$ python benchmarks/cross_engine.py parity
prompt ids (7): [283, 309, 308, 272, 415, 286, 442]...
tokenizer parity: PASS      # 同一份 vocab.json + merges.txt，同一种分
logits parity:   PASS      # logits 逐值一致（<1e-6，BLAS 与手写 SIMD 的累加顺序差）
greedy parity:   PASS      # 32 个生成 token 逐个相同
PARITY: PASS
```

真实权重上同样成立：gpt2-124M 与 SmolLM2-135M 的贪心生成在两个引擎中**逐 token 相同**。CI 中有专门 job 复现以上验证（拉取 Python 引擎源码、共享 fixture、跑 parity）。

## 快速开始

```bash
git clone https://github.com/EAnge1o/nimbus-infer-rs.git
cd nimbus-infer-rs
cargo build --release          # 零依赖，无需任何 crate 下载

# 用 Python 引擎转一个 HuggingFace 模型（见 nimbus-infer 的 convert 命令），
# 或直接用仓库自带 fixtures/tiny-gpt2（随机权重，用于验证）：
./target/release/nimbus-rs generate --model tests/fixtures/tiny-gpt2 \
    --prompt "the engine caches keys and values" --max-new-tokens 24 --temperature 0

# 真实权重 + 量化推理
./target/release/nimbus-rs generate --model checkpoints/gpt2 \
    --prompt "The quick brown fox" --max-new-tokens 64 --temperature 0.8 --top-p 0.9 --seed 7 --stream
./target/release/nimbus-rs bench --model checkpoints/gpt2 --quant int4
./target/release/nimbus-rs inspect --model checkpoints/gpt2 --quant int4
./target/release/nimbus-rs bench-kernel          # 内核微基准
cargo test --release                             # 35 个测试
```

checkpoint 获取（Windows，国内网络用 `HF_ENDPOINT` 类镜像或预下载后 `--local`）：

```bash
git clone --depth 1 https://github.com/EAnge1o/nimbus-infer.git ../nimbus-infer
cd ../nimbus-infer
python -m nimbus_infer convert --repo gpt2 --out ../nimbus-infer-rs/checkpoints/gpt2
# SmolLM2 只有 fast-tokenizer JSON，先用脚本转出 vocab.json + merges.txt：
python ../nimbus-infer-rs/scripts/tokenizer_json_to_gpt2.py tokenizer.json tokenizer_dir/
```

## 设计笔记

**零依赖的代价与偿还。** 没有 serde 就手写 JSON 解析器（`json.rs`，含 `\uXXXX` 代理对）；没有 regex 就把 GPT-2 的预分词正则改写成手写状态机（`tokenizer.rs`，含 `\s+(?!\S)` 的回溯语义）；没有 rayon 就基于 `std::thread::scope` 写 fork-join（`threads.rs`）。总计约 3000 行。

**内存布局决定性能。** 所有线性层权重在加载时转置为 `(out, in)` 行主序，使每个输出元素都是一个**连续**的 SIMD 点积；KV cache 按 `(layer, head, pos, dim)` 排布让注意力读变成顺序流。prefill GEMM 按输出行并行、按序列顺序迭代，让整个激活面板驻留 L2、权重矩阵只从内存流过一次。

**按算子分级并行。** 解码 GEMV 是带宽受限的：超过 4 线程后 worker 之间在争抢 DRAM 而不是增加吞吐（i7-12700H 实测 4 线程比 20 线程快 18%）；prefill GEMM 是计算受限的，按总乘加量分块、以"不让整线程闲着"为下限。注意力扇出则按工作量决定是否值得开线程。

**int4 是真正的 4-bit。** 权重以半字节打包驻留内存（与 Python 版 `quantize()` 的位布局一致），解码时在寄存器内解包：8 字节经 `cvtepu8_epi32` 展宽、`unpacklo/hi + permute2f128` 恢复自然序、减偏置后 FMA。分组 scale 沿 K 维分段应用（128/64/32 自适应，SmolLM2 的 576 维会落到 64）。np.round 的银行家舍入被显式复现，保证与 Python 版量化结果可对比。

**一处有注释的 unsafe。** `UnsafeSlice`（threads.rs）是手写版的 `par_chunks_mut`：fork-join 的 worker 写入不相交的索引区间，安全性由分块构造保证，并以此换来零拷贝的并行写。这是整个 crate 唯一的 unsafe 写入点。

## 已知边界（诚实清单）

- prefill GEMM 落后 OpenBLAS（见上表），LLaMA 类瘦矩阵上更明显。
- SmolLM2 的预分词器对数字逐位切分（`Digits(individual_digits)`），本引擎统一使用 GPT-2 模式；含数字文本的分词可能与 HF 不一致（无数字文本已验证一致；bench 用合成 id 不受影响）。
- 采样用自实现的 xoshiro256++，不复现 numpy PCG64 的随机流（贪心解码不受影响，逐 token 一致）。
- Python 版的量化是"加载时反量化"，因此表里没有 Python 量化行可比；本仓库的量化内核是推理期流式反量化，这正是两者的性能差异来源。

## Roadmap

- [ ] 寄存器分块 GEMM（每 x 行 4~8 路累加），缩小 prefill 与 BLAS 的差距
- [ ] `Digits` 预分词模式，与 SmolLM2 完全对齐
- [ ] BF16 计算路径（当前仅加载时展宽）
- [ ] server 模式对齐 Python 版的 SSE 流式接口

## 同系列

[nimbus-infer](https://github.com/EAnge1o/nimbus-infer)（Python/NumPy 版，本仓库的数值规范来源）· [nimbus-vm](https://github.com/EAnge1o/nimbus-vm) · [nimbus-kv](https://github.com/EAnge1o/nimbus-kv) · [nimbus-db](https://github.com/EAnge1o/nimbus-db) · [nimbus-git](https://github.com/EAnge1o/nimbus-git) · [nimbus-chain](https://github.com/EAnge1o/nimbus-chain)

## License

MIT
