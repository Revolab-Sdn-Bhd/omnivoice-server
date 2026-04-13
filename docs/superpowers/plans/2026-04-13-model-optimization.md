# OmniVoice Model Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize OmniVoice TTS inference speed and throughput using AITune, torch.compile, quantization, and server-level batching.

**Architecture:** OmniVoice uses Qwen3-0.6B (28 layers, hidden_size=1024, 3072 FFN, 16 heads, 8 KV heads) as its LLM backbone. The iterative decoder calls `self.llm()` 32 times per request (configurable via `num_step`). The LLM forward is ~90% of compute. Server runs on NVIDIA H200 NVL (139.8 GB VRAM, compute cap 9.0, FP8/BF16 capable).

**Tech Stack:** PyTorch 2.11+cu130, AITune (to install), TorchAO (to install), TensorRT (to install), torch.compile/inductor

---

## Environment Check Results

| Component | Status |
|-----------|--------|
| GPU | NVIDIA H200 NVL, 139.8 GB VRAM |
| Compute Cap | 9.0 (Hopper) - FP8, BF16 capable |
| PyTorch | 2.11.0+cu130 |
| LLM Backbone | Qwen3-0.6B (28L, 1024H, 3072FFN, 16AH, 8KVH) |
| INT8 Model | Exists at /mnt/data/work/omnivoice-tts/models/OmniVoice_INT8/ |
| TensorRT | NOT installed |
| TorchAO | NOT installed |
| AITune | NOT installed |

---

## Phase 1: Quick Wins (No Dependencies)

### Task 1: Benchmark current baseline

**Files:**
- Create: `benchmark_optimize.py`

- [ ] **Step 1: Write baseline benchmark script**

Create a script that measures single-request latency and throughput for the current
unoptimized model. Must measure:
- Time per LLM forward call (inside _generate_iterative)
- Total generate() wall time
- Audio duration
- RTF (Real-Time Factor)
- TTFA (Time To First Audio)

Run against the existing server at localhost:8880 with num_step=32, 16, 8.

- [ ] **Step 2: Run baseline benchmark**

Run: `python benchmark_optimize.py --url http://localhost:8880 --num-step 32 16 8`
Expected: RTF numbers for each num_step setting. Establish baseline.

- [ ] **Step 3: Commit baseline results**

```bash
git add benchmark_optimize.py
git commit -m "feat: add optimization benchmark script"
```

### Task 2: Benchmark num_step reduction

**Files:**
- Modify: `benchmark_optimize.py`

- [ ] **Step 1: Benchmark with num_step=16 and num_step=8**

Test quality vs speed tradeoff at each step count.
The server already supports --num-step flag.

- [ ] **Step 2: Document quality vs speed findings**

Record which num_step values produce acceptable quality with measurable speedup.

---

## Phase 2: torch.compile on LLM Backbone

### Task 3: Add torch.compile option to ModelService

**Files:**
- Modify: `omnivoice_server/config.py` - add compile_mode setting
- Modify: `omnivoice_server/services/model.py` - compile LLM after loading

- [ ] **Step 1: Add compile_mode config option**

Add to Settings:
```python
compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = "none"
compile_llm_only: bool = True
```

- [ ] **Step 2: Apply torch.compile after model load**

In ModelService._load_sync(), after successful model load:
```python
if self.cfg.compile_mode != "none":
    target = model.llm if self.cfg.compile_llm_only else model
    target = torch.compile(target, mode=self.cfg.compile_mode)
```

- [ ] **Step 3: Test with torch.compile**

Start server with `--compile-mode max-autotune` and verify inference works.
First request triggers compilation (slow), subsequent requests should be faster.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add torch.compile option for LLM backbone"
```

### Task 4: Benchmark torch.compile vs baseline

**Files:**
- Modify: `benchmark_optimize.py`

- [ ] **Step 1: Run benchmark with torch.compile max-autotune**

Compare RTF against baseline. Warm up first (compilation overhead).

- [ ] **Step 2: Test each compile mode**

Test: default, reduce-overhead, max-autotune. Record results.

---

## Phase 3: Install AITune + TorchAO + TensorRT

### Task 5: Install optimization dependencies

**Files:**
- Modify: `pyproject.toml` - add optional dependencies

- [ ] **Step 1: Install aitune with dependencies**

```bash
uv pip install --extra-index-url https://pypi.nvidia.com aitune
```

Verify:
```python
import aitune, torchao, tensorrt
```

- [ ] **Step 2: Add optional dependency group to pyproject.toml**

```toml
[project.optional-dependencies]
optimize = ["aitune", "torchao>=0.13", "tensorrt>=10.5"]
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add optimization dependencies"
```

---

## Phase 4: TorchAO Quantization

### Task 6: Add TorchAO FP8/INT8 quantization option

**Files:**
- Modify: `omnivoice_server/config.py` - add quantization setting
- Modify: `omnivoice_server/services/model.py` - apply quantization

- [ ] **Step 1: Add quantization config option**

```python
quantization: Literal["none", "fp8wo", "fp8dq", "int8wo", "int8dq"] = "none"
```

- [ ] **Step 2: Apply TorchAO quantization after model load**

```python
from torchao.quantization import quantize_, Float8WeightOnlyConfig

if self.cfg.quantization == "fp8wo":
    quantize_(model.llm, config=Float8WeightOnlyConfig())
```

- [ ] **Step 3: Test FP8 weight-only quantization**

Start server with `--quantization fp8wo`, verify output quality.

- [ ] **Step 4: Benchmark quantized model**

Compare RTF vs baseline.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add TorchAO quantization support"
```

---

## Phase 5: AITune Integration

### Task 7: Add AITune AOT tuning to LLM backbone

**Files:**
- Modify: `omnivoice_server/config.py` - add aitune settings
- Modify: `omnivoice_server/services/model.py` - integrate aitune wrap/tune/load
- Create: `scripts/tune_model.py` - standalone tuning script

- [ ] **Step 1: Write standalone tuning script**

Create `scripts/tune_model.py` that:
1. Loads OmniVoice model
2. Wraps `model.llm` with `ait.Module()`
3. Creates sample input data for the LLM forward pass
4. Tunes with FirstWinsStrategy (TensorRT -> TorchInductor -> TorchAO)
5. Saves tuned artifact

- [ ] **Step 2: Add config for loading pre-tuned model**

```python
aitune_checkpoint: str | None = None  # path to .ait file
```

- [ ] **Step 3: Add load path in ModelService**

If `aitune_checkpoint` is set, load the tuned model:
```python
import aitune.torch as ait
model.llm = ait.load(model.llm, cfg.aitune_checkpoint)
```

- [ ] **Step 4: Run tuning and benchmark**

```bash
python scripts/tune_model.py --model-path /path/to/model --output tuned_llm.ait
```

Then benchmark the tuned model vs baseline.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add AITune AOT tuning integration"
```

---

## Phase 6: Server-Level Request Batching

### Task 8: Add batched inference at server level

**Files:**
- Modify: `omnivoice_server/services/inference.py` - add request coalescing
- Modify: `omnivoice_server/config.py` - add batch config

- [ ] **Step 1: Add batch configuration**

```python
batch_enabled: bool = False
batch_max_size: int = 4
batch_timeout_ms: int = 50
```

- [ ] **Step 2: Implement request coalescing**

Collect incoming requests for up to `batch_timeout_ms` or `batch_max_size`,
then call `model.generate()` with a list of texts in one batch.
OmniVoice already supports batch inference via GenerationTask.

- [ ] **Step 3: Benchmark batched vs single inference**

Measure throughput (req/s) with 2, 4, 8 concurrent requests.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add server-level request batching"
```

---

## Phase 7: Final Benchmark + Results

### Task 9: Comprehensive benchmark of all optimization combinations

**Files:**
- Modify: `benchmark_optimize.py`

- [ ] **Step 1: Run full benchmark matrix**

Test combinations:
- Baseline (num_step=32)
- num_step=16
- num_step=8
- torch.compile max-autotune + num_step=16
- FP8 quantization + num_step=16
- AITune TensorRT + num_step=16
- Best combo + batch=4

- [ ] **Step 2: Document results**

Record RTF, TTFA, throughput, quality notes for each config.

- [ ] **Step 3: Update config defaults if clear winner**

Update default settings to the best-performing configuration.
