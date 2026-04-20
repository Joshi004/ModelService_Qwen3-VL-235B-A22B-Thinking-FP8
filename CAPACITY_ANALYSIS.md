# Capacity Analysis: Qwen3-VL-235B-A22B-Thinking-FP8

## 1. Your Current Hardware

| Component | Details |
|-----------|---------|
| **GPUs** | 8x NVIDIA H100 80GB HBM3 |
| **Total VRAM** | 652 GB (8 x 81,559 MiB) |
| **CPU** | Intel Xeon SapphireRapids, 160 cores @ 2.0 GHz |
| **System RAM** | 983 GB (DDR5) |
| **CUDA Version** | 12.8 (driver 570.211.01) |

Each GPU has 81,559 MiB (~80 GB) of HBM3 memory. Right now, 77,549 MiB (~76 GB) per GPU is
used by the model and runtime, leaving about 4 GB free per GPU. This is expected because
`gpu_memory_utilization=0.90` tells vLLM to use up to 90% of each GPU.


## 2. Your Current Configuration

| Setting | Value | What It Does |
|---------|-------|--------------|
| **vLLM Version** | 0.15.1 (V1 engine) | The serving framework |
| **Tensor Parallel** | 8 (all GPUs) | Splits the model across all 8 GPUs |
| **Expert Parallel** | Enabled (16 experts per GPU) | 128 total experts split across 8 GPUs |
| **Max Context Length** | 131,072 tokens (128K) | Maximum input+output per request |
| **Max Concurrent Requests** | 2 (max_num_seqs) | How many requests run at the same time |
| **GPU Memory Utilization** | 0.90 (90%) | Fraction of VRAM vLLM is allowed to use |
| **Chunked Prefill** | Enabled, batch size 8,192 | Processes prompts in 8K-token chunks |
| **Prefix Caching** | Enabled | Reuses cached prompt tokens across requests |
| **Encoder Cache Budget** | 16,384 tokens | Cache for vision encoder outputs |


## 3. How GPU Memory Is Divided (The Big Picture)

When vLLM starts, each GPU's memory is split into three parts:

```
  ┌──────────────────────────────────────┐
  │         Model Weights: ~28 GB        │  ← Fixed, always occupied
  ├──────────────────────────────────────┤
  │         KV Cache: ~40 GB             │  ← Grows/shrinks with active requests
  ├──────────────────────────────────────┤
  │     CUDA Graphs + Overhead: ~12 GB   │  ← Runtime overhead
  └──────────────────────────────────────┘
             Per GPU (~80 GB total)
```

From your logs:
- **Model loading**: 28.24 GB per GPU
- **Available KV cache**: 40.07 GB per GPU
- **Total KV cache capacity**: 894,064 tokens (shared across ALL concurrent requests)
- **CUDA graphs + overhead**: ~12 GB per GPU

The KV cache is the "working memory" that holds the state of every token the model has
seen or generated in all active requests. This is the key resource that determines how
many requests you can run and how large they can be.


## 4. How KV Cache Works (The Core Concept)

Every token that enters or exits the model needs a "Key" and "Value" stored in GPU memory.
This is called the KV cache.

**Per-token KV cache cost for your model:**

```
KV bytes per token per GPU = 2 (K and V) x 1 (KV heads per GPU) x 128 (head_dim) x 94 (layers) x 2 (bfloat16 bytes)
                           = 48,128 bytes
                           ≈ 47 KB per token per GPU
```

Your model has only 4 KV heads (Grouped Query Attention), which is why the per-token cost
is relatively low despite having 94 layers. With 8 GPUs, each GPU handles 1 KV head.

**Your total KV budget: 894,064 tokens.**

This is a shared pool. If you have 2 requests each using 50,000 tokens, that's 100,000
tokens total — about 11% of your KV cache.

**The critical formula:**

```
Total tokens used = Sum of (input_tokens + generated_tokens_so_far) across all active requests
This must be ≤ 894,064 tokens at all times
```


## 5. Model's Context Length Capabilities

| Context Level | Tokens | Status |
|---------------|--------|--------|
| **Your current config** | 131,072 (128K) | What you're running now |
| **Model's native max** | 262,144 (256K) | Supported without any tricks |
| **Extended (with RoPE scaling)** | Up to 1M | Experimental, quality degrades |

Your current 128K setting is a good choice. It's half the native maximum, which gives
the KV cache room to serve multiple requests. Going to 256K would mean each single
request could consume up to 256K of your 894K token budget — leaving room for only
~3.4 concurrent max-length requests instead of ~6.8.

**Can you change it?** Yes. Edit `MAX_MODEL_LEN` in `config.env` and restart. But
increasing it means each request *can* use more memory, reducing how many can run
in parallel.


## 6. Use Case 1 — Transcript Analysis (Finding Moments)

### 6a. How Big Is a Transcript in Tokens?

English speech averages ~150 words per minute. The tokenizer converts roughly
1.3 tokens per word.

| Video Length | Words | Tokens (text only) | With Timestamps + Metadata | + System Prompt + Output |
|-------------|-------|---------------------|----------------------------|--------------------------|
| **1 hour** | ~9,000 | ~12,000 | ~15,000-20,000 | ~20,000-28,000 |
| **2 hours** | ~18,000 | ~23,000 | ~30,000-40,000 | ~35,000-48,000 |
| **3 hours** | ~27,000 | ~35,000 | ~45,000-55,000 | ~50,000-63,000 |
| **5 hours** | ~45,000 | ~58,000 | ~70,000-85,000 | ~78,000-93,000 |
| **6 hours** | ~54,000 | ~70,000 | ~85,000-100,000 | ~93,000-108,000 |

The "With Timestamps + Metadata" column accounts for segment-level timestamps
(like `[00:05:30 - 00:05:45] Speaker A: "..."`) which add ~50-70% overhead to
raw word count.

"+ System Prompt + Output" adds your instruction (~500-1,000 tokens) and the
model's response (up to 8,192 tokens as currently configured, but likely
2,000-4,000 for moment identification).

### 6b. Question 1: How Many 1-Hour Transcripts Can Run in Parallel?

**From a memory (KV cache) perspective:**

Each 1-hour transcript request uses ~20,000-28,000 tokens total.

```
KV cache budget:  894,064 tokens
Per request:       ~25,000 tokens (average estimate)
Memory limit:     894,064 / 25,000 ≈ 35 concurrent requests
```

Memory is not the bottleneck here. You could theoretically fit ~35 such requests.

**From a compute perspective:**

Your observed throughput from the logs:
- 1 request: ~63 tokens/s generation speed
- 2 requests: ~61 tokens/s each (~122 total) — barely any slowdown!

This is because Qwen3-VL-235B is a Mixture-of-Experts model. Only 22B of the 235B
parameters are active per token. The H100 GPUs have plenty of compute headroom to
batch multiple requests without slowing down significantly.

Estimated generation speeds at different concurrency levels:

| Concurrent Requests | Est. Per-Request Speed | Est. Total Throughput |
|---------------------|------------------------|-----------------------|
| 1 | ~63 tok/s | ~63 tok/s |
| 2 | ~61 tok/s | ~122 tok/s |
| 4 | ~55-58 tok/s | ~220-232 tok/s |
| 6 | ~45-50 tok/s | ~270-300 tok/s |

These are estimates. At 4-6 concurrent, the per-request speed will drop somewhat
due to memory bandwidth saturation, but total throughput increases.

**Your current bottleneck: `max_num_seqs=2`**

Right now, even though you have memory and compute for more, you've limited the
system to 2 concurrent requests. Your logs confirm this — when a 3rd request
arrives while 2 are running, it goes to a "Waiting" queue.

**Recommendation for 1-hour transcripts:**

You can safely raise `max_num_seqs` to **4 or 6** for this use case. The KV cache
will still be under 20% utilized, and the MoE architecture handles batching well.

To do this, change in `config.env`:
```
MAX_NUM_SEQS=4    # or 6 for higher throughput
```

And update the `vllm serve` command in `start_service.sh` if needed, then restart.

**Practical throughput estimate with 4 parallel 1-hour transcripts:**
- Prefill per request: ~20,000 tokens / 1,400 tok/s ≈ 14 seconds
- Generation of ~4,000 tokens: ~4,000 / 57 ≈ 70 seconds
- Total per request: ~84 seconds (~1.4 minutes)
- Total throughput: ~4 transcripts per 1.5 minutes ≈ 160 transcripts/hour

### 6c. Question 2: How Big of a Single Transcript Can You Send?

**Hard limit: 131,072 tokens** (your configured max_model_len).

But not all of that is for input. The model also needs room for its output.

```
Max usable input = max_model_len - expected_output_tokens
                 = 131,072 - 8,192 (your MAX_TOKENS)
                 = 122,880 tokens for input
```

**What does 122,880 input tokens translate to?**

| Content Type | Size in 122,880 Tokens |
|-------------|------------------------|
| Pure words | ~94,000 words |
| Plain transcript (no timestamps) | ~7-8 hours of speech |
| Segment-level transcript (with timestamps) | ~5-6 hours of speech |
| Dense transcript (word-level with timestamps) | ~3-4 hours of speech |

**So for your use case: You can send a single transcript of up to ~5-6 hours
(segment-level) in one request, within your current 128K context limit.**

For longer content, you have two options:
1. **Increase MAX_MODEL_LEN to 262,144 (256K)** — this would handle ~12 hours
   of segment-level transcript, but reduces maximum concurrency from ~6.8x to ~3.4x.
2. **Split and process in chunks** — divide the transcript into overlapping
   segments (e.g., 2-3 hour chunks with 10-minute overlap) and process sequentially.

**Time estimate for a 5-hour transcript (single request):**
- Input tokens: ~80,000
- Prefill time: ~80,000 / 1,400 ≈ 57 seconds
- Generation (~8,000 tokens of moment analysis): ~8,000 / 63 ≈ 127 seconds
- Total: ~184 seconds ≈ 3 minutes

**KV cache impact:** ~88,000 tokens used ≈ 9.8% of your KV budget. Even with a
massive single request, you're using less than 10% of available memory.


## 7. Use Case 2 — Video + Transcript Analysis (Moment Deep-Dive)

### 7a. How Video Gets Converted to Tokens

When you send a video to Qwen3-VL, here's what happens internally:

```
Your 30fps video
      ↓
  Sampled at 2 FPS (default)        ← Model takes 2 frames per second, ignores the rest
      ↓
  Frames resized (smart_resize)     ← Resolution adjusted for efficiency
      ↓
  ViT processes patches             ← Each frame split into 16x16 pixel patches
      ↓
  Temporal merge (pairs of 2)       ← Every 2 consecutive frames merged
      ↓
  Spatial merge (2x2)               ← 4 adjacent patches merged into 1 token
      ↓
  Video tokens                      ← Fed into the language model alongside text
```

**Token math for a 5-minute video:**

```
Step 1: Frame sampling
  5 min × 60 sec × 2 FPS = 600 frames sampled (from 9,000 at 30fps)

Step 2: Temporal merging (temporal_patch_size = 2)
  600 frames / 2 = 300 temporal units

Step 3: Spatial tokens per temporal unit
  Depends on resolution. For a typical 720p video after smart_resize:
  Effective pixels per token = 16 (patch) × 2 (spatial_merge) = 32 pixels per side
  At ~448×256 resolution: (448/32) × (256/32) = 14 × 8 = 112 tokens per unit

Step 4: Total video tokens
  300 temporal units × 112 tokens = 33,600 tokens
```

In practice, the model/vLLM may cap video tokens at around 24,576 tokens (a common
default in qwen-vl-utils). So for a 5-minute video, expect **~24,000-34,000 video
tokens** depending on resolution.

### 7b. Total Token Count per Request (Use Case 2)

| Component | Tokens |
|-----------|--------|
| Video (5 min at 2 FPS) | ~24,000-34,000 |
| Word-level transcript (5 min) | ~2,000-3,000 |
| System prompt + question | ~500-1,000 |
| Model output (moment analysis) | ~2,000-4,000 |
| **Total per request** | **~28,000-42,000** |

Let's use ~35,000 as a working average.

### 7c. How Many Parallel Video Requests Can You Handle?

**From KV cache perspective:**

```
KV budget:    894,064 tokens
Per request:   ~35,000 tokens
Memory limit: 894,064 / 35,000 ≈ 25 concurrent requests
```

**From encoder cache perspective:**

The encoder cache budget is 16,384 tokens. This limits how many video tokens
can be processed through the vision encoder at once. With each video having
~24,000-34,000 tokens, only 1 video can be processed through the encoder at
a time. However, once a video's visual features are encoded, they go into the
KV cache and free up the encoder for the next video.

In practice, this means video requests are partially serialized through the
encoder stage but can overlap during generation.

**From compute perspective:**

Video requests are heavier than text-only requests because:
1. The vision encoder (ViT with 27 layers) must process all video frames
2. The prefill phase has more tokens to process
3. Each request has a larger KV cache footprint

Estimated performance:

| Concurrent Requests | Est. Per-Request Speed | Notes |
|---------------------|------------------------|-------|
| 1 | ~60-63 tok/s | Same as text-only |
| 2 | ~55-60 tok/s each | Slight slowdown from larger KV |
| 3-4 | ~45-50 tok/s each | Noticeable but acceptable |

**Recommendation: 2-3 parallel video+transcript requests** is the sweet spot
with your current hardware. The encoder cache (16,384 tokens) and the fact
that video encoding is sequential per request make 2 concurrent the practical
optimum.

**Time estimate per video request:**
- Video encoding (ViT processing): ~10-20 seconds
- Prefill (35,000 tokens): ~35,000 / 1,200 ≈ 29 seconds
- Generation (~3,000 tokens): ~3,000 / 60 ≈ 50 seconds
- Total: ~90-100 seconds per request (~1.5 minutes)
- With 2 parallel: ~2 requests per 1.5 minutes ≈ 80 requests/hour


## 8. Observed Real Performance (From Your Logs)

Your service logs show actual production behavior. Here's what the numbers say:

| Metric | 1 Request Running | 2 Requests Running |
|--------|-------------------|--------------------|
| **Generation speed** | 60-64 tok/s | 118-122 tok/s total (~60 each) |
| **Prompt throughput** | 1,200-1,400 tok/s | 2,400-4,000 tok/s |
| **KV cache usage** | 1-3% | 2-5% |
| **Per-request slowdown** | Baseline | ~3-5% slower per request |

Key observations:
1. **MoE advantage**: Going from 1 to 2 concurrent requests barely reduces
   per-request speed. This is because only 22B of 235B parameters are active,
   leaving lots of compute headroom.

2. **KV cache is vastly underutilized**: At 2-5% usage with 2 requests, you
   have enormous room for either larger inputs or more concurrent requests.

3. **The bottleneck is `max_num_seqs=2`**: Your logs show requests going to
   "Waiting" state when a 3rd arrives. This is an artificial limit, not a
   hardware limit.

4. **Prefix caching works**: The logs show hit rates of 15-29% over time,
   meaning repeated similar prompts (same system prompt, same task instructions)
   are being cached and reused. This saves time on the prefill phase.


## 9. The Capacity Model — How to Calculate These Things Yourself

### The Formula

For any request, estimate its total token footprint:

```
request_tokens = input_tokens + max_output_tokens
```

For total system capacity:

```
max_concurrent_requests = min(
    max_num_seqs,                                   ← Config limit
    KV_cache_tokens / average_request_tokens,        ← Memory limit
    target_generation_speed / min_acceptable_speed   ← Compute limit
)
```

For your system with the current config:

```
max_concurrent = min(
    2,                          ← max_num_seqs (your bottleneck!)
    894,064 / request_tokens,   ← Memory (generous)
    varies                      ← Compute (generous for MoE)
)
```

### Step-by-Step: How to Estimate for Any New Workload

**Step 1: Estimate input tokens**
- Text: words × 1.3 (for English). Add 50-70% for timestamps/metadata.
- Video: (duration_sec × 2 FPS / 2) × tokens_per_temporal_unit
  Where tokens_per_temporal_unit ≈ 100-120 for standard resolution
- Images: roughly (H/32) × (W/32) tokens per image

**Step 2: Estimate output tokens**
- Short answer: 500-2,000 tokens
- Detailed analysis: 2,000-4,000 tokens
- Comprehensive report: 4,000-8,000 tokens

**Step 3: Check against limits**
- Total (input + output) must be ≤ max_model_len (131,072)
- Sum of all concurrent request totals must be ≤ 894,064 (KV cache)
- Number of concurrent requests ≤ max_num_seqs

**Step 4: Estimate time**
- Prefill time ≈ input_tokens / 1,300 (seconds, for single request)
- Generation time ≈ output_tokens / 60 (seconds, per request)
- Total ≈ prefill + generation


## 10. Scaling — What to Do When You Need More

### Option A: Tune Current Config (Free, Immediate)

What you can change right now without any hardware changes:

| Change | Effect | Risk |
|--------|--------|------|
| Raise `max_num_seqs` from 2 to 4-6 | More concurrent requests | Slight per-request slowdown |
| Raise `gpu_memory_utilization` from 0.90 to 0.95 | ~10% more KV cache | Slightly higher OOM risk |
| Lower `MAX_MODEL_LEN` to 65536 | More concurrent requests possible | Can't handle very long inputs |
| Raise `MAX_MODEL_LEN` to 262144 | Handle longer inputs | Fewer concurrent requests |

**Best immediate change for your use cases:**

For Use Case 1 (transcripts only), change `MAX_NUM_SEQS=4` or `MAX_NUM_SEQS=6`.
You have the memory and compute for it.

For Use Case 2 (video), `MAX_NUM_SEQS=2` or `3` is fine since video encoding
is the bottleneck.

### Option B: Vertical Scaling (More/Better GPUs on Same Machine)

| Upgrade | What It Gets You |
|---------|-----------------|
| **8x H100 → 8x H200 (141 GB each)** | ~75% more VRAM = ~75% more KV cache ≈ ~1.56M tokens. Nearly double the concurrency. |
| **Add more GPUs (if motherboard supports)** | Not practical — you're already at 8 GPUs with TP=8. |

**Verdict**: Vertical scaling through H200 would help if you need much higher
concurrency or much longer context per request. But it's expensive and you're
currently using only 2-5% of your KV cache.

### Option C: Horizontal Scaling (Multiple Machines)

This is the right answer when you need **more total throughput** (more requests
per hour) rather than longer context or higher concurrency on a single request.

```
Machine 1: vLLM instance (8x H100) → handles requests 1-N
Machine 2: vLLM instance (8x H100) → handles requests N+1-2N
    ↑
Load balancer distributes requests across instances
```

Each machine runs an independent vLLM instance. You put a load balancer
(nginx, HAProxy, or a simple Python round-robin) in front.

| # of Machines | Total Concurrent Requests | Total Throughput |
|---------------|---------------------------|------------------|
| 1 (current) | 2-6 | ~63-300 tok/s |
| 2 | 4-12 | ~126-600 tok/s |
| 4 | 8-24 | ~252-1200 tok/s |

**When to horizontally scale:**
- When you need more than ~6 concurrent text requests or ~3 video requests
- When you need total throughput beyond what 1 machine can deliver
- When you want redundancy (one machine can go down)

**When to vertically scale:**
- When you need longer context per request (beyond 128K or 256K)
- When you need the absolute lowest latency for a single request
- When individual requests are very large (huge videos, very long transcripts)

### Option D: Data Parallel Mode in vLLM

vLLM supports `data_parallel_size` which can split your 8 GPUs differently.
For example, instead of 8-way tensor parallel for one big model instance,
you could run 2 copies of the model, each using 4 GPUs:

```
Current:  TP=8, DP=1  → 1 model copy across 8 GPUs
Option:   TP=4, DP=2  → 2 model copies, each across 4 GPUs
```

With TP=4 for this FP8 model (~236 GB weights ÷ 4 GPUs = ~59 GB per GPU),
each GPU would have 59 GB for weights + overhead, leaving very little for
KV cache (~10-15 GB per GPU). This would severely limit context length and
concurrency per instance.

**Verdict**: TP=8, DP=1 is the right choice for this model. The MoE
architecture already handles batching efficiently, and you need the KV cache
space more than you need model replicas.


## 11. Summary: Direct Answers to Your Questions

### Use Case 1: Transcript Analysis

**Q: How many 1-hour transcripts can I process in parallel?**

With current config (max_num_seqs=2): **2 at a time**, with others queued.
If you raise max_num_seqs to 4-6: **4-6 at a time** comfortably.
Memory can support up to ~35, but compute will cap practical throughput at 6-8.

**Q: How long of a transcript can I send in a single request?**

A segment-level transcript of **up to ~5-6 hours** fits within your 128K context.
If you increase MAX_MODEL_LEN to 256K, you can handle **~12 hours** in one request.
A word-level transcript (more verbose) would cap around **~3-4 hours** at 128K.

### Use Case 2: Video + Transcript Analysis

**Q: How many 5-minute video+transcript requests can I process in parallel?**

With current config: **2 at a time** (the max_num_seqs limit).
Could safely go to **3** by raising max_num_seqs=3.
Beyond 3, the vision encoder becomes the bottleneck.

Throughput: approximately **80 video analysis requests per hour** with 2 concurrent.

### Scaling Decisions

| Need | Solution |
|------|----------|
| More concurrent short requests | Raise max_num_seqs to 4-6 (free!) |
| Longer single transcripts | Raise MAX_MODEL_LEN to 256K (trade-off: less concurrency) |
| More total throughput | Add another machine (horizontal scaling) |
| Much longer context (>256K) | Upgrade to H200 GPUs (vertical scaling) |
| Redundancy / fault tolerance | Add another machine with load balancer |


## 12. Quick Reference: Your System's Key Numbers

| Metric | Value |
|--------|-------|
| Total VRAM | 652 GB |
| Model weights (per GPU) | ~28 GB |
| KV cache (per GPU) | ~40 GB |
| Total KV cache tokens | 894,064 |
| KV cost per token (per GPU) | ~47 KB |
| Max context length (configured) | 131,072 tokens |
| Max context length (model native) | 262,144 tokens |
| Max concurrent requests (configured) | 2 |
| Generation speed (1 request) | ~63 tok/s |
| Generation speed (2 requests) | ~61 tok/s each |
| Prompt processing speed | ~1,200-1,400 tok/s |
| 1 hour transcript | ~15,000-20,000 tokens |
| 5 min video (2 FPS, merged) | ~24,000-34,000 tokens |
| Time for 1-hour transcript request | ~80 seconds |
| Time for 5-min video request | ~90-100 seconds |
