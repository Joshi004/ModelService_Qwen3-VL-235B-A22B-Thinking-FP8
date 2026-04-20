# How Your Model Service Works — A Complete Guide

This document explains how the Qwen3-VL-235B model service works on your hardware,
how to reason about its capacity, what your logs tell you, and how to plan for
your two use cases.


---


## Part 1: Your Setup at a Glance

**Hardware**

| | Details |
|--|---------|
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| Total VRAM | 652 GB |
| CPU | Intel Xeon SapphireRapids, 160 cores |
| System RAM | 983 GB |

**Software**

| | Details |
|--|---------|
| Framework | vLLM 0.15.1 (V1 engine) |
| Model | Qwen3-VL-235B-A22B-Thinking-FP8 |
| Quantization | FP8 (1 byte per weight parameter) |

**Key Config Values**

| Setting | Value | What it does |
|---------|-------|-------------|
| tensor_parallel_size | 8 | Model split across all 8 GPUs |
| max_model_len | 131,072 | Maximum tokens per single request |
| max_num_seqs | 2 | Maximum concurrent requests |
| gpu_memory_utilization | 0.90 | vLLM can use up to 90% of each GPU |
| enable_expert_parallel | true | 128 experts distributed 16-per-GPU |


---


## Part 2: How GPU Memory Is Divided

When vLLM starts, it does three things with each GPU's 80 GB:

1. **Loads model weights** — 28.24 GB per GPU. These are fixed and never change.
2. **Allocates KV cache** — 40.07 GB per GPU. This is the working memory for active requests.
3. **Reserves overhead** — ~12 GB for CUDA graphs, temporary computation buffers, runtime.

The KV cache is the critical shared resource. It holds 894,064 tokens worth of
state across all active requests combined.


### What is KV cache?

When the model reads a token, each of its 94 layers computes two vectors — a Key (K)
and a Value (V). These capture that token's meaning in context. The model stores them
so it doesn't have to recompute them every time it generates a new output word.

Think of it as a notepad the model keeps for each conversation. Longer conversations
need bigger notepads. More concurrent conversations need more notepads. The GPU memory
is the shelf where they all sit.


### Why 894,064 tokens?

Each token's KV costs 47 KB per GPU in your setup:

```
94 layers × 1 KV head per GPU × 128 dimensions × 2 (K + V) × 2 bytes (BF16)
= 48,128 bytes ≈ 47 KB
```

40.07 GB ÷ 47 KB = ~894,064 tokens

**Why BF16 (2 bytes) if the model is FP8?** The FP8 applies only to model weights.
KV cache is always stored in BF16 (higher precision) because KV errors accumulate
across 94 layers and would degrade output quality if stored in FP8. The benefit of
FP8 weights is that they use half the space (~28 GB per GPU instead of ~56 GB),
which frees up more room for KV cache.


### Why does the model have only 4 KV heads instead of 64?

The model uses 64 query heads (Q) but only 4 key-value heads (K,V) — a technique
called Grouped Query Attention (GQA). Every 16 query heads share one set of K,V.

The query heads ask different questions ("What's the subject?", "What's the emotion?",
"What action is happening?"), but they all consult the same shared reference material
(K and V). This cuts KV memory by 16x compared to having 64 KV heads, which is what
makes 128K context windows practical on current hardware.

Quality barely suffers because most of the "question asking" intelligence lives in
the query heads, not in the keys and values.


---


## Part 3: The Three Capacity Ceilings

Every request has to pass three gates. The real capacity is always the
**lowest** of these three.

### Ceiling 1: max_model_len (per-request size limit)

No single request can exceed 131,072 tokens (input + output combined).
This is a hard wall set in your config. The model natively supports up to
262,144 (256K), so you can increase this if needed.

### Ceiling 2: KV cache (total memory across all requests)

The combined tokens of all active requests cannot exceed 894,064.

### Ceiling 3: max_num_seqs (concurrency config limit)

You set this to 2. Even if memory and compute allow more, vLLM will not
start a third request. Extra requests go into a "Waiting" queue.

**This is currently your tightest bottleneck for transcript-only work.**
Your logs confirm this: requests enter "Waiting" state while KV cache is only
2-5% utilized. You have massive unused capacity.


### What the three ceilings look like for each use case

**Use Case 1 — 1-hour transcript (~20K tokens per request):**

| Ceiling | Limit | How many fit |
|---------|-------|-------------|
| max_model_len | 131,072 per request | Each request is 20K — easily fits |
| KV cache | 894,064 total | 894,064 ÷ 20,000 ≈ **44 requests** |
| max_num_seqs | 2 | **2 requests** ← this is the wall |
| Compute saturation | ~6-8 before noticeable slowdown | 6-8 requests |

**Use Case 2 — 5-min video + transcript (~35K tokens per request):**

| Ceiling | Limit | How many fit |
|---------|-------|-------------|
| max_model_len | 131,072 per request | Each request is 35K — easily fits |
| KV cache | 894,064 total | 894,064 ÷ 35,000 ≈ **25 requests** |
| max_num_seqs | 2 | **2 requests** ← this is the wall |
| Compute saturation | ~4-6 before noticeable slowdown | 4-6 requests |


---


## Part 4: How the Model Processes Video

This is important: **Qwen3-VL is a vision-language model, not an audio model.
It cannot process audio. The audio track in your video files is completely
ignored and discarded.**

When you send a video, here is exactly what happens:

```
Your .mp4 file (30fps, with audio)
     ↓
Audio track → DISCARDED (model has no audio encoder, no audio token type)
     ↓
Video frames → sampled at 2 FPS (model's default, ignores the other 28 frames per second)
     ↓
600 frames (for a 5-minute video)
     ↓
Paired temporally (temporal_patch_size = 2) → 300 frame-pairs
     ↓
Each frame split into 16×16 pixel patches by the ViT (27-layer vision encoder)
     ↓
Patches spatially merged 2×2 (spatial_merge_size = 2)
     ↓
~100-120 tokens per frame-pair (depends on resolution)
     ↓
Total: ~30,000-36,000 video tokens fed into the language model alongside your text
```

The transcript you provide is **essential** — it is the only way the model
knows what was said in the video. Without it, the model only sees facial
expressions, gestures, scenes, and on-screen text.


### What else is in the video token count?

Just visual frames. Nothing else. No audio waveform, no audio features,
no subtitles extracted from the video container. Only the pixel data from
sampled frames, processed through the vision encoder.


### Encoder cache (16,384 tokens) — what it really means

The encoder cache is **not a processing limit**. Your video can produce
30,000+ tokens and that's fine — all of them get processed and fed into
the language model.

The 16,384 number is a **reuse cache**. If you send the exact same video
in a later request, up to 16,384 of its visual tokens can be served from
cache instead of being re-encoded. This saves time on repeated requests
with the same video. It does not limit what you can send.


---


## Part 5: Use Case 1 — Transcript Analysis (Finding Moments)


### How big is a transcript in tokens?

English speech averages ~150 words per minute. The tokenizer produces ~1.3
tokens per word. Segment-level timestamps add ~50-70% overhead.

| Video Length | Raw Words | Segment-Level Transcript (with timestamps) | Total with prompt + output |
|-------------|-----------|---------------------------------------------|----------------------------|
| 1 hour | ~9,000 | ~15,000-20,000 tokens | ~20,000-28,000 |
| 2 hours | ~18,000 | ~30,000-40,000 tokens | ~35,000-48,000 |
| 3 hours | ~27,000 | ~45,000-55,000 tokens | ~50,000-63,000 |
| 5 hours | ~45,000 | ~70,000-85,000 tokens | ~78,000-93,000 |
| 6 hours | ~54,000 | ~85,000-100,000 tokens | ~93,000-108,000 |

"Total with prompt + output" includes your system prompt (~500-1,000 tokens)
and the model's response (up to 8,192 tokens for moment identification).


### Question: How many 1-hour transcripts in parallel?

With current config (max_num_seqs=2): **2 at a time**, others queue.

If you raise max_num_seqs to 4-6 (which is safe — you have the memory and
compute): **4-6 at a time**.

From your logs, each request in this range uses ~1.5-2.5% of KV cache.
Even 6 requests would use only ~15%. You have massive headroom.

**What happens to speed with more concurrent requests:**

Your logs show real measured data for 1 and 2 concurrent requests.
Beyond 2 is estimated based on MoE compute characteristics:

| Concurrent | Per-Request Speed | Total Speed | Per-Request Slowdown |
|-----------|-------------------|-------------|---------------------|
| 1 | 63 tok/s | 63 tok/s | baseline |
| 2 (measured) | 61 tok/s | 122 tok/s | 3% slower |
| 4 (estimated) | 55 tok/s | 220 tok/s | ~13% slower |
| 6 (estimated) | 45 tok/s | 270 tok/s | ~29% slower |

The per-request speed drops, but total throughput keeps rising.
More concurrent requests = each individual request takes a bit longer,
but you process more requests per hour overall.

**Diminishing returns means speed only. Quality is never affected.**
The model computes exactly the same math regardless of load.


### Question: How long of a single transcript can I send?

Your hard limit is 131,072 tokens (max_model_len). Subtract output room:

```
Max input = 131,072 - 8,192 (output) = 122,880 tokens
```

This translates to:
- **Segment-level transcript**: ~5-6 hours of video
- **Word-level transcript** (more verbose): ~3-4 hours

For longer content, either:
1. Raise MAX_MODEL_LEN to 262,144 (the model's native max) — handles ~12 hours
2. Split into chunks with some overlap and process sequentially

**Important**: a 35-hour transcript (~525K tokens) will NOT fit in a single
request even at the model's native 256K limit. "35 parallel 1-hour requests"
works, but "1 request with 35 hours" does not — these are different constraints.


### Optimization: batch related work into one request

If you need to run 5 different analyses on the same transcript (find sports
moments, emotional moments, funny moments, etc.), sending them as **one
request with multiple questions** is better than 5 separate requests:

- One request: transcript processed once, model answers all 5 questions
- Five requests: transcript processed 5 times (prefix caching helps speed
  but does NOT reduce KV memory — each request still needs its own KV space)


---


## Part 6: Use Case 2 — Video + Transcript Analysis (Moment Deep-Dive)


### Token breakdown per request

| Component | Tokens | Notes |
|-----------|--------|-------|
| Video frames (5 min, 2 FPS) | ~24,000-34,000 | Visual only; audio is discarded |
| Word-level transcript (5 min) | ~2,000-3,000 | This is how the model "hears" what was said |
| System prompt + question | ~500-1,000 | |
| Model output | ~2,000-4,000 | |
| Audio | 0 | Model cannot process audio |
| **Total per request** | **~28,000-42,000** | Working average: ~35,000 |


### How many parallel video requests?

Raising max_num_seqs to 4-6 is safe from a KV cache perspective.
Even 6 requests at 35K tokens = 210K tokens = ~23% of KV budget.

**But there is a serial bottleneck: the vision encoder.**

Every video's frames must pass through a 27-layer vision encoder (ViT)
before they become tokens. This encoding step happens serially — one video
at a time. It takes roughly 15-30 seconds per video.

However, **once encoding finishes, generation runs in parallel.** So the
pipeline overlaps:

```
           Encoding    Generation
Request 1: [====]      [===========]
Request 2:      [====] [===========]
Request 3:           [====] [========]
Request 4:                [====] [===]
```

With 4 requests, you always have 2-3 in the generation phase running in
parallel, while 1 is being encoded. The encoder stays busy but never blocks
generation.

**Practical recommendation**: raise max_num_seqs to 4. This gives you a
smooth pipeline — the encoder is always working on the next video while
2-3 previous requests generate output in parallel.


### Time estimate per video request

| Phase | Duration | Notes |
|-------|----------|-------|
| Vision encoding (ViT) | ~15-30 seconds | Serial per video |
| Prefill (35K tokens through 94 layers) | ~25-35 seconds | Can overlap with other requests' generation |
| Generation (~3K output tokens) | ~50-75 seconds | Fully parallel with other requests |
| **Total** | **~90-120 seconds** | |

With 4 concurrent and pipeline overlap, throughput is roughly
**2-3 completed requests per minute** or **120-180 per hour**.


---


## Part 7: How Mixture of Experts (MoE) Works — and Why Batching Helps

Your model has 235 billion parameters total, but only 22 billion are active
for any single token. The other 213 billion sit idle.

Each of the 94 layers has 128 "expert" sub-networks. For every token, a
router picks the best 8 experts out of 128. Different tokens get routed
to different experts.

**Experts are NOT like human specialists who get "busy."** They are just
matrix operations. The GPU can run the same expert on tokens from 5
different requests simultaneously — it's just a bigger matrix multiply,
which the GPU handles more efficiently than 5 small ones.

More parallel requests = bigger batch per expert = **better GPU utilization**.

This is why going from 1 to 2 concurrent requests barely slowed down
per-request speed in your logs (63 → 61 tok/s). The MoE architecture
naturally has spare compute capacity that batching fills.

The 128 experts are physically distributed 16-per-GPU. When a token on GPU 2
needs an expert on GPU 5, data travels via NVLink (900 GB/s on H100).
At 2-6 concurrent requests, this communication overhead is negligible.


---


## Part 8: Reading Your Logs — What Every Number Means

Your logs emit a status line every 10 seconds. Here is how to read it:

```
Engine 000: Avg prompt throughput: 1412.1 tokens/s,
            Avg generation throughput: 63.5 tokens/s,
            Running: 1 reqs, Waiting: 0 reqs,
            GPU KV cache usage: 1.6%,
            Prefix cache hit rate: 24.6%,
            MM cache hit rate: 12.2%
```


### Avg prompt throughput (tokens/s)

How fast the model processed input tokens in the last 10 seconds.

- **Non-zero** means a new request arrived and its input is being processed (prefill phase)
- **0.0** means no new requests arrived — the model is only generating output
- Typical values on your system: 1,200-1,600 tok/s for text, up to 3,400 tok/s for batched inputs

This tells you the system is ingesting a new request's input.


### Avg generation throughput (tokens/s)

How many output tokens were produced across ALL active requests combined.

- **~63 tok/s** with 1 request = your baseline single-request speed
- **~120-122 tok/s** with 2 requests = near-linear scaling (MoE benefit)
- **0.0** means no requests are active (idle)
- A small number like **2.4 tok/s** means a request just finished and there was
  partial generation in the last 10-second window

This is the most important performance metric.


### Running / Waiting

- **Running**: requests actively being processed (limited by max_num_seqs=2)
- **Waiting**: requests queued because max_num_seqs is full

Your logs show "Waiting: 1 reqs" at several points — this proves your
max_num_seqs=2 is the active bottleneck, not memory or compute.


### GPU KV cache usage (%)

What fraction of the 894,064-token KV budget is currently occupied.

- **Grows over time** as the model generates more output tokens (each new
  output token adds to the KV cache)
- **Drops to 0.0%** when a request completes (KV is freed)
- Typical for your text requests: 1.5% → 2.5% (≈ 13,400-22,350 tokens)
- Typical for your video requests: 3.9% → 4.5% (≈ 34,868-40,233 tokens)

This tells you the actual size of active requests in memory.


### Prefix cache hit rate (%)

What fraction of incoming prompt tokens were found in cache and
didn't need to be recomputed.

- **Climbs over time**: 0% → 14.8% → 27% → 33% → 37.4%
- This means your requests share similar prefixes (same system prompt,
  same instructions). The first request caches its prefix; subsequent
  requests reuse it.
- Higher = faster prefill for repeated prompts
- Does NOT reduce KV memory usage — cached tokens still occupy KV space
  per request. This only saves computation time.


### MM cache hit rate (%)

"MM" = multimodal. How often visual features (from images/videos) were
found in the encoder cache and reused.

- Climbs from 0% → 30.9% over your session
- Means you're re-analyzing the same videos across multiple requests
- When this is > 0%, the vision encoder (ViT) can skip re-encoding
  previously seen visual content


### HTTP response lines

```
"POST /v1/chat/completions HTTP/1.1" 200 OK
```

Each "200 OK" marks a completed request. You can count these to know how
many requests were served. In your full log, there are approximately
75+ completed requests across the session.


### Putting it together: anatomy of one request

From your logs, here is the lifecycle of a typical text-only request:

```
13:37:35  prompt throughput: 646.1 tok/s, KV: 0.7%    ← Prefill phase begins
13:37:45  generation: 64.2 tok/s, KV: 0.8%            ← Generating output
13:37:55  generation: 64.2 tok/s, KV: 0.9%            ← KV slowly grows
13:38:05  generation: 64.1 tok/s, KV: 0.9%
13:38:15  generation: 64.1 tok/s, KV: 1.0%
13:38:25  generation: 64.0 tok/s, KV: 1.1%
13:38:35  generation: 63.9 tok/s, KV: 1.2%
13:38:45  generation: 63.9 tok/s, KV: 1.2%
13:38:55  generation: 63.8 tok/s, KV: 1.3%
13:39:05  generation: 63.7 tok/s, KV: 1.4%
13:39:15  generation: 63.6 tok/s, KV: 1.4%
13:39:25  "POST /v1/chat/completions" 200 OK           ← Request complete
13:39:35  generation: 2.9 tok/s, KV: 0.0%             ← Draining, KV freed
```

Observations:
- Prefill took one 10-second window at 646 tok/s ≈ 6,460 input tokens
- Generation ran for ~100 seconds at ~64 tok/s ≈ 6,400 output tokens
- KV grew from 0.7% to 1.5% → about 6,260-13,410 tokens total
- Total request: ~12,700 tokens, completed in ~2 minutes

Here is a larger request (likely with video):

```
16:01:56  prompt throughput: 3431.5 tok/s, KV: 3.9%   ← Big prefill (video tokens!)
16:02:06  generation: 61.0 tok/s, KV: 4.0%            ← Slightly slower (larger KV)
...
16:02:56  generation: 60.5 tok/s, KV: 4.3%
16:03:06  "POST /v1/chat/completions" 200 OK           ← Done
16:03:06  KV: 0.0%                                     ← Freed
```

Observations:
- Prefill burst: 3,431 tok/s — much faster because this includes a batch of
  visual tokens being flushed into the KV cache
- KV started at 3.9% = ~34,868 tokens — this is the video+transcript size
- Generation slightly slower at ~61 tok/s (vs 64 for text-only) — the
  larger KV cache means more memory bandwidth per generation step
- Total: ~38,500 tokens, completed in ~70 seconds


### Your log tells a story: Two distinct workload phases

Looking at the full log timeline:

| Time Period | What Happened |
|-------------|--------------|
| 06:48-06:51 | Service starts, model loads (2 min), KV cache allocated |
| 07:04-07:07 | First 2 requests (text-only, one at a time) |
| 07:07-13:37 | **6.5 hours idle** |
| 13:37-15:12 | Burst of many requests (mix of 1 and 2 concurrent, some with video) |
| 15:12-16:01 | 50 minutes idle |
| 16:01-16:07 | A few requests including video |
| 16:07-17:41 | 1.5 hours idle |
| 17:41-19:03 | Another burst of requests (steady stream, mostly 1 at a time, some 2 concurrent) |

Key finding: **most of the time, only 1 request runs at a time**. The
max_num_seqs=2 limit was hit only occasionally, and even then the system
was comfortable (2-5% KV usage). This system is heavily underutilized.


---


## Part 9: How to Think About Capacity (The Mental Model)

When planning for a workload, ask three questions:

### Question 1: Does each request fit?

```
input_tokens + expected_output_tokens ≤ max_model_len (131,072)
```

If not, either shorten the input, increase max_model_len, or split the work.


### Question 2: Does the total fit in memory?

```
Sum of all concurrent requests' tokens ≤ 894,064 (KV budget)
```

Each request's tokens = input + all output generated so far.


### Question 3: Is max_num_seqs high enough?

```
Desired concurrent requests ≤ max_num_seqs
```

If not, change the config.


### Estimating tokens for any input

- **Text**: word_count × 1.3
- **Segment transcript**: word_count × 1.3 × 1.6 (timestamps add ~60%)
- **Word-level transcript**: word_count × 1.3 × 2.2 (timestamps per word)
- **Video**: (duration_seconds × 2 FPS ÷ 2) × 100-120
- **Images**: (height ÷ 32) × (width ÷ 32) per image
- **Audio**: 0 (not processed)


### Estimating time for any request

```
Prefill time ≈ input_tokens ÷ 1,300 seconds
Generation time ≈ output_tokens ÷ 60 seconds (per request)
Vision encoding ≈ 15-30 seconds (if video is included)
Total ≈ encoding + prefill + generation
```


---


## Part 10: Scaling Options

### Immediate (free, config changes only)

| Change | What you get | Risk |
|--------|-------------|------|
| max_num_seqs: 2 → 4 | Double the concurrent text requests | Slight per-request slowdown |
| max_num_seqs: 2 → 6 | 3x concurrent text requests | More noticeable per-request slowdown |
| gpu_memory_utilization: 0.90 → 0.95 | ~10% more KV budget (~980K tokens) | Slightly higher OOM risk |
| MAX_MODEL_LEN: 131K → 256K | Handle transcripts up to ~12 hours | Fewer max concurrent at full context |
| --kv-cache-dtype fp8 | Double KV budget (~1.8M tokens) | Slight quality risk for precision-sensitive tasks |


### Vertical scaling (bigger hardware)

| Upgrade | Effect |
|---------|--------|
| 8x H100 → 8x H200 (141 GB each) | ~75% more VRAM → ~1.56M token KV budget |

Only needed if you regularly hit KV cache limits, which is unlikely
with your current workloads (2-5% usage).


### Horizontal scaling (more machines)

Add a second 8x H100 machine with a load balancer in front.
Each machine runs its own vLLM instance independently.

| Machines | Total Concurrent | When to do this |
|----------|-----------------|-----------------|
| 1 (current) | 2-6 | Current |
| 2 | 4-12 | When you need 100+ requests/hour |
| 4 | 8-24 | When you need 300+ requests/hour |

This is the right answer when you need **more total throughput** rather
than longer context or faster individual requests.


---


## Part 11: Quick Reference Card

| What you want to know | Answer |
|-----------------------|--------|
| Max tokens per request | 131,072 (configurable up to 262,144) |
| Total KV cache budget | 894,064 tokens |
| KV cost per token | 47 KB per GPU |
| Single request generation speed | 63 tok/s |
| Two concurrent generation speed | 61 tok/s each (122 total) |
| 1-hour transcript size | ~15,000-20,000 tokens |
| 5-minute video size | ~24,000-34,000 tokens (visual only, no audio) |
| Max single transcript length | ~5-6 hours (segment-level, at 128K context) |
| Safe concurrent text requests | 4-6 (raise max_num_seqs) |
| Safe concurrent video requests | 3-4 (raise max_num_seqs) |
| Does the model process audio? | No — audio is completely discarded |
| Does more concurrency hurt quality? | No — only speed is affected |
| Current biggest bottleneck | max_num_seqs=2 (artificial config limit) |
| KV cache utilization in your logs | 2-5% (massively underutilized) |
| Prefix cache hit rate trend | Climbed from 0% to 37% over the session |
