# Qwen3-VL-235B Service — Scaling & Latency Notes

A focused "where we are / what to do next" write-up for two questions:

1. How do we make **each request** (video-moment refinement) faster?
2. How do we **scale horizontally** across more GPU nodes with a smart,
   KV-cache / context aware load balancer?

This is a decision doc, not a design doc. Just enough numbers to choose a direction.

---

## 1. What the service is actually doing today

- **Model:** `Qwen3-VL-235B-A22B-Thinking-FP8` (MoE, 236B total / ~22B active, FP8).
- **Runtime:** `vllm 0.15.1`, V1 engine, `enable-expert-parallel`, chunked prefill on.
- **Hardware:** 1 node, 8× H100 80GB, tensor-parallel = 8, `gpu-memory-utilization = 0.90`.
- **Limits in `config.env`:**
  - `MAX_MODEL_LEN = 131072` (128K context)
  - `MAX_NUM_SEQS = 4` (max 4 in-flight requests)
  - `LIMIT_MM_PER_PROMPT = {"image": 4, "video": 1}`
  - `MAX_TOKENS = 8192` (generation cap)
- **Serving:** plain HTTP on `0.0.0.0:8010`, no reverse proxy, no TLS, no LB.

**Cold-start cost** (from `logs/service.log`, one-time):
- Weight load: **125 s**
- Engine init + KV cache + warmup: **45 s**
- Total startup: **~3 min**

---

## 2. What the logs tell us about the current load

Extracted from `logs/service.log` (the live vLLM engine heartbeat line:
`Avg prompt throughput … Avg generation throughput … Running … Waiting … GPU KV cache usage …`).

### Throughput

| Phase | Peak observed | Typical |
|---|---|---|
| Prefill (prompt) throughput | **~6,800 tok/s** | 1,300–3,400 tok/s |
| Decode (generation) throughput, aggregate | **~246 tok/s** | 200–240 tok/s |

### Per-request decode speed vs concurrency

Computed over all heartbeat samples in the log:

| Running reqs | Aggregate gen tok/s | Per-request tok/s | Samples |
|---|---|---|---|
| 1 | 35 | **35** | 149 |
| 2 | 119 | **59** | 54 |
| 3 | 180 | **60** | 98 |
| 4 | 218 | **55** | 282 |

**Read:** Going from 1 → 4 concurrent requests gives ~6× aggregate throughput but
per-request speed only drops from ~60 to ~55 tok/s. We are nowhere near compute
saturation at `MAX_NUM_SEQS = 4`.

### KV cache & queue

- **Total KV cache:** 894,064 tokens (40 GiB per GPU after weights).
- **Max concurrency the cache can hold** (for 131K-token requests): 6.82x.
- **Observed GPU KV usage:** 4–10% almost always. We never come close to filling it.
- **Waiting queue:** hits 1–3 requests occasionally, so we *do* queue — but because
  `MAX_NUM_SEQS = 4`, not because of KV pressure.

### Cache hit rates

- **Prefix cache hit rate:** mostly 0–15%. Low.
- **MM (vision) cache hit rate:** ~0% almost always. Vision features are never reused.
- **Total requests served in this log window:** **5,870**.

---

## 3. Where time goes on one moment-refinement request

The task: *2–3 min of transcript + 2–3 min of video clip → find exact start/end of a moment.*

Rough per-request cost today (based on the numbers above and Qwen3-VL's video tokenization, ~2 fps × a few hundred tokens per frame):

| Stage | Typical size | Time at current throughput |
|---|---|---|
| Vision encoding + prefill | ~20–60K tokens (video dominates) | **4–12 s** (≈5K tok/s prefill) |
| "Thinking" + answer generation | 500–2,000 tokens | **10–35 s** (≈55 tok/s/req) |
| **End-to-end** | | **~15–45 s per request** |

### The two things eating the clock

1. **Video tokens** dominate prefill. One 3-min clip can be tens of thousands
   of tokens before any text prompt is added.
2. **Thinking tokens** dominate decode. This is the `-Thinking` variant — it
   emits internal reasoning before the final answer, inflating generated tokens
   by 3–10×.

Everything else (system prompt, transcript, answer) is rounding error.

---

## 4. Making each request faster — options ranked by bang/buck

Ordered by expected impact on p50 latency for this specific task.

### Tier 1 — Biggest wins, low risk

1. **Shrink the video input.** Qwen3-VL's video tokens scale with
   `fps × resolution × duration`. Drop fps (e.g. 2 → 1 fps), resize longer-side
   (e.g. 768 → 448), and prefer pre-segmented clips. This is the single largest
   lever — halving video tokens ~halves prefill and also frees KV cache.

2. **Coarse-to-fine (two-stage) localization.** Don't hand the model a full
   3-minute clip for fine-grained timestamping. Run a cheap pass first
   (keyframes, CLIP/embedding scan, or even this model at 0.5 fps) to pick a
   ~15–30 s window, then send only that window at higher fps for precise
   start/end. Cuts per-request tokens by 4–10×.

3. **Cap thinking budget.** The `-Thinking` model emits reasoning tokens that
   rarely add quality on a pure timestamp-extraction task. Options:
   - Set a hard `max_tokens` (e.g. 512) and use a strict output schema.
   - Switch to `Qwen3-VL-235B-A22B-Instruct-FP8` (non-thinking) for this task —
     same vision backbone, no reasoning tax. Usually 3–5× faster generation.

4. **Structured / guided decoding.** Force JSON output of the form
   `{"start": "00:01:32.4", "end": "00:01:47.1"}`. vLLM supports this natively.
   Combined with (3), generation drops to ~50–150 tokens.

### Tier 2 — Config tweaks on the current box

5. **Lower `MAX_MODEL_LEN` to what you actually use.** 131K is huge for this
   task. Dropping to 32K frees KV cache and lets vLLM batch more aggressively.

6. **Raise `MAX_NUM_SEQS`.** We observed KV cache at 4–10% with `MAX_NUM_SEQS=4`
   and no per-request slowdown going 1→4. 8–16 is almost certainly safe and
   improves aggregate throughput, which indirectly cuts queue-wait latency.

7. **Shape prompts for prefix caching.** Current prefix hit rate is ~5–15%.
   Put the system prompt + task instructions + schema *first*, and any
   request-specific data (video, transcript) *after*. Identical prefixes then
   get cached across calls → free prefill for the stable part.

8. **Keep MoE config on.** We saw the warning:
   *"Config file not found for `E=16,N=1536,…fp8_w8a8,block_shape=[128,128]`. Using default MoE config. Performance might be sub-optimal."* Generate or pull the
   tuned MoE config (vLLM ships a tuner); single-digit % MoE gains.

### Tier 3 — Bigger lifts, revisit if still too slow

9. **Speculative decoding / EAGLE** — 1.5–2× decode speedup if a compatible
   draft model is available for Qwen3-VL.
10. **Disaggregated prefill/decode** (vLLM / llm-d) — separates the video-heavy
    prefill onto one pool and decode onto another. Big win when prefill and
    decode fight for resources; our logs show exactly that pattern.

---

## 5. Scaling horizontally — why plain load balancing fails here

Most L4/L7 load balancers (nginx, HAProxy, ELB) distribute on **connections**
or **round-robin**. That breaks badly for LLM inference because:

- **Requests are wildly unequal.** A 30 s video costs 10× a 10 s one. RR sends
  the next huge request to whichever node is "next", not the least-loaded one.
- **KV cache is sticky state.** If the same long system prompt / transcript /
  video is reused (retries, chains, refinement loops), only the node that has
  it cached avoids re-prefill. RR actively destroys that locality.
- **Decode is memory-bandwidth bound.** A node with 3 long decodes in flight is
  slower per-token than a node with 0, but both may look "up" to a TCP LB.
- **Head-of-line blocking** on `Waiting` queues is invisible to the LB.

### What we actually want: a **KV-cache-aware, load-aware router**

Two signals, combined:

1. **Context / prefix signal.** Hash a stable prefix of the request
   (system prompt + transcript id + video id). Route the same prefix to the
   same node when possible. Amplifies prefix cache hit rate from ~10% towards
   50–80% on workloads with reuse.

2. **Live load signal.** Every vLLM node exposes Prometheus metrics at
   `/metrics` — we already have heartbeat equivalents in the log. Use:
   - `vllm:num_requests_running`
   - `vllm:num_requests_waiting`
   - `vllm:gpu_cache_usage_perc`
   - `vllm:time_to_first_token_seconds`
   
   Pick the node with the lowest weighted score.

Routing rule (simple and effective):

- If some node has the request's prefix cached **and** its queue depth is below
  a threshold → send there.
- Else → send to the node with the lowest `(running + waiting) * avg_ttft`
  score.

### Options to implement this

| Option | What it is | Fit |
|---|---|---|
| **vLLM Production Stack / `vllm-router`** | Official vLLM router with prefix-aware + load-aware routing. | Best default — built for exactly this. |
| **AIBrix** (ByteDance, OSS) | K8s-native LLM gateway with KV-aware routing, autoscaling, disagg prefill. | Strong choice if on K8s. |
| **llm-d** (Red Hat / IBM) | K8s-native, supports disaggregated prefill/decode and prefix routing. | Heavier setup, more features. |
| **NVIDIA Dynamo / Triton** | Ensemble + smart scheduling. | Overkill unless already standardized on Triton. |
| **Roll-your-own router** | Small FastAPI/Go service: hash prefix → consistent-hash ring, read `/metrics` every 1 s, pick node. | ~200 lines, fully under our control, good if we only need 2–5 nodes. |

### What **not** to use

- Plain nginx / HAProxy / AWS ALB round-robin or least-connections.
- Sticky sessions by client IP (wrong locality key — we want prefix locality,
  not client locality).

### Gotchas when we add nodes

- Each node is a full 8× H100 replica, ~236 GB weights. Model download and
  warmup is ~3 min per node — plan for rolling restarts.
- Prefix-aware routing relies on **stable, identifiable prefixes**. Client
  should send a `prefix_id` header (or we hash the first N tokens) so the
  router can group related requests.
- If we turn on disaggregated prefill, prefill nodes and decode nodes have
  very different resource profiles — the router must be aware of that.

---

## 6. Recommended next steps (in order)

1. **Measure before changing.** Turn on vLLM's Prometheus endpoint and log
   per-request prompt/generation token counts. Today the log only shows
   engine-wide averages, so we can't see p50/p95 per request.
2. **Cut video tokens** (fps + resolution + two-stage localization). Expect
   2–4× per-request speedup with no infra change.
3. **A/B the `-Instruct` (non-thinking) variant** for this task with structured
   JSON output. If quality holds, this alone likely halves p50.
4. **Raise `MAX_NUM_SEQS` to 8–16 and drop `MAX_MODEL_LEN` to 32K** (assuming
   we don't actually need 128K). Safe given observed KV usage.
5. **Stand up the router before the second node.** Even with one node behind
   it, you validate the routing contract. Start with `vllm-router` (prefix +
   load aware) unless we have a reason to build custom.
6. **Only then add node #2.** Use the router's metrics to confirm we're
   actually getting prefix-cache amplification, not just dumb fan-out.

---

## 7. Open questions for the team

- Do real prompts **reuse** prefixes (same transcript re-sent for multiple
  moment queries)? If yes, prefix-aware routing is a huge win. If every
  request is fully unique, the routing story collapses to "load-aware only".
- Is the **thinking** variant actually helping quality on timestamp tasks, or
  are we paying for reasoning we discard? Needs a small eval.
- How tight is the latency SLA? 15 s p50 is achievable today with config
  changes; <5 s likely needs coarse-to-fine + non-thinking + guided decoding.
