# Load-Aware Routing for LLM Inference

How do we decide whether to send the next request to a given replica of
Qwen3-VL when "number of in-flight requests" is **not** a good load signal?

Companion to `MND.md` (which covers prefix/KV-cache affinity). This doc is
only about **load awareness** — choosing *how much more work a replica can
safely take right now*.

---

## 1. Why normal load balancers don't work here

A plain REST service is roughly *O(1)* per request: CPU finishes the call,
releases the thread, next request comes in. "Number of open connections" or
"requests per second" correlates tightly with load.

LLM serving is nothing like that:

- **Cost per request varies 10–100×.** A 30 s video + long thinking answer
  can take 40 s; a short text query can take 2 s. Both look identical to an
  L4/L7 LB.
- **Requests overlap on the GPU.** vLLM continuously batches; the 2nd, 3rd,
  4th request get stitched into the same forward pass as the 1st. Cost is
  non-additive.
- **State lives on the GPU for the whole request.** KV cache is held from
  prefill until the last decode token. A "running request" that is generating
  5,000 thinking tokens holds memory and bandwidth for tens of seconds.
- **Two very different phases.** Prefill is compute-bound and huge for video
  prompts; decode is memory-bandwidth bound and long for thinking models.
  A replica can be busy prefilling while decode slots are free, and vice
  versa.

Because of this, "send to the replica with the fewest connections" routinely
sends work to a node that is about to stall.

---

## 2. What actually predicts load on a vLLM replica

vLLM exposes these on `/metrics` (Prometheus) and in its heartbeat log line.
They are the signals a real load-aware router needs.

| Signal | What it tells you | Why it matters |
|---|---|---|
| `vllm:num_requests_running` | Requests currently on the GPU | Capped by `MAX_NUM_SEQS` |
| `vllm:num_requests_waiting` | Requests queued, not yet scheduled | **Head-of-line delay** — the killer signal |
| `vllm:gpu_cache_usage_perc` | % of KV cache blocks in use | Near 100% → preemptions / eviction |
| `vllm:kv_cache_usage_perc` (alias) | Same as above | |
| `vllm:time_to_first_token_seconds` | Prefill latency (histogram) | Spikes when prefill is saturated |
| `vllm:time_per_output_token_seconds` | Decode latency | Rises as batch + KV grow |
| `vllm:prompt_tokens_total` | Counter of prompt tokens served | Prefill pressure trend |
| `vllm:generation_tokens_total` | Counter of generated tokens | Decode pressure trend |
| `vllm:request_queue_time_seconds` | Time requests sit in queue | Direct SLO signal |
| `vllm:num_preemptions_total` | Requests kicked out due to KV pressure | Warning flag |

Plus node-level signals:

- `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_MEM_COPY_UTIL`, `DCGM_FI_DEV_FB_USED`
  from `dcgm-exporter`.

### What the client/router can estimate per request

The router doesn't know exactly how long a request will take, but it can
**cheaply estimate cost** before dispatching:

- **Prefill cost** ≈ `prompt_tokens` (+ a big multiplier for video frames).
  Count vision tokens from fps × duration × tokens-per-frame.
- **Decode cost** ≈ `max_tokens` requested (upper bound). For thinking models,
  assume it will actually use most of it.
- **KV cost** ≈ `(prompt_tokens + expected_output_tokens) × bytes_per_token`.

That lets the router do **admission control** (not just routing):
"can this replica fit another ~40 GB of KV for this request without evicting
someone already running?"

---

## 3. Policies people actually use in production

Ordered from dumb → smart. In practice teams pick one and tune.

### P0 — Round-robin / least-connections

Baseline. Works for warm-up, fails under real load. Keep it only as a
fallback when metrics scrape fails.

### P1 — Least-requests (by `num_running + num_waiting`)

Way better than RR for LLMs because it at least tracks what's on the GPU.
Still bad when request sizes differ a lot: a replica with 1 huge request can
look "idle" compared to one with 3 small ones.

### P2 — Least-loaded by weighted score

Compose a single scalar per replica:

```
score = a * num_running
      + b * num_waiting
      + c * kv_cache_usage_perc
      + d * ewma(ttft_seconds)
```

Pick the minimum. Typical weights put queue depth and KV usage highest.
This is what most OSS LLM routers actually do under the hood.

### P3 — Score + cost-aware admission

Before picking, compute the request's estimated KV footprint and reject (or
defer) any replica where `kv_cache_usage_perc + est_kv_for_request > 90%`.
Avoids the "looks fine, then preempts everyone" failure mode.

### P4 — Prefix-aware + load-aware (hybrid)

*This is the production-grade target.*

```
1. Compute prefix hash of the request.
2. Find replicas that have this prefix cached (gossip / cache table).
3. Among them, if any has headroom (P3 check) → send there.
4. Else fall back to P2 across all replicas.
```

Gains KV reuse without overloading any single hot replica.

### P5 — Disaggregated prefill/decode routing

Split replicas into a **prefill pool** and a **decode pool** (vLLM / llm-d
support this). The router sends the prefill half of a request to a
prefill-heavy node, streams the KV cache to a decode node, and completes
there. Huge for video-heavy prefill + long thinking decode (exactly our
workload), but operationally the most complex.

---

## 4. Production options — pros and cons

### A. vLLM Production Stack (`vllm-router`)

- **What it is:** Official vLLM router. Does prefix-aware routing,
  least-loaded routing, scrapes `/metrics` from each replica.
- **Pros:** Built for vLLM, knows its metrics natively, easy to run, same
  project as the engine.
- **Cons:** Younger than alternatives; fewer enterprise-grade features (no
  real policy engine, limited observability). Admission control is basic.
- **When to pick:** Default choice if you're on vLLM and want something
  working in a day.

### B. AIBrix (ByteDance, open source)

- **What it is:** K8s-native LLM gateway + control plane. KV-aware routing,
  autoscaling, optional disaggregated prefill/decode, distributed KV cache.
- **Pros:** Most complete OSS offering in this space. Prefix routing, load
  routing, and horizontal autoscaling are integrated. Designed for the
  problem you're describing.
- **Cons:** Requires Kubernetes. More moving parts; steeper operational
  curve. Tied to their CRDs.
- **When to pick:** On K8s, planning to run 3+ nodes, want autoscaling
  included.

### C. llm-d (Red Hat / IBM / Google / others)

- **What it is:** K8s-native distributed inference framework built around
  vLLM. First-class support for **disaggregated prefill/decode** and
  prefix-cache-aware scheduling via the Envoy Gateway "Inference Gateway"
  (GIE) API.
- **Pros:** Strong architecture for mixed workloads (our prefill-heavy +
  decode-heavy video case). Backed by multiple vendors, standardizing
  around Kubernetes Gateway API Inference Extension.
- **Cons:** Heavy. Requires Envoy Gateway, K8s, a fair amount of YAML.
  Early days — moving fast, some rough edges.
- **When to pick:** You want the disaggregated prefill/decode split and
  you're committed to K8s + Envoy.

### D. Envoy AI Gateway / Gateway API Inference Extension (GIE)

- **What it is:** An emerging standard CRD + Envoy extension for "smart"
  LLM load balancing. Not an engine; it's the *routing layer*.
- **Pros:** Vendor-neutral standard; integrates with llm-d, AIBrix, and
  direct vLLM pods. Model-aware metrics (queue depth, KV usage) baked into
  the spec.
- **Cons:** Still maturing. Requires Envoy Gateway infrastructure.
- **When to pick:** You already run Envoy Gateway and want the routing
  decision to be a first-class K8s resource.

### E. NVIDIA Dynamo / Triton Inference Server

- **What it is:** NVIDIA's serving stack, now with disaggregated
  prefill/decode and smart routing (Dynamo). Supports vLLM and TensorRT-LLM
  as backends.
- **Pros:** Enterprise support path. Strong if you also plan to run
  TensorRT-LLM. Good observability.
- **Cons:** Heavier, NVIDIA-flavored, more to learn. Overkill for 2–3 nodes.
- **When to pick:** You're standardizing on NVIDIA's stack across inference
  services, or you want a vendor-supported path.

### F. Custom router (FastAPI / Go, ~300 lines)

- **What it is:** A tiny service that scrapes `/metrics` from each replica
  every 1 s, keeps a small in-memory table, and implements P3/P4.
- **Pros:** Exactly the behavior you want, no K8s required, easy to debug,
  you own every decision. Fine for 2–5 nodes.
- **Cons:** You own it forever. No autoscaling, no multi-tenant policy,
  no fancy retries/failover unless you build them.
- **When to pick:** Small fleet (≤5 nodes), no K8s, want to understand
  the routing behavior deeply before committing to a framework.

---

## 5. Comparison table

| Option | Prefix-aware | Load-aware | Cost-aware admission | Disagg P/D | K8s required | Ops cost |
|---|---|---|---|---|---|---|
| Nginx / HAProxy RR | no | no | no | no | no | trivial |
| `vllm-router` | yes | yes (basic) | partial | no | no | low |
| AIBrix | yes | yes | yes | optional | **yes** | medium |
| llm-d + Envoy GIE | yes | yes | yes | **yes** | **yes** | high |
| Envoy AI Gateway / GIE | yes | yes | yes | depends on backend | **yes** | high |
| Triton / Dynamo | yes | yes | yes | yes | optional | high |
| Custom router | yes (if you build it) | yes | yes | no | no | medium (ongoing) |

---

## 6. A minimal policy that covers 90% of the problem

If you only implement one thing, this is the one. It's what most of the
options above do internally.

**Per replica, keep a live `score`:**

```
score = num_waiting * W1                 # queue pain (biggest weight)
      + num_running * W2                 # active load
      + kv_cache_usage_perc * W3         # memory pressure
      + ewma(ttft_seconds) * W4          # observed slowness
```

**For each incoming request:**

```
1. Estimate this request's KV cost from (prompt_tokens + max_tokens).
2. Filter out any replica where
      kv_cache_usage_perc + est_kv_pct_for_request > 90%.
3. (If prefix-aware) Prefer replicas that have its prefix cached,
      unless their score is >2× the global minimum.
4. Among survivors, pick the lowest score.
5. If no replica survives step 2, queue the request at the router
      and retry in 200 ms.
```

Typical weights to start with: `W1=3, W2=1, W3=2, W4=1`. Tune by watching
p95 latency.

This gives you load-aware + cost-aware routing in a few hundred lines,
regardless of which deployment stack you end up on.

---

## 7. Recommendation for us

Given today we run **one node, no LB, no TLS** (per `MND.md`), the
pragmatic ladder is:

1. **Turn on vLLM's Prometheus endpoint** on this node. Expose
   `/metrics`. This alone unlocks every option below.
2. **Start with `vllm-router`** in front of the current single node.
   Validates the contract, gives us prefix + load-aware routing out of the
   box, costs a day of work.
3. **Add node #2.** Only then do we learn what real multi-node behavior
   looks like. Watch preemption counts and queue times — they'll tell us
   whether admission control (P3) is needed.
4. **Revisit when nodes ≥ 3 or latency SLO gets tighter.** At that point
   evaluate AIBrix (if on K8s) or llm-d (if we also want disaggregated
   prefill/decode — likely relevant because our prefill is video-heavy).

**Do not** build a custom router yet. It's tempting and it's only 300 lines,
but `vllm-router` already does the P2+P4 policy and we'd be reinventing it.
Build custom only if `vllm-router` can't express a policy we actually need.

---

## 8. Open questions to answer before picking a framework

- Are we going to run this on **Kubernetes** or on bare-metal VMs?
  (Answer changes the shortlist from `vllm-router` / custom → AIBrix / llm-d.)
- Do we want **autoscaling** (bring nodes up/down with traffic) or
  a fixed fleet? Only AIBrix / llm-d do this well.
- What's the **latency SLO** — p50, p95, p99? P3 admission only pays off
  when SLOs are tight enough that preemptions matter.
- Will we expose the model to **multiple tenants / priorities**? If yes,
  the router also needs fairness / priority, which pushes toward AIBrix or
  llm-d.
