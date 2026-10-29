# Experimental Resumable Prefill

This remains an unqualified experiment with passing CPU scheduler tests and limited all-rank GPU smoke coverage. Short output/cache checks and a two-stream overlap check passed with the scheduler enabled. Full numerical equivalence, mixed-load latency, and long-context behavior remain unqualified. The port lets incumbent decoders run between a newcomer's prefill pieces without copying the key/value (KV) pool.

## Purpose and Mechanics

When a new prefill request arrives, it is processed in chunks (default 2048 rows) rather than as a single blocking operation. To prevent starvation of existing decoders, the scheduler manages a "time debt":

- **Scheduling:** The newcomer receives at most one prefill piece per decode round.
- **Time Debt:** A piece taking time `t` owes `t` of completed decode time before the next piece. Debt survives request completion and gates the next admission, targeting a 50% prefill share with at least one decode round between pieces.
- **Solo Mode:** In solo execution (no incumbent decoders), the newcomer drains pieces without a decode rate cap.

The stepped path disables cross-chunk read-ahead of Engram's host-resident lookup tables on every rank. Overlap within a chunk remains enabled. The head fences each piece before returning. Scheduling uses a separate monotonic clock, leaving the adaptive learner's clock alone.

One pending prefill reserves admission capacity but stays outside the decode core. Its restored cache entry is reported before the first yield, so incumbent completion can evict that entry safely.

## Configuration and Usage

This feature is enabled via the environment variable `TF_DS_RESUMABLE_PREFILL=1`. If this flag is absent, the system defaults to the legacy blocking prefill behavior.

Recipe `v126-resumable` pins the experimental source independently of stock `v126`. Profile `1.2.6-resumable` selects that recipe and the separate `bin-1.2.6-resumable` directory. All ranks require matching source/build receipts and the same runtime flag.

**Note:** This experimental profile is intentionally marked as `qualified_pool=false` and `serving_smoke_qualified=false` to maintain existing qualification guards.

### Launcher Invocation (Example Only)

The following command is for illustrative purposes; running it does not validate or start the service:

```sh
python3 -m tensorfold.launch --config tensorfold/cluster.local.json --variant 1.2.6-resumable --rank 0
python3 -m tensorfold.launch --config tensorfold/cluster.local.json --variant 1.2.6-resumable --rank 1
python3 -m tensorfold.launch --config tensorfold/cluster.local.json --variant 1.2.6-resumable --rank 2
```

## Semantics and Tradeoffs

- **Caller Visibility:** The intended response format is unchanged. The newcomer emits no token until its prefill completes; incumbents can continue meanwhile. Numerical equivalence still needs GPU qualification.
- **Bounding:** Progress is bounded in units of pieces, not by a hard latency guarantee (e.g., encoding, restore IO, Engram, or snapshots).
- **Tradeoffs:** Potential for cold prefill or C1 throughput regression due to synchronization and prefetch overhead.
- **Timing:** API `prefill_seconds` includes intervening decode rounds; it is not GPU-only prefill compute time.

## Limitations

This implementation does not include:
- Short request bypass
- Background preemption
- New KV management or kernels
- New draft policies

## CPU verification

The main Zig CPU suite passed all 23 build steps. Six additional CPU module checks also passed. Behavioral tests drive the real host and serving wrapper over incremental fake-model prompt work. They cover incumbent interleaving, first-token timing, cancellation, stop, queue order, ContextFull retry, prefix eviction and eight mixed sampled/greedy streams matching blocking outputs.

Source reconstruction and drift rejection passed for every recipe, including this experiment. All 49 Python tests passed, including print-only launch, build-receipt isolation, qualification-only command rendering, and rejection of unqualified normal launches.

These checks use synthetic state, not model weights or GPUs.

The full native recipe built both ARM64 serving binaries and a matching build receipt on a separate x86-64 Fedora host. No Docker, model weights or GPU execution were needed. Copied binaries, build logs and receipt hashes were verified.

A source-identical stock `v126` reference was built separately with the same toolchain and kernel blobs. Both ARM64 executable sets have since booted on all three ranks; the experimental set passed the limited scheduler-enabled GPU smoke checks below.

The head needs glibc 2.29 or later; the follower needs glibc 2.28 or later. Both use `/lib/ld-linux-aarch64.so.1`, compatible with the inspected target's glibc 2.39. Recheck follower libraries before launch. This verifies basic binary ABI requirements, not GPU libraries or runtime correctness.

Local head builds exceeded the isolated 2 GiB CPU-memory limit. Keep that limit while GLM-5.3-Flash runs; the successful off-host build used a separate 6 GiB cap.

## Qualification still required

The profile has no inherited GPU results. The launcher guard has not been bypassed; `--run` remains blocked for this unqualified profile.

The native binaries are installed separately from mutable source checkouts. The scheduler-enabled GPU smoke exercised a 9,031-token newcomer while an incumbent produced a coherent 1,024-token counting response. The incumbent continued emitting during the newcomer request; the newcomer and its fully cached repeat both answered the arithmetic question correctly. These synthetic checks do not establish general numerical parity or representative workload performance.

In an approved GPU window, compare the experimental binary with the flag absent against retained stock `v126`. This checks the blocking-path refactor; that comparison remains outstanding. Beyond the scheduler-enabled smoke checks, validate all-rank frame ordering, cancellation/cleanup, cold and cached equivalence, tools, vision and memory.

Then compare C8 mixed workloads at 131k and 262k, including cold newcomers and tool-result appends. Record per-stream delivery gaps and time to first token (TTFT), not aggregate throughput alone.

The stock context limit remains 401,408 tokens. Two approximately 500k-token streams require a separate profile and qualification.

GLM's resume guarantees do not automatically transfer to DeepSeek. Upstream issue #1395 reports inconsistent replies on repeated DeepSeek prompts; compare the same cache state against the retained v126 baseline.

## References

- [GLM TensorFold piece-scheduler design](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docs/BATCHING-DESIGN.md): source of the fairness policy, not a port of its model kernels.
- [DS41F 1.2.6 production guide](DS41F-1.2.6.md)
- [Recipes and build instructions](RECIPES.md)
- [DeepSeek repeated-prompt issue #1395](https://github.com/ashhart/TensorFold/issues/1395)
