# Capacity scoring v2: the sybil loophole and the fix (analysis, 15 Oct 2026)

Branch: `fusion/sn92-fast-launch-20261015-capv2-own` from `release/1.2-rc` (11d57b9). Not deployed. The Owner decides.

## 1. What we measured live (read-only, Sandy)

Sources: the miner-dashboard collector (`reax-miner-dashboard-20261015/collector/state/miners.sqlite`: round
observations, weight sets and a finney metagraph cache), the lead's STATE.md and the CoS board entry.

**No validator capacity log lines are stored on Sandy.** The collector keeps the round lines only, not
`uid N [pool] capacity probe: ...`. So I could not collect the measured capacity distribution itself. The lead can
get it read-only on the validator host:
`docker logs --since 3h <validator container> 2>&1 | grep 'capacity probe:'`.
What the stored data does show:

| Evidence | Value |
|---|---|
| Validator on 1.2 (capacity probes live) | since 12:10:47Z; first capacity-weighted weight set at block 9,288,996 |
| Shadow run before the switch (STATE.md 11:45Z) | "UID4 40.0/s = target; several external UIDs also at cap" |
| Metagraph at block 9,289,210 (14:06Z): UIDs with incentive > 0 | **96 UIDs from only 22 coldkeys** |
| Incentive per UID | min 0.0069, median 0.01054, max 0.0106; coefficient of variation **4.4 %** |
| Hotkeys per coldkey (top) | 11, 10, 9, 8, 7, 6, 6, 6 ...; each coldkey's incentive is almost exactly proportional to its hotkey count (11 → 0.111, 10 → 0.104, 9 → 0.095, 8 → 0.084) |
| First 1.2 weight set (9,288,996, ~70 min after the switch, capacity still ramping with α = 0.3) | 85 s1-fast UIDs weighted, 13 already at the maximum 0.0286; implied capacity factor deciles 0 / 0 / 0.09 / 0.13 / 0.15 / 0.24 / 0.27 / 0.40 / 0.83 / 1.0 |
| s1-fast round latency (8 items, last 6 h, per-UID mean) | p10 0.32 s, p50 0.62 s, p90 2.3 s |

Reading: about an hour after the switch, almost every UID's capacity average reached the 40/s cap. From then on, pay
inside s1-fast is the same for every UID. Each coldkey earns in proportion to how many hotkeys it registered, and the
hardware behind them does not matter.

Reference throughput (docs/CALIBRATION.md, one decision at a time unless noted): RTX PRO 6000 55.6 (74 through the
runtime's HTTP endpoint), H100 60.5, RTX 4090 30.4 (49 via HTTP), RTX 5090 41.6, L40S 31.0, RTX A6000 19.3, RTX 3090
26.7; vLLM bf16 batched: PRO 6000 135, H100 226, RTX 4090 57 decisions/s. s1-pro (deterministic Rune): 15–17 / 73–77 /
93–108 decisions/s at concurrency 1 / 16 / 64.

## 2. Why the 1.2.0rc2 rule creates the sybil incentive

In-pool weight = `ema × min(1, c / T)` with T = 40/s for s1-fast. The probe loads all UIDs at once, so N hotkeys
on a machine of throughput C each measure about C / N. Their combined factor is

    N × min(1, C / (N T)) = min(N, C / T),

while one hotkey on the same machine gets `min(1, C / T)`. For C = 160/s (one PRO 6000 on vLLM):

| Hotkeys on the machine | Factor each | Combined units | vs one hotkey |
|---|---|---|---|
| 1 | 1 (160 capped at 40) | 1 | 1× |
| 2 | 1 (80 capped) | 2 | 2× |
| 4 | 1 (40) | 4 | **4×** |
| 8 | 0.5 (20) | 4 | 4× |

So splitting pays until each share measures exactly the target. A "solo" machine is worth one share, whatever it
does. The public statement ("several hotkeys on one machine don't add anything") held only for machines at or below
40/s.

There is a second, quieter cause. Even with a linear factor, the probe could not *see* a fast machine through one
hotkey. It offered each UID two lanes of 16 decisions (32 in flight). A closed loop with round-trip time Z then shows
at most `16 L / (16/C + Z)` decisions/s (L = lanes):

| Machine C | Z = 0.05 s | Z = 0.10 s | Z = 0.25 s |
|---|---|---|---|
| 2 lanes (rc2), 160/s | 213 → 160 | 160 → 160 | 91 |
| 2 lanes (rc2), 226/s | 265 → 226 | 187 | 100 |
| 2 lanes (rc2), 640/s | 427 | 256 | 116 |
| **16 lanes (v2), 640/s** | 640 | 640 | 640 (930 bound) |

With two lanes, a 4-hotkey split (8 lanes) measured more than one hotkey even under a linear rule. The test
`test_few_probe_lanes_cannot_see_a_distant_fast_machine` reproduces this: over a 0.1 s link and a 2 s test window, a 320/s
machine measured 192/s alone and 288/s split over four hotkeys (2 lanes); with 16 lanes both measured 272/s (stable
over 3 runs; the rest is the in-window counting loss of a 2 s window). The default miner limits (2 concurrent requests and 20 requests a minute
per validator) cap a hotkey the same way (about 18 × 16 decisions per probe).

A third, small hole: each UID's requests were generated inside its own probe coroutine, at about 50 µs per decision
(measured). With 100 UIDs × 160 decisions that staggers the windows by about 0.8 s on a 3 s window, so siblings did
not fully overlap. A UID that ran out of requests was scored by "decisions / time to last answer", so a sibling that
stopped early freed the machine for the others.

## 3. Options

| | Rule | Sybil-neutral? | Pros | Cons |
|---|---|---|---|---|
| **(a)** | `ema × min(K, c / T)`, K far above one machine (s1-fast K = 16 → 640/s) | Yes for every machine up to K·T: N hotkeys measure ≤ C in total and earn ≤ one hotkey. Above K·T a split is paid exactly in proportion to throughput, never more. | One hotkey per machine is optimal; a hotkey with more GPUs earns more; the probe cost and the effect of one burst are bounded per hotkey | Pays for throughput nobody routes yet (paid traffic is small); pay concentrates on big fleets; the probe must be able to measure up to K·T (needs (d)) |
| (b) | `ema × c / T` (no cap) | Yes | Simplest rule | Probe cost per UID unbounded; one outlier measurement moves the whole pool; validators would need unbounded load capacity to measure it, so in practice the measurement ceiling becomes the cap anyway |
| (c) | aggregate per coldkey / per IP (e.g. min(1, Σc/T) per coldkey) | **No** | Keeps the old "cap per operator" idea | A new coldkey is free and unlinkable (today's 96 UIDs are already spread over 22 coldkeys, and spreading them over 96 costs nothing); IPs are cheap (VPS front-ends, proxies, IPv6 ranges) and shared by honest operators (cloud NAT, one datacenter, Lium/RunPod hosts) → false accusations. Also contradicts the published rule "one hotkey per node, many nodes per person are fine". Useful only as monitoring (DESIGN-POOLS 4.4) |
| **(d)** | probe changes (required by (a) or (b)) | Makes measurement match reality | See section 4 | Validators and miners must carry more probe traffic; miners need the new defaults (or the env vars) to be measured above ~55/s |

**Recommendation: (a) + (d)**, plus counting the tier-order rule in capacity units (section 5). Implemented on the
branch.

## 4. Probe changes (d) and their cost

| | rc2 | v2 |
|---|---|---|
| Factor | min(1, c/T) | min(K, c/T); K = 16 (s1-fast, s1-pro, s1-cpu), 10 (s1-lite) |
| Windows | each UID's own, after building its requests | **one common window**; all requests prepared before it opens (in a worker thread) |
| Counting | in-window, or decisions / time-to-last if the UID ran out | **in-window only**, divided by the whole window, so Σ over siblings ≤ C · W whatever they do |
| Lanes offered per UID | 2 | s1-fast 16, s1-pro 16, s1-lite 8, s1-cpu 4 (the miner accepts up to its own limit; a refused lane just ends, no penalty) |
| Requests prepared per UID | enough for the target | enough for 4× the UID's previous measurement (≥ 1 unit, ≤ cap), so a 600/s node is measured at full speed by its 2nd–3rd probe and the validator's cost follows real capacity |
| Window × items/request | fast 3 s × 16, pro 5 s × 8, lite 16 s × 8, cpu 30 s × 2 | fast **5 s** × 16, pro **8 s** × 8, lite 16 s × 8, cpu 30 s × **4** |
| Connection pool | the round's (256 connections, force-close) | probe-only pool sized for all lanes |
| Verified per UID and probe | ¼ of answers, 3..40 (cpu 8, pro 10) | unchanged: verification cost does **not** grow with throughput |
| Miner defaults | 2 concurrent / 20 per minute per validator, 4 running, 8 pending bodies per IP | **16 / 240**, 8 running, 32 pending; new `REAX_RUNTIME_INFLIGHT` (16) |

Validator cost per s1-fast probe, with 100 UIDs (today: 99):
- Preparation is about 50 µs and 376 bytes per decision (measured). At today's 40/s averages the budget is 160/s × 5 s
  = 800 decisions per UID, so 80,000 decisions: about 4 s of CPU in a worker thread and about 30 MB prepared. At the cap
  (all 100 UIDs at 640/s) it is 320,000 decisions: about 16 s and 120 MB. That is bounded by the cap, and in practice
  by 4× the real pool.
- Sent is what miners actually answer: Σ C × W. At 100 UIDs × 100/s that is 50,000 decisions, about 19 MB out in 5 s
  (about 30 Mbit/s).
- Reference verification stays at ≤ 40 decisions per UID: 4,000 per probe for 100 UIDs. That is about 73 s on a PRO
  6000 at 55/s, the same as rc2 (it was already that large; the lead may want `verified_max` 24 for s1-fast, see
  open issues).
- Partial garbage does not pay. If a share g of a probe's answers is garbage, 40 secret checks catch it (a strike, and
  the moving average restarts) with probability 1 − (1 − g)^40: g = 2 % → 55 %, 5 % → 87 %, 10 % → 98.5 %, against a
  throughput gain of at most 1/(1 − g).
- The counting error: only the requests still in flight at the window's end are lost, less than one request per lane.
  For a one-at-a-time runtime that is ≤ 16 decisions (≤ 8 % at the target, less above it). The same relative error
  applies to every hotkey on a machine, so it does not favour splitting.

What it means for the other pools:
- **s1-pro**: unit 10/s, cap 160/s (H100 108/s at concurrency 64). 16 lanes × 8 = 128 in flight is enough for the
  batched Rune build. Requests at the cap: 176 per probe.
- **s1-lite**: unit 8/s, cap 80/s (RTX 5070 Ti 33/s, A6000 24/s). Requests stay at 8 decisions, so an Orin at 0.7/s
  still finishes its first request in 11 s of the 16 s window. A big GPU running bf16 in s1-lite could exceed 80/s. A
  split is then paid in proportion only, and the tier-order rule keeps s1-lite's pay per unit at ≤ ½ of the GPU tier's.
  So that GPU earns more in s1-fast.
- **s1-cpu**: unit 1/s, cap 16/s (a 64-core server). 4 decisions per request: a 0.5/s CPU still answers within 8 s of
  the 30 s window.
- **Rate limits**: requests at the cap + 2 round requests stay ≤ the new default 240/min. Old-default miners (20/min,
  2 lanes) keep working and are measured up to about 18 × 16 / 5 s ≈ 58/s. Above that they must update or set the
  env vars. Raising a limit costs nothing, so there is never a reason to register a second hotkey instead.

## 5. Tier-order rule: count units, not hotkeys

The rule "average weight per miner in a tier ≤ ½ × that of the tier above" counted **hotkeys**. That is a second
sybil lever. With 50 GPU miners and one CPU node, the CPU tier is capped at 0.5 × the GPU average × 1. Split the CPU
node into 10 hotkeys and its tier cap grows ten times: with the rc2 code the ten hotkeys together earned 0.0588 (the CPU
tier's whole budget) against 0.0094 for the solo node, **6.25×**. v2 counts capacity units (Σ factor; a pool without capacity data counts its miners). A tier full of healthy
target-speed nodes has exactly as many units as miners, so the published 80/15/5 tables are unchanged
(`test_tier_order_counts_units_so_target_speed_tiers_behave_as_before`).

## 6. Effect on today's miners (estimate)

- Pay in s1-fast moves from "equal per hotkey" to "proportional to verified throughput up to 640/s per hotkey". An
  operator's total no longer depends on how many hotkeys they hold. It depends on how many decisions per second their
  machines actually verify.
- Today every hotkey measures ≥ 40/s at the same moment. So each coldkey with N hotkeys already delivers ≥ 40·N
  decisions/s in the probe. Those totals stay. What changes is the part *above* 40/s per hotkey, which rc2 never
  measured. Operators who registered many hotkeys on few fast GPUs keep their total. Operators running one hotkey per
  fast GPU (e.g. vLLM bf16 on an H100, 226/s) gain up to 5.6× per hotkey. Without the probe log (section 1) I cannot
  say who is who.
- The batched vLLM bf16 path (2–3× the reference path, "honest, passes, at own risk" in CALIBRATION.md) becomes
  clearly more profitable. Expect miners to move to it: customers get the same model within tolerance, and it loses
  about 0.16 % of rounds.
- Transition: stored capacity averages are ≤ 40 (clamped by rc2). After the upgrade the next probe prepares 160/s per
  UID, then 4× each probe, so fast hotkeys reach their speed within 2–3 probes (about 1 h at one probe per ~10 rounds).
- Registrations: the incentive to register more hotkeys for the same hardware disappears, so pressure on the burn
  price from that motive should drop.

## 7. Suggested public wording (for the lead; no returns or prices, MiCAR)

> Capacity scoring change (announced before it goes live): inside a pool, pay will follow verified throughput linearly
> up to a per-hotkey cap far above one machine (s1-fast: 640 decisions/s). All hotkeys of a pool are still probed at
> the same moment, so several hotkeys on one machine together measure what the machine does. One hotkey per machine is
> the best setup; a hotkey with more GPUs earns more. Miners should update to the new release (or set
> REAX_PER_CALLER_CONCURRENCY=16 and REAX_PER_CALLER_PER_MINUTE=240) so a probe can measure their full speed.

## 8. Open issues

- Live per-UID probe numbers are missing (section 1). Grep the validator log before announcing figures.
- `weights_version` is not bumped on the branch (scoring changes). The lead decides the release number (I called it
  "Unreleased" in CHANGELOG).
- Not measured on a GPU: real vLLM or reference runtimes behind the 16-lane probe, and the validator's CPU and
  bandwidth with about 100 real miners at once. The fake-machine tests use one-at-a-time machines on localhost.
- s1-fast reference verification per probe is already about 73 s for 100 UIDs (unchanged). If validators run short
  of time, `verified_max` 24 still catches 5 % garbage with 71 % probability.
- Paying for throughput above demand is an economic choice. The router routes little paid traffic yet, and K = 16
  makes raw capacity the main lever inside s1-fast. A lower K (e.g. 8 = 320/s, still above any single GPU measured)
  halves the probe cost at the cap and stays sybil-neutral for single-GPU machines. Multi-GPU nodes above 320/s would
  then be paid in proportion through extra hotkeys, which is still never more than their throughput.
