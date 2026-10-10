# Feature flags

## Switched off in this release (cannot be enabled)

These flags are reserved for features that are not included in this release.
Setting any of the variables below to `1` stops the miner or validator at startup with an error, so nobody can
believe a feature is active when it is not.

| Variable | Feature | Why it is off | What is needed to turn it on |
|---|---|---|---|
| `REAX_FEATURE_GDPR_TIER` | Regional admission controls | An admission issuer, revocation support, policy review and testing are required | Complete the admission workflow and validate it on a development network |
| `REAX_FEATURE_ORGANIC_TRAFFIC` | Routing requests from external clients | No external-request route is provided in this release | A reviewed service and privacy safeguards are required |
| `REAX_FEATURE_IMAGES` | Image inputs | Image decoding is an attack surface (decompression bombs, parser bugs) and changes the reference cost | Isolated decoder with limits, image-capable approved model, re-calibration |
| `REAX_FEATURE_CAPACITY_PROBES` | Burst capacity audits and capacity-weighted rewards | Not included in this release | Complete the design, implementation and validation |
| `REAX_FEATURE_MULTI_MODEL` | More than one approved model **in the mpm1 (rc3) mechanism** | Each model needs its own calibration and reference compute on validators | Per-model tolerance measurement, catalog signing, validator hardware plan. Under mechanism s1 the two production models are separate *pools* instead; the second pool (s1-pro) has its own switch below |
| `REAX_FEATURE_SCORE_AND_NOUL_TYPES` | Ordered-score and "noul" question types | Only finite-choice items have a calibrated tolerance | Extraction method + calibration per type |
| `REAX_FEATURE_LABELLED_ITEMS` | Scoring against items with known answers | A labelled pool can leak and be memorised; MPM avoids answer keys entirely | Rotation/sealing process, leakage monitoring, review |

## Real switches (on/off)

| Variable | Default | Effect |
|---|---|---|
| `REAX_MECHANISM` | `s1` | `s1` = release 1.1 mechanism (miners serve exactly the System1 production models, protocol `reax-s1/1`); `mpm1` = the unchanged rc3 mechanism (Qwen3-4B, protocol `reax-mpm/1`) |
| `REAX_FEATURE_S1_PRO_POOL` | `0` (fixed by the release in 1.1) | `1` enables the s1-pro pool: shares become s1-fast 0.35 / s1-pro 0.65 and **every validator then needs an 80–96 GB GPU** for the s1-pro reference runtime. Overridable only on development networks (`REAX_ALLOW_DEV_MODELS=1`, refused on finney); enabling it on mainnet needs a new release |
| `REAX_PUBLISH_COMMITMENT` | `1` | miner (mechanism s1): publish the on-chain commitment `reax-s1/1:<model>:<manifest_id>` on start. `0` skips publishing (use when the commitment is already set); without a matching commitment validators do not query the miner |
| `REAX_SET_WEIGHTS` | `1` | `0` runs the validator in shadow mode: it scores and logs weights but never submits them |
| `REAX_SERVE_AXON` | `1` | `0` stops the miner from publishing its endpoint (use when it is already published) |
| `REAX_ALLOW_DEV_MODELS` | `0` | Allows the small test models in the catalog. Never on mainnet |
| `REAX_ALLOW_PRIVATE_IPS` | `0` | Lets validators query private/loopback miner addresses. Local test networks only; link-local addresses stay blocked |
| `REAX_BURN_FRACTION` | `0.0` | Share of each weight vector given to the subnet owner UID. Must be identical on all validators; set only by agreement among validators |
| `REAX_LOG_JSON` | `0` | JSON log lines for log shippers |
