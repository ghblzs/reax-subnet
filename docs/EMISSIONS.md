# Emissions, owner share and burn

## What the chain does (dynamic TAO)

Each subnet's emission is split by the chain, not by this code: the **subnet owner cut** (18 % by default; check the live chain) goes
to the owner, and the rest is shared between miners (by incentive, from validator weights) and validators
(by dividends, from stake and bonds). This code only decides the **miner weights** each validator submits.

## Roles

* **Subnet owner** = the account that registered the subnet and holds the owner coldkey. The chain pays the owner cut to that coldkey.
* **Operator** = the REAX maintainers, who develop and maintain this code and may run a reference
  validator. Nothing in this repository moves TAO or pays anyone.

The owner-cut percentage is a chain parameter and can change; verify its current value on the live chain.

## The one knob in code: `REAX_BURN_FRACTION`

Validators can give a fixed share of every weight vector to the owner UID (the UID of the subnet owner hotkey).
Weight on the owner UID earns no miner reward for anyone except the owner hotkey; whether the chain pays it to
the owner or recycles it depends on the runtime version, so check on testnet before relying on either effect.

* Default `0.0`: all miner weight goes to miners that pass the fidelity check.
* When **no** miner qualifies (start-up, outage), validators put all weight on the owner UID instead of setting
  nothing, so they keep their validator trust. This is automatic and independent of the setting.
* Any non-zero value must be agreed among validators and set identically on every validator; otherwise
  validators disagree and lose trust.
* `REAX_BURN_FRACTION=1` is also the emergency brake: it stops all miner rewards once validators holding most of
  the stake have switched and submitted new weights (consensus is stake-weighted).
* With a partial burn (0 < value < 1) and no owner UID on the subnet, the burn has no recipient and is ignored:
  miners are paid in full. The owner hotkey is registered when the subnet is created, so this only matters if
  that UID was deregistered.
