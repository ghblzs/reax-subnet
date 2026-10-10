#!/usr/bin/env python3
"""Create a REAX test subnet on a LOCAL subtensor with throwaway dev wallets.

Never point this at finney or test: it funds roles from the public //Alice dev
account, which only has money on local chains. Writes <work>/subnet.json.

Usage: bootstrap.py <work_dir> <role> [<role> ...]   (role names like v1, m1)
Roles starting with "v" are staked as validators.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

import bittensor as bt
from bittensor import calls
from bittensor.keyfiles import Keypair

ENDPOINT = os.environ.get("REAX_NETWORK", "ws://127.0.0.1:9951")


class _Dev:
    def __init__(self, kp):
        self.coldkey = self.coldkeypub = self.hotkey = kp


def _ok(result, what):
    if not result.success:
        sys.exit(f"{what} failed: {result.message}")


def main() -> None:
    if not ENDPOINT.startswith(("ws://127.0.0.1", "ws://localhost")):
        sys.exit("bootstrap.py only runs against a local chain")
    work = os.path.abspath(sys.argv[1])
    roles = sys.argv[2:]
    full = os.path.join(work, "wallets-full")
    server = os.path.join(work, "wallets-server")
    st = bt.Subtensor(network=ENDPOINT)
    alice = _Dev(Keypair.create_from_uri("//Alice"))

    def wallet(name):
        w = bt.Wallet(name=name, hotkey="default", path=full)
        w.create_new_coldkey(use_password=False, suppress=True, overwrite=True)
        w.create_new_hotkey(use_password=False, suppress=True, overwrite=True)
        # Server copy: hotkey and public coldkey only. Miners and validators never need the coldkey.
        os.makedirs(os.path.join(server, name, "hotkeys"), exist_ok=True)
        shutil.copy2(os.path.join(full, name, "hotkeys", "default"), os.path.join(server, name, "hotkeys", "default"))
        shutil.copy2(os.path.join(full, name, "coldkeypub.txt"), os.path.join(server, name, "coldkeypub.txt"))
        return w

    owner = wallet("owner")
    _ok(st.execute(bt.Transfer(owner.coldkeypub.ss58_address, 5000), alice), "fund owner")
    before = {s.netuid for s in st.subnets.all()}
    _ok(st.execute(bt.RegisterSubnet(owner.hotkey.ss58_address), owner), "register subnet")
    netuid = max({s.netuid for s in st.subnets.all()} - before)
    _ok(st.execute(bt.StartCall(netuid), owner), "start call")
    out = {"endpoint": ENDPOINT, "netuid": netuid, "owner_hotkey": owner.hotkey.ss58_address, "roles": {}}
    for role in roles:
        w = wallet(role)
        _ok(st.execute(bt.Transfer(w.coldkeypub.ss58_address, 200), alice), f"fund {role}")
        # Raw call: the SDK routes BurnedRegister through the MEV shield, which a
        # fast-block local chain does not decrypt reliably.
        _ok(st.submit_call(calls.SubtensorModule.burned_register(netuid=netuid, hotkey=w.hotkey.ss58_address), w),
            f"register {role}")
        if role.startswith("v"):
            _ok(st.execute(bt.AddStake(w.hotkey.ss58_address, netuid, 100, slippage_protection=False), w), f"stake {role}")
        out["roles"][role] = {"hotkey": w.hotkey.ss58_address, "uid": st.neurons.uid(w.hotkey.ss58_address, netuid)}
        time.sleep(0.5)
    out["hyperparameters"] = st.subnets.subnet_hyperparameters(netuid)
    with open(os.path.join(work, "subnet.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "hyperparameters"}, indent=1))


if __name__ == "__main__":
    main()
