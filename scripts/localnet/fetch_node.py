#!/usr/bin/env python3
"""Fetch the official subtensor localnet image layers without a Docker daemon.

Downloads ghcr.io/raofoundation/subtensor-localnet:<tag> (amd64), verifies
every layer's sha256 against the manifest and unpacks the node binaries plus
the glibc they need into <dest>/rootfs. Usage: fetch_node.py <dest> [tag-or-digest]

The default is the exact amd64 manifest used for the 1.0 end-to-end evidence
(published as :latest on 3 Oct 2026, config sha256:1c28df70...).
"""

import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request

REPO = "raofoundation/subtensor-localnet"
DEFAULT_REF = "sha256:8f5a52f62bc49e16dffc7113d732394d948a5c0d0012826df093cbed0ba7030b"
ACCEPT = ",".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def main() -> None:
    dest = os.path.abspath(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REF
    token = json.load(urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{REPO}:pull"))["token"]

    def get(path):
        req = urllib.request.Request(
            f"https://ghcr.io/v2/{REPO}/{path}", headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT}
        )
        return urllib.request.urlopen(req)

    index = json.load(get(f"manifests/{tag}"))
    if "manifests" in index:
        digest = next(m["digest"] for m in index["manifests"] if m.get("platform", {}).get("architecture") == "amd64")
        manifest = json.load(get(f"manifests/{digest}"))
    else:
        digest, manifest = tag, index
    rootfs = os.path.join(dest, "rootfs")
    os.makedirs(rootfs, exist_ok=True)
    for layer in manifest["layers"]:
        data = get(f"blobs/{layer['digest']}").read()
        if "sha256:" + hashlib.sha256(data).hexdigest() != layer["digest"]:
            sys.exit(f"digest mismatch for {layer['digest']}")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(rootfs, filter="data")
            else:  # pragma: no cover - Python without the extraction filter
                sys.exit("this Python lacks tarfile extraction filters; use Python >= 3.10.12/3.11.4/3.12")
    with open(os.path.join(dest, "IMAGE.json"), "w") as fh:
        json.dump({"repository": REPO, "tag": tag, "manifest": digest, "config": manifest["config"]["digest"]}, fh, indent=1)
    print(f"unpacked {REPO}:{tag} ({digest}) into {rootfs}")


if __name__ == "__main__":
    main()
