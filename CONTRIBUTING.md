# Contributing

Thanks for helping with REAX. The code is MIT licensed (see [LICENSE](LICENSE)).

## Run the tests

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
python -m compileall -q src tests scripts
python -m pytest -q
```

Tests that need a downloaded model are skipped unless the model is cached (`-m "not model"` skips them explicitly).
CI runs the suite on Python 3.10-3.13 and once against the hash-pinned `requirements/lock.txt`.

## Pull requests

* Keep a change focused and explain what it does and why; link the issue if there is one.
* Add or update tests for behaviour changes. Do not weaken an existing test to make it pass.
* Scoring, manifest or wire-protocol changes need a new `WEIGHTS_VERSION_KEY`, a CHANGELOG entry and a calibration
  note (see [docs/CALIBRATION.md](docs/CALIBRATION.md)), because validators must agree on them.
* Dependencies are pinned exactly and hash-locked in `requirements/lock.txt`; say why when you change one.
* Match the style of the surrounding code.

## No secrets

Never commit wallets, mnemonics, coldkeys, hotkeys, tokens, `.env` files or validator state, and do not paste them
into issues or pull requests. Keep coldkeys off mining and validating machines.

## Security issues

Do not open a public issue for a vulnerability. Follow [SECURITY.md](SECURITY.md) (GitHub private vulnerability reporting).
