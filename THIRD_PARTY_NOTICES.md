# Third-party notices

This repository's code is licensed under MIT; see [LICENSE](LICENSE). Model weights are not included in this repository or its
images. Operators obtain them from the upstream repositories at the pinned revisions and remain subject to the upstream terms.

| Component | Licence and source | Attribution and handling |
|---|---|---|
| Plumb-4B weights | Apache-2.0; [pinned Hugging Face revision](https://huggingface.co/crh225/plumb-4b/tree/55de037801a8a9b9de3db5c0e16cef86210c2186) | The model repository contains its own `LICENSE`, `NOTICE` and `NOTICE-jevk5` files. Those upstream files are not part of this repository; consult them when obtaining or redistributing the weights. |
| JevK5 runtime | Apache-2.0; [source at pinned commit](https://github.com/allebee/jevk5/tree/85238d7be5527370c43206fe54cd752eb3134c1b) | `LICENSES/JevK5-LICENSE.txt` and `LICENSES/JevK5-NOTICE.txt` reproduce the upstream files at that commit. The notice identifies the JevK5 authors and third-party work. |
| SemIf prompt and readout adaptation | MIT; [upstream source at pinned commit](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a) | Adapted through JevK5. The full upstream MIT licence is included at `LICENSES/SemIf-MIT.txt`; the JevK5 notice also attributes the adaptation. |
| Qwen3.5-4B, the base of Plumb-4B | Apache-2.0; see the Plumb model repository above | The Qwen attribution is included in `LICENSES/JevK5-NOTICE.txt`. No model weights are redistributed here. |
| Surogate Rune 26B-A4B v3 weights | Apache-2.0; [pinned Hugging Face revision](https://huggingface.co/surogate/rune-26b-a4b-GGUF/tree/c6b360d47895bb77bdf3805a13ee5a5557ab1921) | The repository is gated. Each operator obtains access through their own account; REAX does not mirror the weights. |
| Google Gemma 4 26B-A4B-it base of Rune | Apache-2.0; [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4) | The upstream model card identifies Google DeepMind as the author. |
| `runtimes/s1-pro/chat_template_nospace.jinja` | Apache-2.0; source template distributed with the Gemma 4 / Rune weights | Modified by the REAX authors for generation-prompt whitespace handling. The file header identifies the source and modification; see `LICENSES/Apache-2.0.txt`. |
| vLLM 0.30.0, PyTorch 2.13, Transformers, Accelerate, Flash Linear Attention, Einops, FastAPI, Uvicorn, Bittensor SDK, aiohttp and pinned dependencies | Their respective upstream licences; see the pinned image definitions and `requirements/` lock files | Images install or use these components from their upstream distributions. Their licence and attribution metadata remains with those distributions. |

The complete Apache License 2.0 text is included at `LICENSES/Apache-2.0.txt`. Runtime images include this directory, this notice file and the repository `LICENSE` under `/licenses/`.
