"""Fixed chat templates of the specialist track (docs/SPECIALIST-TRACKS.md, "Chat templates").

A submission's own ``chat_template.jinja`` / ``tokenizer_config.json["chat_template"]`` is never rendered: a template is
code (Jinja) and the validator host parses it inside the vLLM container that holds the GPU. Every allowed architecture
maps to one template shipped in this package (``templates/*.jinja``, plain ``messages`` rendering, no tools), passed to
vLLM with ``--chat-template``. Architectures without an entry are refused by the submission rules.
"""

from __future__ import annotations

import os
from typing import Mapping

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# architecture (config.json) -> template file
TEMPLATE_FOR_ARCH: Mapping[str, str] = {
    "Qwen3ForCausalLM": "qwen.jinja",
    "Qwen3MoeForCausalLM": "qwen.jinja",
    "Qwen3_5ForCausalLM": "qwen.jinja",
    "Qwen3_5ForConditionalGeneration": "qwen.jinja",
    "Gemma3ForCausalLM": "gemma3.jinja",
    "Gemma3ForConditionalGeneration": "gemma3.jinja",
    "LlamaForCausalLM": "llama3.jinja",
}

# Files of a submission that carry a chat template or parse as code-like input and are never used (deleted from the
# validator's copy after the download; see sanitize_dir).
TEMPLATE_FILE_PREFIX = "chat_template."


def template_name(architecture: str) -> str | None:
    return TEMPLATE_FOR_ARCH.get(architecture)


def template_path(name: str) -> str:
    return os.path.join(TEMPLATE_DIR, name)
