"""Chat template rendering and tokenization, exactly as Hugging Face does them, without importing transformers.

- **Template:** `chat_template.jinja`, else `tokenizer_config.json`'s `chat_template` (a string, or a list of
  named templates: "tool_use" when tools are given and present, else "default"; HF's `get_chat_template`).
- **Special tokens:** the named ones from `tokenizer_config.json` (`bos_token`, `eos_token`, ...), passed to the
  template like `apply_chat_template` does, with request kwargs taking precedence.
- **Tokenizer:** `tokenizers.Tokenizer.from_file("tokenizer.json")`, as shipped. Never transformers' class-based
  loading: it builds a wrong tokenizer for Step-3.5.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path
from typing import Any

import jinja2.exceptions
import jinja2.meta
import numpy as np
from tokenizers import Tokenizer

from .content_format import ContentFormat, detect_content_format
from .jinja_env import MAX_OPERATIONS, compile_template, make_environment, metered, operations
from .token_cache import SegmentCache, pick_split_token

log = logging.getLogger(__name__)

# transformers' SPECIAL_TOKENS_ATTRIBUTES: the named special tokens apply_chat_template passes to templates.
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")
# Never taken from chat_template_kwargs: the template variables `render` passes itself, and `apply_chat_template`'s
# own `chat_template` and `tokenize`, even where a template names them (vLLM drops both too)
RESERVED_TEMPLATE_VARS = frozenset({"messages", "tools", "documents", "add_generation_prompt", "chat_template",
                                    "tokenize"})
_PROBE = "\x00llm-mini-infer cache hint probe\x00"  # the content of the probe turn that locates a message's end
MAX_CACHE_HINTS = 4  # a request's hints that count, the deepest: Anthropic's limit (`Renderer.hint_offsets`)


class TemplateError(ValueError):
    """The chat template rejected the conversation (e.g. via `raise_exception`): a 400 for the client. `select_template`
    raises it too, when the model has no chat template to use."""


def _reason(exc: BaseException) -> str:
    """The 400's whole message, as vLLM answers a template error: the first Jinja `TemplateError` in the chain,
    else the error itself."""
    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, jinja2.exceptions.TemplateError):
            return str(cur)
        cur = cur.__cause__ or cur.__context__
    return str(exc)


def _token_content(value: Any) -> str | None:
    if isinstance(value, dict):  # serialized AddedToken
        value = value.get("content")
    return None if value is None else str(value)


def load_special_tokens(tokenizer_config: dict[str, Any]) -> dict[str, str]:
    special = {}
    for key in SPECIAL_TOKEN_KEYS:
        content = _token_content(tokenizer_config.get(key))
        if content is not None:
            special[key] = content
    return special


def load_chat_templates(model_dir: Path, tokenizer_config: dict[str, Any]) -> str | dict[str, str] | None:
    path = model_dir / "chat_template.jinja"
    if path.exists():
        return path.read_text()
    tpl = tokenizer_config.get("chat_template")
    if isinstance(tpl, list):
        return {t["name"]: t["template"] for t in tpl}
    return tpl


def select_template(templates: str | dict[str, str] | None, has_tools: bool) -> str:
    if isinstance(templates, dict):
        if has_tools and "tool_use" in templates:
            return templates["tool_use"]
        if "default" in templates:
            return templates["default"]
        raise TemplateError(f"the model has several chat templates and no default: {sorted(templates)}")
    if templates is None:
        raise TemplateError("the model has no chat template")
    return templates


class Renderer:
    """Loaded once per frontend process."""

    def __init__(self, model_dir: str | Path, tokenizer_dir: str | Path | None = None):
        self.model_dir = Path(model_dir)
        tok_dir = Path(tokenizer_dir) if tokenizer_dir else self.model_dir
        cfg_path = tok_dir / "tokenizer_config.json"
        self.tokenizer_config: dict[str, Any] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.special_tokens = load_special_tokens(self.tokenizer_config)
        self.templates = load_chat_templates(tok_dir, self.tokenizer_config)
        self.tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        if self.templates is None:
            log.warning("%s has no chat template: /v1/chat/completions will fail", tok_dir)

    @cached_property
    def default_template(self) -> str:
        return select_template(self.templates, has_tools=False)

    @cached_property
    def content_format(self) -> ContentFormat:
        return detect_content_format(self.default_template)

    @cached_property
    def template_variables(self) -> frozenset[str]:
        ast = make_environment().parse(self.default_template)
        return frozenset(jinja2.meta.find_undeclared_variables(ast))

    @cached_property
    def supports_developer_role(self) -> bool:
        tpl = self.default_template
        return '"developer"' in tpl or "'developer'" in tpl

    def filter_template_kwargs(self, kwargs: dict[str, Any] | None) -> dict[str, Any]:
        """The template's variables in `chat_template_kwargs`: keep those the template uses, drop the rest and the
        reserved ones (`chat_template` and `tokenize` among them) silently. A request's own `chat_template` is refused
        before (`api.protocol.check_chat_template`)."""
        if not kwargs:
            return {}
        allowed = self.template_variables - RESERVED_TEMPLATE_VARS
        return {k: v for k, v in kwargs.items() if k in allowed}

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
               add_generation_prompt: bool = True, **kwargs: Any) -> str:
        """`tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, ...)`, exactly. Within the budget in
        force (`jinja_env.operations`), else one of its own (`MAX_OPERATIONS`); past it, a TemplateError."""
        source = select_template(self.templates, has_tools=tools is not None)
        template_kwargs = {**self.special_tokens, **kwargs}
        try:
            with nullcontext() if metered() else operations(MAX_OPERATIONS):
                return compile_template(source).render(messages=messages, tools=tools, documents=None,
                                                       add_generation_prompt=add_generation_prompt, **template_kwargs)
        except Exception as exc:  # raise_exception, undefined fields, list content in text-only templates, ...
            raise TemplateError(_reason(exc)) from exc

    @cached_property
    def segment_cache(self) -> SegmentCache | None:
        probe = None
        if self.templates:
            try:
                probe = self.render([{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "a"}, {"role": "user", "content": "v"}])
            except TemplateError:
                try:
                    probe = self.default_template
                except TemplateError:
                    probe = None  # a template list with no default: no cache hint; requests answer 400 instead
        pick = pick_split_token(self.tokenizer, probe)
        if pick is None:
            log.info("no safe split token in the chat template: prompt tokenization is not cached")
            return None
        return SegmentCache(self.tokenizer, *pick)

    def encode(self, text: str, add_special_tokens: bool = False) -> np.ndarray:
        """Token IDs of `text`, identical to `tokenizer.encode(text, add_special_tokens=add_special_tokens).ids`
        (`encode_batch_fast` skips the offsets; long repeated prompt parts come from the segment cache)."""
        if not add_special_tokens and self.segment_cache is not None:
            return self.segment_cache.encode(text)
        return np.asarray(self.tokenizer.encode_batch_fast([text], add_special_tokens=add_special_tokens)[0].ids,
                          dtype=np.int32)

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    @cached_property
    def neutral(self) -> Any:
        """The tool results' tokenization (`tool_results.Neutral`)."""
        from .tool_results import Neutral

        return Neutral(self.tokenizer)

    # -- cache hints ---------------------------------------------------------------------------------------------

    @cached_property
    def _special_texts(self) -> tuple[str, ...]:
        return tuple(t.content for t in self.tokenizer.get_added_tokens_decoder().values() if t.special and t.content)

    def hint_offsets(self, messages: list[dict[str, Any]], hinted: list[int], text: str, ids: np.ndarray,
                     **kwargs: Any) -> list[int]:
        """Token offsets in the prompt `ids` (rendered as `text`) where each hinted message ends, sorted
        (architecture.md, "Cache hints"). The text up to a message's end must tokenize to exactly the prompt's first
        tokens and end before the prompt does; other hints are dropped. Of more than `MAX_CACHE_HINTS`, the deepest
        count: each costs a render of the conversation up to its message."""
        hinted = sorted(hinted)
        if len(hinted) > MAX_CACHE_HINTS:
            log.debug("%d cache hints: the deepest %d count", len(hinted), MAX_CACHE_HINTS)
            hinted = hinted[-MAX_CACHE_HINTS:]
        out = set()
        for i in hinted:
            end = self._message_end(messages, i, text, kwargs)
            if not end:
                continue
            head = self.encode(text[:end])
            n = len(head)
            if 0 < n < len(ids) and np.array_equal(ids[:n], head):
                out.add(n)
            else:
                log.debug("cache hint on message %d: no token boundary inside the prompt at character %d", i, end)
        return sorted(out)

    def _message_end(self, messages: list[dict[str, Any]], i: int, text: str, kwargs: dict[str, Any]) -> int | None:
        try:
            prefix = self.render(messages[:i + 1], add_generation_prompt=False, **kwargs)
            if text.startswith(prefix) and prefix:
                return len(prefix)
        except TemplateError:
            pass
        try:
            probe = self.render([*messages[:i + 1], {"role": "user", "content": _PROBE}], add_generation_prompt=False,
                                **kwargs)
        except TemplateError:
            return None
        n = 0
        limit = min(len(text), len(probe))
        while n < limit and text[n] == probe[n]:
            n += 1
        start = max((text.rfind(t, 0, n) for t in self._special_texts), default=-1)
        return start if start > 0 else None
