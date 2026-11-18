"""OpenAI chat-completions and /tokenize request types (msgspec), and their validation.

- Unknown fields are a 400 (`forbid_unknown_fields`), and so is a body nested deeper than msgspec decodes (about
  10,000 levels: CPython's C recursion limit) or one with a string that is not UTF-8.
- Fields we know but do not implement are accepted only at their default values (`n=1`, neutral penalties, ...) and
  are a 400 otherwise.
- What vLLM refuses too is refused with its message, `(parameter=P, value=V)` suffix and param
  (`validation_error`).
- **`logprobs`, `top_logprobs`** (`logprobs_param`): vLLM's checks, messages and limit (20: the OpenAI API's, and
  the default of vLLM's `--max-logprobs`); `top_logprobs` -1 is the whole vocabulary, so over it. As in vLLM,
  `top_logprobs` null is absent, 0: with `logprobs` true, the sampled tokens' logprobs and no top ones.
- **Tools** (`tools`, `tool_choice`, `parallel_tool_calls`; architecture.md, "Tool calls"): `check_tool_usage` has
  vLLM's checks, order, messages and params; a malformed tool is a 400 of ours (`api.tools.check_tools`).
  `tool_choice` absent and null differ (UNSET): absent, it is "auto" with tools and "none" without.
- **Reasoning** (architecture.md, "Reasoning"): `reasoning_effort` is one of vLLM's seven efforts or null, else its
  400 (`check_reasoning_effort`); `include_reasoning` false leaves the thought out. `thinking_token_budget` is an
  unknown field.
- **Messages:** unknown keys in a message, in a tool call or in its `function` are ignored, as the OpenAI clients'
  tool loops echo the assistant's message back (`function_call`, `audio`, `annotations`, a call's `index`). An
  assistant's tool call has an `id`, the type `function`, a `name` and string `arguments`: anything else is a 400, the
  type `custom` with vLLM's message.
- `min_tokens` and content parts other than text, refusal and `image_url` are a 400. `image_url` parts (a data URI,
  or an http(s) URL with `images.allow_urls`; OpenAI's object or a bare string) are taken when the model has a vision
  tower and `images.enabled` (`api.images`); otherwise they are a 400 too.
- `return_token_ids`: the prompt's and the output's token ids in the response (`prompt_token_ids`, a choice's
  `token_ids`; streamed, the prompt's in the first chunk and each chunk's own).
- **`response_format`** (`grammar`): OpenAI's `{"type": "text"}`, `{"type": "json_object"}` and
  `{"type": "json_schema", "json_schema": {"name", "description"?, "schema", "strict"?}}`, strict or not (`strict` is
  ignored: the schema is always enforced, as vLLM does). The schema stays the JSON text the request carries
  (`msgspec.Raw`) until the frontend checks it (`App.grammar`). vLLM's `structured_outputs` field and the `guided_*`
  fields are unknown fields here, and a `response_format` of type `structural_tag` is a 400 too.
- **`chat_template`** (`check_chat_template`): a template the request supplies, top-level or not null in its own
  `chat_template_kwargs`, is refused with vLLM's 400; every request renders with the model's template.
- **Cache hints**: `"cache_control": {"type": "ephemeral"}` on a message or on its last content part
  (architecture.md, "Cache hints"); any other value or place is a 400.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import msgspec


class ApiError(Exception):
    def __init__(self, status: int, message: str, param: str | None = None, kind: str = "BadRequestError"):
        super().__init__(message)
        self.status, self.message, self.param, self.kind = status, message, param, kind

    def body(self) -> bytes:
        return msgspec.json.encode({"error": {"message": self.message, "type": self.kind, "param": self.param,
                                              "code": self.status}})


def validation_error(message: str, parameter: str | None = None, value: Any = None, cls: type = ApiError) -> ApiError:
    """A 400 as vLLM answers a request check it fails: the message followed by `(parameter=P, value=V)` with
    those given, param P."""
    extras = [f"{k}={v}" for k, v in (("parameter", parameter), ("value", value)) if v is not None]
    return cls(400, f"{message} ({', '.join(extras)})" if extras else message, parameter)


class _Strict(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    pass


class CacheControl(_Strict):
    type: Literal["ephemeral"]


class TextPart(_Strict, tag="text", tag_field="type"):
    text: str
    cache_control: CacheControl | None = None


class ImageURLPart(_Strict, tag="image_url", tag_field="type"):
    image_url: Any
    cache_control: CacheControl | None = None
    uuid: str | None = None  # vLLM's media identifier; accepted and ignored (we hash the bytes)


class ImagePart(_Strict, tag="image", tag_field="type"):
    image: Any


class InputAudioPart(_Strict, tag="input_audio", tag_field="type"):
    input_audio: Any


class VideoURLPart(_Strict, tag="video_url", tag_field="type"):
    video_url: Any


class FilePart(_Strict, tag="file", tag_field="type"):
    file: Any


class RefusalPart(_Strict, tag="refusal", tag_field="type"):
    refusal: str


ContentPart = TextPart | ImageURLPart | ImagePart | InputAudioPart | VideoURLPart | FilePart | RefusalPart
_MEDIA_PARTS = (ImageURLPart, ImagePart, InputAudioPart, VideoURLPart, FilePart)


class _Lenient(msgspec.Struct, omit_defaults=True):
    """Unknown keys are ignored (module docstring, "Messages")."""


class FunctionCall(_Lenient):
    name: str
    arguments: str


class ToolCall(_Lenient, tag="function", tag_field="type"):
    id: str
    function: FunctionCall


class CustomToolCall(_Lenient, tag="custom", tag_field="type"):
    """The OpenAI API's custom tool call: a 400 (`check_messages`)."""


class Message(_Lenient):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall | CustomToolCall] | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    cache_control: CacheControl | None = None

    @property
    def cache_hint(self) -> bool:
        """A cache_control marker on the message or on its last content part."""
        if self.cache_control is not None:
            return True
        parts = self.content
        return isinstance(parts, list) and bool(parts) and getattr(parts[-1], "cache_control", None) is not None


class JsonSchemaSpec(_Strict):
    name: str
    description: str | None = None
    schema: msgspec.Raw = msgspec.Raw()  # its JSON text as received (module docstring); empty when absent
    strict: bool | None = None


class TextFormat(_Strict, tag="text", tag_field="type"):
    pass


class JsonObjectFormat(_Strict, tag="json_object", tag_field="type"):
    pass


class JsonSchemaFormat(_Strict, tag="json_schema", tag_field="type"):
    json_schema: JsonSchemaSpec | None = None


ResponseFormat = TextFormat | JsonObjectFormat | JsonSchemaFormat


class StreamOptions(_Strict):  # unset fields: vLLM refuses only a non-empty object without `stream`
    include_usage: bool | msgspec.UnsetType = msgspec.UNSET
    continuous_usage_stats: bool | msgspec.UnsetType = msgspec.UNSET


class ChatCompletionRequest(_Strict):
    messages: list[Message]
    model: str | None = None
    # Supported
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = None  # null or absent: 0 (module docstring)
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template: str | None = None  # refused unless null (`check_chat_template`)
    add_generation_prompt: bool = True
    skip_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    priority: int = 0
    cache_salt: Any = None  # a string or None: `check_cache_salt` refuses the rest with vLLM's message
    user: str | None = None
    # Known, accepted only at their defaults
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logit_bias: dict[str, float] | None = None
    spaces_between_special_tokens: bool = True
    continue_final_message: bool = False
    echo: bool = False
    response_format: ResponseFormat | None = None  # supported (`App.grammar`)
    # Tool calls (module docstring): checked by `check_tool_usage` and `api.tools`
    tools: list[Any] | None = None
    tool_choice: Any = msgspec.UNSET  # absent: "auto" with tools, "none" without; null differs (module docstring)
    parallel_tool_calls: bool | None = None  # None: true
    return_token_ids: bool | None = None
    # Reasoning (module docstring): one of `output.reasoning.EFFORTS` (`check_reasoning_effort`), and whether the reply
    # carries the thought
    reasoning_effort: Any = None
    include_reasoning: bool = True
    # Not yet
    min_tokens: int = 0


class TokenizeRequest(_Strict):
    """/tokenize in the shape of vLLM's endpoint, for clients written against it: either `prompt` (completion form) or
    `messages` (chat form)."""

    model: str | None = None
    prompt: str | None = None
    messages: list[Message] | None = None
    add_special_tokens: bool | None = None  # True for prompts, False for chats (the template adds its own)
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    return_token_strs: bool = False
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template: str | None = None  # refused unless null (`check_chat_template`)
    tools: list[Any] | None = None


class DetokenizeRequest(_Strict):
    tokens: list[int]
    model: str | None = None


_NULL = msgspec.Raw(b"null")
_chat_decoder = msgspec.json.Decoder(ChatCompletionRequest)
_tokenize_decoder = msgspec.json.Decoder(TokenizeRequest)
_detokenize_decoder = msgspec.json.Decoder(DetokenizeRequest)


def _decode(decoder: msgspec.json.Decoder, body: bytes) -> Any:
    try:
        return decoder.decode(body)
    except msgspec.ValidationError as exc:
        raise ApiError(400, f"invalid request: {exc}") from exc
    except msgspec.DecodeError as exc:
        raise ApiError(400, f"invalid JSON: {exc}") from exc
    except RecursionError as exc:  # deeper than msgspec decodes (module docstring)
        raise ApiError(400, "invalid JSON: nested too deeply") from exc
    except UnicodeDecodeError as exc:  # a string that is not UTF-8 (outside strings, msgspec raises DecodeError)
        raise ApiError(400, f"invalid JSON: {exc}") from exc


def decode_chat(body: bytes) -> ChatCompletionRequest:
    req = _decode(_chat_decoder, body)
    spec = getattr(req.response_format, "json_schema", None)
    if spec is not None:  # its own bytes: the decoder's Raw is a view that would keep the whole body alive
        spec.schema = spec.schema.copy()
    return req


def decode_tokenize(body: bytes) -> TokenizeRequest:
    return _decode(_tokenize_decoder, body)


def decode_detokenize(body: bytes) -> DetokenizeRequest:
    return _decode(_detokenize_decoder, body)


MAX_STOP_TOKEN_IDS = 16  # distinct ones (`validate_chat`)
_DEFAULT_ONLY = {"n": 1, "frequency_penalty": 0.0, "presence_penalty": 0.0,
                 "repetition_penalty": 1.0, "spaces_between_special_tokens": True, "continue_final_message": False,
                 "echo": False}
MAX_LOGPROBS = 20  # runner.base.MAX_LOGPROBS: the OpenAI API's limit, the default of vLLM's `--max-logprobs`


class LogprobsLimit(ApiError):
    """More top logprobs than the limit: vLLM refuses it as its engine takes the request, so a streamed request
    gets it in the stream, with status 200 (`App.chat`)."""


def logprobs_param(req: ChatCompletionRequest, vocab: int) -> int:
    """The top logprobs a valid request asks for (`validate_chat`), -1 for none: vLLM's conversion and gate
    (`top_logprobs` null or absent: 0), then its engine's limit (`LogprobsLimit`)."""
    if not req.logprobs:
        return -1
    n = req.top_logprobs or 0
    if n == -1:
        n = vocab
    if n > MAX_LOGPROBS:
        raise validation_error(f"Requested sample logprobs of {n}, which is greater than max allowed: {MAX_LOGPROBS}",
                               "logprobs", n, LogprobsLimit)
    return n


UNTRUSTED_TEMPLATE = ("Chat template is passed with request, but --trust-request-chat-template is not set. Refused "
                      "request with untrusted chat template.")


def check_chat_template(req: ChatCompletionRequest | TokenizeRequest) -> None:
    """A chat template from the request, top-level or not null in its own `chat_template_kwargs`: vLLM's 400,
    its message alone (no parameter suffix), streamed or not. A null passes, and so does `tokenize` (the renderer drops
    it). The server's defaults (`model.chat_template_kwargs`) are not checked: the renderer drops a template there."""
    kw = req.chat_template_kwargs
    if req.chat_template is not None or (kw and kw.get("chat_template") is not None):
        raise ApiError(400, UNTRUSTED_TEMPLATE)


def check_messages(messages: list[Message], images: bool = False) -> None:
    """`images`: the server takes `image_url` parts."""
    if not messages:
        raise ApiError(400, "messages must not be empty", "messages")
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls and any(isinstance(c, CustomToolCall) for c in msg.tool_calls):
            raise validation_error("chat completions only support assistant tool_calls of type 'function'.",
                                   "tool_calls")
        if isinstance(msg.content, list):
            for i, part in enumerate(msg.content):
                if isinstance(part, ImageURLPart):
                    if not images:
                        raise ApiError(400, "this model does not take images (no vision tower, or images.enabled "
                                            "is false)", "messages")
                elif isinstance(part, _MEDIA_PARTS):
                    raise ApiError(400, f"'{part.__struct_config__.tag}' content is not supported", "messages")
                if getattr(part, "cache_control", None) is not None and i != len(msg.content) - 1:
                    raise ApiError(400, "'cache_control' is supported on a message or on its last content part",
                                   "messages")


_USAGE = 'Correct usage: `{"type": "function", "function": {"name": "my_function"}}`'


def check_tool_usage(req: ChatCompletionRequest) -> None:
    """The `check_tool_usage` of vllm-compatibility.md#tool-calls, in its order and words: an empty `tools`; then
    nothing to check for `tool_choice` "none" or null; `tools` missing; a string other than "auto" and "required"; a
    named tool's `function`, its `name`, and a name no tool has. `tool_choice` absent with tools becomes "auto" (the
    module docstring). A tool that is not an object with a `function` object matches no name here: `check_tools`
    refuses it."""
    tools, tc = req.tools, req.tool_choice
    if tools == []:  # the OpenAI API's rule
        raise validation_error("`tools` must not be an empty array. Either provide at least one tool or omit the field "
                               "entirely.", "tools")
    if tc is msgspec.UNSET:
        tc = req.tool_choice = "auto" if tools else msgspec.UNSET
    if tc is msgspec.UNSET or tc is None or tc == "none":
        return
    if tools is None:
        raise validation_error("When using `tool_choice`, `tools` must be set.", "tool_choice")
    if tc not in ("auto", "required") and not isinstance(tc, dict):
        raise validation_error(f"Invalid value for `tool_choice`: {tc}! Only named tools, \"none\", \"auto\" or "
                               "\"required\" are supported.", "tool_choice")
    if isinstance(tc, dict):
        function = tc.get("function")
        if not isinstance(function, dict):
            raise validation_error(f"Invalid value for `function`: `{function}` in `tool_choice`! {_USAGE}",
                                   "tool_choice.function")
        if "name" not in function:
            raise validation_error(f"Expected field `name` in `function` in `tool_choice`! {_USAGE}",
                                   "tool_choice.function.name")
        name = function["name"]
        if not isinstance(name, str) or not name:
            raise validation_error(f"Invalid `name` in `function`: `{name}` in `tool_choice`! {_USAGE}",
                                   "tool_choice.function.name")
        if not any(isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name") == name
                   for t in tools):
            raise validation_error("The tool specified in `tool_choice` does not match any of the specified `tools`",
                                   "tool_choice")


_EFFORTS = "'none', 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max'"
_SANITIZE = (re.compile(r" at 0x[0-9a-fA-F]+>"), re.compile(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?'),
             re.compile(r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+"),
             re.compile(r"(?:/[\w\-]+)+/[\w\-]+\.\w+"))


def sanitize(message: str) -> str:
    """An error message as vLLM's `sanitize_message` hands it to the client: memory addresses, tracebacks and file
    paths (a URL's path among them) cut."""
    message = _SANITIZE[0].sub(">", message)
    message = _SANITIZE[1].sub("", message)
    message = _SANITIZE[2].sub("<path>", message)
    return _SANITIZE[3].sub("<path>", message).strip()


def _error_input(value: Any) -> Any:
    """vLLM's `_summarize_error_input`: a string as it is, a container described, any other value as text,
    each cut at 200 characters."""
    if isinstance(value, str):
        return value[:200] + "...[truncated]" if len(value) > 200 else value
    if isinstance(value, list | dict):
        return f"<{type(value).__name__} of {len(value)} items>"
    text = str(value)
    return text[:200] + "...[truncated]" if len(text) > 200 else text


def check_reasoning_effort(value: Any) -> None:
    """`reasoning_effort` outside vLLM's `ReasoningEffort` (`output.reasoning.EFFORTS`; null passes): its 400,
    as its validation handler words pydantic's `literal_error` (measured with its pydantic 2.12 and 2.14 and FastAPI
    0.136), sanitized as it sanitizes it."""
    from ..output.reasoning import EFFORTS

    if value is None or (isinstance(value, str) and value in EFFORTS):
        return
    err = str({"type": "literal_error", "loc": "body.reasoning_effort", "msg": f"Input should be {_EFFORTS}",
               "input": _error_input(value), "ctx": {"expected": _EFFORTS}})
    if len(err) > 1000:
        err = err[:1000] + "...[truncated]"
    raise ApiError(400, sanitize(f"1 validation error:\n  {err}"), "body.reasoning_effort", "Bad Request")


_SALT_FORBIDDEN = frozenset("@/\\\x00")


def check_cache_salt(salt: object) -> None:
    """vLLM's `validate_cache_salt` and its two texts: a salt is a non-empty string of at most 128 characters, with no
    '@', '/', '\\' or NUL. It names a prefix-cache namespace and keys the encoder cache's entries."""
    if salt is None:
        return
    if not isinstance(salt, str) or not salt:
        raise validation_error("Parameter 'cache_salt' must be a non-empty string if provided.", "cache_salt")
    if len(salt) > 128 or not _SALT_FORBIDDEN.isdisjoint(salt):
        raise validation_error("Parameter 'cache_salt' must be at most 128 characters and must not contain '@', "
                               "'/', '\\\\', or NUL.", "cache_salt")


def validate_chat(req: ChatCompletionRequest, vocab: int, images: bool = False) -> None:
    """A 400 for what we refuse; `stop_token_ids` deduplicated in place. `vocab`: the model's vocabulary size
    (config.json's, the logits' width), the bound for token ids. `check_tool_usage` comes first, as a validator of the
    whole request before its fields (a message's decode error, msgspec's, comes before it), then the fields vLLM
    types (`check_reasoning_effort`)."""
    check_tool_usage(req)
    check_reasoning_effort(req.reasoning_effort)
    check_messages(req.messages, images)
    for field, default in _DEFAULT_ONLY.items():
        if getattr(req, field) != default:
            raise ApiError(400, f"'{field}' is only supported at its default ({default!r})", field)
    top = req.top_logprobs
    if isinstance(top, int):  # vLLM's checks (module docstring)
        if top < 0 and top != -1:
            raise validation_error("`top_logprobs` must be a positive value or -1.", "top_logprobs", top)
        if (top == -1 or top > 0) and not req.logprobs:
            raise validation_error("when using `top_logprobs`, `logprobs` must be set to true.", "top_logprobs")
    if req.logit_bias:
        raise ApiError(400, "'logit_bias' is not supported yet", "logit_bias")
    rf = req.response_format
    if isinstance(rf, JsonSchemaFormat):
        if rf.json_schema is None:
            raise validation_error("When response_format type is 'json_schema', the 'json_schema' field must be "
                                   "provided.", "response_format")
        if not rf.json_schema.schema or rf.json_schema.schema == _NULL:
            raise ApiError(400, "'response_format.json_schema.schema' must be provided", "response_format")
    if req.min_tokens:
        raise ApiError(400, "'min_tokens' is not supported yet", "min_tokens")
    if req.stream_options is not None:
        if not req.stream and req.stream_options != StreamOptions():
            raise validation_error("Stream options can only be defined when `stream=True`.", "stream_options")
        if req.stream_options.continuous_usage_stats:
            raise ApiError(400, "'continuous_usage_stats' is not supported yet", "stream_options")
    check_cache_salt(req.cache_salt)  # vLLM's last request check
    if req.tools:  # where the compatibility contract's field type checks run: after the request's checks
        from .tools import check_tools

        check_tools(req.tools, req.tool_choice)
    t, p = req.temperature, req.top_p  # vLLM's sampling checks, in its order, and their messages
    if t is not None and t < 0:
        raise validation_error(f"temperature must be non-negative, got {t}.", "temperature", t)
    if t is not None and t > 2:
        raise validation_error(f"temperature must be in [0, 2], got {t}.", "temperature", t)
    if p is not None and not 0 < p <= 1:  # any smaller than 1 / the vocabulary keeps the top token alone
        raise validation_error(f"top_p must be in (0, 1], got {p}.", "top_p", p)
    if req.top_k is not None and req.top_k < -1:
        raise validation_error(f"top_k must be 0 (disable), or at least 1, got {req.top_k}.")
    if req.min_p is not None and not 0 <= req.min_p <= 1:
        raise validation_error(f"min_p must be in [0, 1], got {req.min_p}.")
    v = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
    if v is not None and v < 1:
        raise validation_error(f"max_tokens must be at least 1, got {v}.", "max_tokens", v)
    if req.seed is not None and not -(1 << 63) <= req.seed < (1 << 63):  # vLLM's bounds (-1: unseeded)
        raise ApiError(400, "'seed' must fit in a signed 64-bit integer", "seed")
    if not -(1 << 63) <= req.priority < (1 << 63):
        raise ApiError(400, "'priority' must fit in a signed 64-bit integer", "priority")
    stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    if any(not s for s in stops):
        raise validation_error("stop cannot contain an empty string.")
    if len(stops) > 16:
        raise ApiError(400, "at most 16 stop strings", "stop")
    if req.stop_token_ids:
        ids = list(dict.fromkeys(req.stop_token_ids))
        # The engine's per-slot stop table widens to the most ids a request brought, for its life, and every step
        # compares each row's tokens with it: at 64 rows of 1 + 5 tokens, 16 columns cost about 11 µs a step more
        # than its first 4 (8,192: 7 ms)
        if len(ids) > MAX_STOP_TOKEN_IDS:
            raise ApiError(400, f"at most {MAX_STOP_TOKEN_IDS} distinct stop_token_ids", "stop_token_ids")
        bad = [t for t in ids if not 0 <= t < vocab]
        if bad:
            raise validation_error(f"token_id(s) {bad} in stop_token_ids contain out-of-vocab token ids. Vocabulary "
                                   f"size: {vocab}", "stop_token_ids", bad)
        req.stop_token_ids = ids
