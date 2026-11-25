"""A chat request's tools (architecture.md, "Tool calls"): their checks, the form the chat template gets, and the
request's choice.

- **Checks** (`check_tools`): each tool is an object of type `function` (the default) with a `function` object that has
  a string `name`, a `description` string or null, `parameters` an object or null, and `strict` and `defer_loading`
  booleans or null (at both levels); anything else is a 400, param `tools`. A named `tool_choice` has the type
  `function`. The checks of `tool_choice` itself are `api.protocol.check_tool_usage`'s.
- **The template's form** (`template_tool`): the normalized dump vLLM renders (its `ChatCompletionToolsParam`),
  whatever key order or extra keys the client sent: `{"type": "function", "function": {"name", "description",
  "parameters"[, "strict"][, "defer_loading"]}[, "defer_loading"]}`; an absent `description` or `parameters` is null,
  `strict` and `defer_loading` appear when set, and a top-level `defer_loading` is copied into `function` when that has
  none. Qwen's template prints each tool with `tojson`, so the form is the prompt's bytes.
- **The choice** (`ToolRequest`): "none", "auto", "required", "named" (`name`), or None for an explicit null: the tools
  are rendered and parsed, and no call is kept (the OpenAI API's `tool_choice`; vLLM's `keeps_tool_calls`).
  `parallel_tool_calls` false keeps the first call only.
"""

from __future__ import annotations

from typing import Any

import msgspec

from .protocol import ApiError


def _bad(i: int, what: str) -> ApiError:
    return ApiError(400, f"tools[{i}]: {what}", "tools")


def check_tools(tools: list[Any], tool_choice: Any = None) -> None:
    """A 400 for a malformed tool (module docstring), and for a named `tool_choice` of another type."""
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise _bad(i, "a tool must be an object")
        kind = tool.get("type", "function")
        if kind != "function":
            raise _bad(i, f"only tools of type 'function' are supported, not {kind!r}")
        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise _bad(i, "'function' must be an object")
        if not isinstance(fn.get("name"), str):
            raise _bad(i, "'function.name' must be a string")
        if not isinstance(fn.get("description"), str | None):
            raise _bad(i, "'function.description' must be a string or null")
        if not isinstance(fn.get("parameters"), dict | None):
            raise _bad(i, "'function.parameters' must be an object or null")
        for where, obj, key in (("function.", fn, "strict"), ("function.", fn, "defer_loading"),
                                ("", tool, "defer_loading")):
            if not isinstance(obj.get(key), bool | None):
                raise _bad(i, f"'{where}{key}' must be a boolean or null")
    if isinstance(tool_choice, dict) and tool_choice.get("type", "function") != "function":
        raise ApiError(400, "a named 'tool_choice' must have the type 'function'", "tool_choice")


def template_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """The form a checked tool reaches the chat template in (module docstring)."""
    fn = tool["function"]
    out_fn: dict[str, Any] = {"name": fn["name"], "description": fn.get("description"),
                              "parameters": fn.get("parameters")}
    if fn.get("strict") is not None:
        out_fn["strict"] = fn["strict"]
    top = tool.get("defer_loading")
    inner = fn.get("defer_loading")
    if inner is None:
        inner = top
    if inner is not None:
        out_fn["defer_loading"] = inner
    out: dict[str, Any] = {"type": "function", "function": out_fn}
    if top is not None:
        out["defer_loading"] = top
    return out


class ToolRequest:
    """A request's checked tools: `tools`, their template form; `choice` and `name` (module docstring); `parallel`;
    `schemas`, each tool's `parameters` by name (the first of two tools with one name, as the parse types by it)."""

    __slots__ = ("choice", "name", "parallel", "schemas", "tools")

    def __init__(self, tools: list[dict[str, Any]], tool_choice: Any, parallel_tool_calls: bool | None):
        self.tools = [template_tool(t) for t in tools]
        self.schemas: dict[str, dict | None] = {}
        for t in self.tools:
            self.schemas.setdefault(t["function"]["name"], t["function"]["parameters"])
        self.name: str | None = None
        if isinstance(tool_choice, dict):
            self.choice: str | None = "named"
            self.name = tool_choice["function"]["name"]
        elif tool_choice is msgspec.UNSET:
            self.choice = "auto"
        else:
            self.choice = tool_choice  # "none", "auto", "required", or None (null)
        self.parallel = parallel_tool_calls is not False

    @property
    def forced(self) -> bool:
        """`required` or a named tool: the content is "" rather than null."""
        return self.choice in ("required", "named")


def tool_request(tools: list[Any] | None, tool_choice: Any, parallel_tool_calls: bool | None) -> ToolRequest | None:
    """The request's checked tools; None without any (an empty list: `/tokenize`'s, which takes one)."""
    if not tools:
        return None
    return ToolRequest(tools, tool_choice, parallel_tool_calls)
