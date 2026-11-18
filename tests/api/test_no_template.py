"""Regression for #22: a model with no chat template (or a template list without a default) must answer a
JSON 400 that names the missing template -- from `render_conversation`, a chat request and /tokenize, with or
without `chat_template_kwargs` -- not an unhandled TemplateError (a 500) or a startup traceback."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models

from llm_mini_infer import config
from llm_mini_infer.api import app as appmod
from llm_mini_infer.api.protocol import ApiError


def _model_dir(tmp_path: Path, chat_template=None) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    tok = Tokenizer(models.WordLevel(vocab={"hi": 0, "there": 1, "<unk>": 2}, unk_token="<unk>"))
    tok.save(str(d / "tokenizer.json"))
    tcfg = {} if chat_template is None else {"chat_template": chat_template}
    (d / "tokenizer_config.json").write_text(json.dumps(tcfg))
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "vocab_size": 100, "hidden_size": 8,
        "num_attention_heads": 2, "num_hidden_layers": 1,
    }))
    return d


class _StubClient:
    """EngineClient without ZMQ: ipc:// sockets are unavailable off Linux."""

    def __init__(self, *args, **kwargs):
        self.reqs = {}
        self.queued_tokens = 0

    async def close(self):
        pass


def _app(tmp_path: Path, monkeypatch, chat_template=None):
    d = _model_dir(tmp_path, chat_template)
    cfg = config.derive(config.load(None, {"model.path": str(d)}))
    monkeypatch.setattr(appmod, "EngineClient", _StubClient)
    return appmod.App(cfg, 0, str(tmp_path / "ipc"))


class _Receive:
    def __init__(self, body: bytes):
        self.body, self.read = body, False
        self.block = asyncio.Event()

    async def __call__(self):
        if not self.read:
            self.read = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.block.wait()
        return {"type": "http.disconnect"}


def _call(app, path: str, body: dict) -> tuple[int, dict]:
    out = []

    async def send(msg):
        out.append(msg)

    receive = _Receive(json.dumps(body).encode())
    asyncio.run(app.http({"path": path, "method": "POST", "headers": []}, receive, send))
    start = next(m for m in out if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in out if m["type"] == "http.response.body")
    return start["status"], json.loads(payload)


CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def test_render_conversation_raises_api_error_without_template(tmp_path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch)
    with pytest.raises(ApiError) as e:
        app.render_conversation([{"role": "user", "content": "hi"}], True, None)
    assert e.value.status == 400
    assert "no chat template" in e.value.message


@pytest.mark.parametrize("kwargs", [None, {"enable_thinking": False}])
def test_chat_request_400_without_template(tmp_path, monkeypatch, kwargs) -> None:
    app = _app(tmp_path, monkeypatch)
    body = dict(CHAT)
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    status, payload = _call(app, "/v1/chat/completions", body)
    assert status == 400
    assert "no chat template" in payload["error"]["message"]


@pytest.mark.parametrize("kwargs", [None, {"enable_thinking": False}])
def test_tokenize_request_400_without_template(tmp_path, monkeypatch, kwargs) -> None:
    app = _app(tmp_path, monkeypatch)
    body = dict(CHAT)
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    status, payload = _call(app, "/tokenize", body)
    assert status == 400
    assert "no chat template" in payload["error"]["message"]


def test_template_list_without_default_builds_and_answers_400(tmp_path, monkeypatch) -> None:
    templates = [{"name": "tool_use", "template": "{% for m in messages %}{{ m.content }}{% endfor %}"}]
    app = _app(tmp_path, monkeypatch, chat_template=templates)
    status, payload = _call(app, "/v1/chat/completions", CHAT)
    assert status == 400
    assert "no default" in payload["error"]["message"]
