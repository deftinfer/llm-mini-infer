# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/v1/serial_utils.py of vLLM v0.30.0 (https://github.com/vllm-project/vllm), and changed for
# llm-mini-infer.
"""Wire format and ZMQ `ipc://` sockets of the engine's messages: the frontends' and the launcher's to it, its outputs
to the frontends (architecture.md, "Frontend and engine messages").

A msgpack-encoded msgspec Struct, with numpy arrays of `inline_threshold` bytes or more sent as separate frames that are
read back with `np.frombuffer` (no copy on receipt); smaller arrays are inlined as raw msgpack extension bytes. Frames:
`[type byte, msgpack header, *array buffers]`.

Arrays handed to `encode` must not be modified afterwards: ZMQ may still be sending them from its I/O thread.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from typing import Any

import msgspec
import numpy as np
import zmq

from .. import leftovers
from .messages import MESSAGE_TYPES

_EXT_RAW = 1
_TYPES_BY_BYTE = {v: k for k, v in MESSAGE_TYPES.items()}


class Encoder:
    """Not thread-safe (it stashes the message's array buffers while encoding)."""

    def __init__(self, inline_threshold: int = 256):
        self.inline_threshold = inline_threshold
        self._enc = msgspec.msgpack.Encoder(enc_hook=self._enc_hook)
        self._bufs: list[Any] | None = None

    def encode(self, msg: msgspec.Struct) -> list[Any]:
        self._bufs = bufs = [MESSAGE_TYPES[type(msg)], b""]
        try:
            bufs[1] = self._enc.encode(msg)
        finally:
            self._bufs = None
        return bufs

    def _enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in "OV":
            data = obj.data if obj.flags.c_contiguous else obj.tobytes()
            if obj.nbytes < self.inline_threshold or self._bufs is None:
                return obj.dtype.str, obj.shape, msgspec.msgpack.Ext(_EXT_RAW, data)
            self._bufs.append(data)
            return obj.dtype.str, obj.shape, len(self._bufs) - 1
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"cannot encode {type(obj).__name__}")


class Decoder:
    """Arrays in decoded messages are views of the received frames (no copy). Not thread-safe."""

    def __init__(self) -> None:
        self._aux: Sequence[Any] = ()
        self._decs = {b: msgspec.msgpack.Decoder(t, dec_hook=self._dec_hook, ext_hook=self._ext_hook)
                      for b, t in _TYPES_BY_BYTE.items()}

    def decode(self, frames: Sequence[Any]) -> msgspec.Struct:
        bufs = [f.buffer if isinstance(f, zmq.Frame) else f for f in frames]
        kind = bytes(bufs[0])
        self._aux = bufs
        try:
            return self._decs[kind].decode(bufs[1])
        finally:
            self._aux = ()

    def _dec_hook(self, typ: type, obj: Any) -> Any:
        if typ is np.ndarray:
            dtype, shape, data = obj
            buf = self._aux[data] if isinstance(data, int) else data
            return np.frombuffer(buf, dtype=dtype).reshape(shape)
        raise TypeError(f"cannot decode {typ}")

    @staticmethod
    def _ext_hook(code: int, data: memoryview) -> Any:
        if code == _EXT_RAW:
            return data
        raise TypeError(f"unknown msgpack extension {code}")


def ipc_dir(base: str | None = None) -> str:
    """A short private directory for the `ipc://` socket files (sun_path is limited to about 104 bytes), in /tmp
    (`leftovers.tmp_root`) unless `base` names another."""
    import tempfile

    return tempfile.mkdtemp(prefix=leftovers.PREFIX, dir=base or leftovers.tmp_root())


def locked_dir() -> tuple[str, int | None]:
    """The launcher's socket directory: an `ipc_dir` with its lock file, locked (`leftovers`). Its path and the lock's
    descriptor: close it once the directory is removed. On a filesystem that refuses flock
    (`leftovers.refuses_flock`) the directory has no lock file and the descriptor is None, with a warning: no sweep
    removes a directory without one, while this server lives or after a kill."""
    path = ipc_dir()
    lock = os.path.join(path, leftovers.LOCK_FILE)
    try:
        return path, leftovers.locked_file(lock)
    except BaseException as exc:
        if leftovers.refuses_flock(exc):  # `locked_file` removed what it made: no lock file
            leftovers.warn_unlocked(lock, exc, "this server's socket directory has no lock file: if the server is "
                                    "killed, no start removes the directory, which stays until it is removed by hand")
            return path, None
        shutil.rmtree(path, ignore_errors=True)
        raise


def engine_input_addr(path: str) -> str:
    return f"ipc://{path}/engine.sock"


def frontend_output_addr(path: str, index: int) -> str:
    return f"ipc://{path}/frontend-{index}.sock"


def configure_socket(sock: zmq.Socket) -> None:
    # No high-water marks: a slow consumer queues instead of losing step outputs; LINGER 0 so shutdown never
    # blocks on undeliverable messages.
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    sock.setsockopt(zmq.LINGER, 0)
