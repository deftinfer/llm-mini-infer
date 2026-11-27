from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
import textwrap
import traceback

_CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"  # traceback's own words
_CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n\n"
LOG_QUEUE_LINES = 10_000  # the lines a queued process's writer may fall behind by (`Queued`)


class WithoutMessages(logging.Filter):
    """A logged exception (`log.exception`, uvicorn's "Exception in ASGI application") by its types and frames alone
    (`traceback_without_messages`): in the processes that handle requests, an exception's message can quote one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = traceback_without_messages(record.exc_info[1]).rstrip()
            record.exc_info = None
        return True


class Queued(logging.handlers.QueueHandler):
    """A process's log lines handed to a writer thread (`setup_logging(queued=True)`), so a stderr that stops reading
    blocks that thread and never the caller: a frontend's event loop. At most `LOG_QUEUE_LINES` wait; past that a line
    is dropped and counted, and the next line that finds room is preceded by one WARNING with the count. The lines
    keep their order, and each is written whole by one write, as a direct handler writes it."""

    def __init__(self, q: queue.Queue):
        super().__init__(q)
        self.dropped = 0

    def enqueue(self, record: logging.LogRecord) -> None:  # under the handler's lock (`Handler.handle`)
        if self.dropped:
            note = logging.LogRecord(__name__, logging.WARNING, __file__, 0,
                                     "%d log lines were dropped: stderr took no more and %d were waiting",
                                     (self.dropped, LOG_QUEUE_LINES), None)
            try:
                self.queue.put_nowait(self.prepare(note))
            except queue.Full:
                self.dropped += 1
                return
            self.dropped = 0
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


class _Writer(logging.handlers.QueueListener):
    def enqueue_sentinel(self) -> None:
        self.queue.put(self._sentinel)  # waits for room: at a stop, every line queued before it is written

    def stop(self) -> None:
        if self._thread is not None:  # once: the process's exit stops it again
            super().stop()


def setup_logging(level: str, process: str, without_messages: bool = False,
                  queued: bool = False) -> logging.handlers.QueueListener | None:
    """One format for every process: `INFO 10-03 14:00:00.123 [engine] message`. `without_messages`: logged
    exceptions without their messages (`WithoutMessages`), for the processes that handle requests' contents. `queued`
    (the frontends): the lines go through `Queued` to a writer thread, stopped at the process's exit once it has written
    every line (the writer, returned); the other processes write directly (the launcher exits by `os._exit` on a
    signal and the pool's workers are killed at a stop: a queue would lose their last lines)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(f"%(levelname)s %(asctime)s.%(msecs)03d [{process}] %(message)s",
                                           datefmt="%m-%d %H:%M:%S"))
    front: logging.Handler = handler
    writer = None
    if queued:
        q: queue.Queue = queue.Queue(LOG_QUEUE_LINES)
        front = Queued(q)
        writer = _Writer(q, handler)
        writer.start()
        atexit.register(writer.stop)
    if without_messages:
        front.addFilter(WithoutMessages())
    root = logging.getLogger()
    root.handlers[:] = [front]
    root.setLevel(level.upper())
    # huggingface_hub's HTTP client (httpx2) logs every request of a download at INFO (llm_mini_infer.hub logs the
    # download)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx2", "httpcore2"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
    return writer


def traceback_without_messages(exc: BaseException) -> str:
    """`exc`'s traceback as Python prints it (after those of the exceptions it was raised from or while handling; an
    exception group's members, numbered, after the group's own), but each exception named by its type alone: no message
    and no notes, which can quote what a request carried (a prompt, a schema, an output)."""
    chain: list[BaseException] = []
    e: BaseException | None = exc
    while e is not None and not any(e is seen for seen in chain):
        chain.append(e)
        e = e.__cause__ if e.__cause__ is not None else None if e.__suppress_context__ else e.__context__
    out = []
    for i, e in enumerate(reversed(chain)):
        if i:
            out.append(_CAUSE if e.__cause__ is not None else _CONTEXT)
        if e.__traceback__ is not None:
            out.append("Traceback (most recent call last):\n")
            out.extend(traceback.format_tb(e.__traceback__))
        t = type(e)
        out.append((t.__qualname__ if t.__module__ in ("builtins", "__main__") else f"{t.__module__}.{t.__qualname__}")
                   + "\n")
        if isinstance(e, BaseExceptionGroup):
            for j, member in enumerate(e.exceptions, 1):
                out.append(f"+---------------- {j} ----------------\n")
                out.append(textwrap.indent(traceback_without_messages(member), "| "))
            out.append("+------------------------------------\n")
    return "".join(out)
