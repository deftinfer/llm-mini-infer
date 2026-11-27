"""Gemma 4's image tests: pixels, the engine with images, and transformers' references
(tests/models/test_gemma4_images.py)."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import blake3
import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest, ImageRef
from llm_mini_infer.runner.base import PoolConfig
from llm_mini_infer.sched.scheduler import Scheduler
from llm_mini_infer.vision import arena as px_arena
from llm_mini_infer.vision.family import span_key, transposed
from llm_mini_infer.vision.probe import probe

from .tiny_gemma import MM_IMAGE, hf_mm_model

pytest.importorskip("transformers")

EOS = 511


def png(w: int, h: int, seed: int) -> bytes:
    from PIL import Image

    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[: h // 3] //= 3
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def text(n: int, seed: int) -> list[int]:
    return [(i * 37 + 11 + seed * 101) % 490 + 3 for i in range(n)]  # below the image tokens (500-503)


class Pixels:
    """A pixel arena (`vision.arena`) at `path` for the model in `model_dir` (its image family: Gemma 4's or Qwen's),
    and prompts whose images go into it."""

    def __init__(self, model_dir, path: str, size: int = 1 << 22):
        from llm_mini_infer.vision import family as vf

        self.fam = vf.load(model_dir)
        px_arena.remove(path)
        px_arena.create(path, size)
        self.arena = px_arena.map_writable(path)
        self.next_px = 0

    def prompt(self, parts: list) -> tuple[list[int], list[ImageRef], list]:
        """parts: token lists and PNG images -> (ids with each image expanded to <boi>, its soft tokens, <eoi>; the
        images' refs with their pixels in the arena; the PIL images for transformers)."""
        from PIL import Image

        ids: list[int] = []
        refs: list[ImageRef] = []
        pils = []
        for part in parts:
            if isinstance(part, bytes):
                header = probe(part)
                size = self.fam.size(header)
                px = self.fam.pixels(part, header, size)
                if tuple(px.shape[1:]) != (size.height, size.width):  # Pillow turned the image: the decode decides
                    size = transposed(size)
                px = px.numpy().reshape(-1)
                off = self.next_px
                self.arena.buf[off:off + len(px)] = px.tobytes()
                self.next_px += -(-len(px) // 4096) * 4096
                h = blake3.blake3(part)
                h.update(struct.pack("<II", size.height, size.width))
                ids.append(self.fam.start_id)  # the tiny Gemma's MM_BOI, MM_IMAGE and MM_EOI (or a Qwen's)
                refs.append(ImageRef(key=span_key(h.digest()), offset=len(ids), tokens=size.tokens, grid_t=1,
                                     grid_h=size.grid[1], grid_w=size.grid[2], height=size.height, width=size.width,
                                     pixels=off))
                ids += [self.fam.pad_id] * size.tokens + [self.fam.end_id]
                pils.append(Image.open(io.BytesIO(part)))
            else:
                ids += list(part)
        return ids, refs, pils


class Engine:
    """The CPU runner with images, the scheduler (window mode, bidirectional spans) and the engine loop; images go
    through the family's pool path into a pixel arena; the logits hook keyed by (rid, position) (`hook`: a runner with a
    logits hook runs every step eagerly, so graph tests go without)."""

    def __init__(self, path, tmp, *, budget: int = 128, caching: bool = True, row_tokens: int = 0,
                 num_pages: int = 256, max_num_seqs: int = 2, enc_rows: int = 256, enc_budget: int = 160,
                 device: str = "cpu", graphs: bool = True, wps: int = 4, mtp_k: int = 0, hook: bool = True,
                 **runner_kw):
        from llm_mini_infer.runner.cuda.gemma import GemmaRunner
        from llm_mini_infer.runner.cuda.images import VisionOptions

        self.arena_path = str(tmp / f"arena-{budget}-{caching}-{row_tokens}-{num_pages}-{device}-{graphs}-{mtp_k}")
        self.pixels = Pixels(path, self.arena_path)
        self.fam = self.pixels.fam
        cuda = device == "cuda"
        runner_kw.setdefault("dtype", torch.bfloat16 if cuda else torch.float32)
        self.runner = GemmaRunner(path, device=device, max_num_seqs=max_num_seqs, max_num_batched_tokens=budget,
                                  max_model_len=512, check=True, memory_gb=2.0 if cuda else None, graphs=graphs,
                                  vision=VisionOptions(encoder_rows=enc_rows, encoder_budget=enc_budget,
                                                       arena=self.arena_path), mtp_k=mtp_k, **runner_kw)
        full = self.runner.full_page_size(wps)
        self.runner.configure(PoolConfig(page_size=full, num_pages=num_pages, num_state_slots=max_num_seqs + 1,
                                         max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=512,
                                         eos_token_ids=(EOS,), mtp_k=mtp_k, window_page_size=wps))
        caps = self.runner.capabilities()
        assert caps.bidirectional_images and caps.images
        self.sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=512,
                               page_size=full, num_pages=num_pages, num_state_slots=max_num_seqs + 1,
                               eos_token_ids=(EOS,), hybrid=False, prefix_caching=caching, checkpoint_grid=wps,
                               decode_checkpoint_interval=16, window=caps.window, window_page_size=wps,
                               encoder_rows=caps.encoder_rows, encoder_budget=caps.encoder_budget,
                               positions=self.fam.positions, mixed_step_row_tokens=row_tokens,
                               bidirectional_images=caps.bidirectional_images, mtp_k=mtp_k, draft_kv=caps.draft_kv)
        self.core = EngineCore(self.sched, self.runner)
        self.logits: dict[tuple[int, int], torch.Tensor] = {}
        self._slot_rid: dict[int, dict[int, int]] = {}
        orig = self.runner.launch_forward

        def record(plan, _internal=False, _capture=None):
            if not _internal:
                self._slot_rid[plan.step_id] = {int(s): int(self.sched.fe_rid[s]) for s in plan.sched_slot}
                self._rows = plan.sched_slot.tolist()
            return orig(plan, _internal=_internal, _capture=_capture)

        self.runner.launch_forward = record

        def on_logits(c) -> None:
            slots = list(self._rows)
            for i, (row, pos) in enumerate(zip(c.rows.tolist(), c.positions.tolist())):
                rid = self._slot_rid[c.step_id][slots[row]] if c.step_id in self._slot_rid else -1
                self.logits[(rid, pos)] = c.logits[i].detach().float().cpu().clone()

        if hook:
            self.runner.logits_hook = on_logits
        self.outs: dict[int, list[int]] = {}
        self.cached: dict[int, int] = {}

    def prompt(self, parts: list) -> tuple[list[int], list[ImageRef], list]:
        return self.pixels.prompt(parts)

    def add(self, rid: int, ids: list[int], refs: list[ImageRef], n: int, hints: list[int] = ()) -> None:
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(ids, np.int32), max_tokens=n,
                                    temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=rid, stop_token_ids=[],
                                    images=refs, cache_hints=list(hints)))
        self.outs[rid] = []

    def run(self, max_turns: int = 400, on_turn=None) -> None:
        turns = 0
        while self.core.busy and turns < max_turns:
            rep = self.core.step()
            turns += 1
            if on_turn is not None:
                on_turn(turns)
            if rep is None:
                continue
            for o in rep.outputs.values():
                for rid, c in zip(o.first_rids.tolist(), o.first_cached.tolist()):
                    self.cached[rid] = c
                off = 0
                for rid, k in zip(o.rids.tolist(), o.n_new.tolist()):
                    self.outs[rid].extend(o.tokens[off:off + k].tolist())
                    off += k
        assert not self.core.busy
        self.sched.check_accounting()


def greedy_mm(model, ids: list[int], pils: list, n: int) -> list[int]:
    """transformers' greedy continuation, one full forward per token (the image spans bidirectional each time)."""
    from .tiny_gemma import hf_image_processor

    px = hf_image_processor(Path(model.name_or_path))(images=pils, return_tensors="pt") if pils else None
    seq, out = list(ids), []
    with torch.no_grad():
        for _ in range(n):
            t = torch.tensor([seq])
            kw = {}
            if px is not None:
                kw = dict(pixel_values=px["pixel_values"], image_position_ids=px["image_position_ids"],
                          mm_token_type_ids=(t == MM_IMAGE).long())
            nxt = int(model(input_ids=t, **kw).logits[0, -1].argmax())
            out.append(nxt)
            if nxt == EOS:
                break
            seq.append(nxt)
    return out


def hf_mm_logits_causal(model_dir, ids, pils):
    """HF's forward with images but without `mm_token_type_ids`: every span causal."""
    from .tiny_gemma import hf_image_processor

    model = hf_mm_model(model_dir)
    out = hf_image_processor(model_dir)(images=pils, return_tensors="pt")
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), pixel_values=out["pixel_values"],
                     image_position_ids=out["image_position_ids"]).logits[0].float()
