"""The MTP scheduler suite's workloads and driver (tests/sched/test_mtp_sched.py)."""

from __future__ import annotations

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.runner.fake import fake_generate
from llm_mini_infer.sched.plan import STOP_W, reserve_positions

from .fuzzing import event, nan_report, nan_write, st
from .harness import EOS, Spec, prompt_of
from .sched_cases import VOCAB, stem


def output_of(spec: Spec, n: int) -> list[int]:
    eos = {spec.prompt_len + spec.eos_after: EOS} if spec.eos_after is not None else {}
    return fake_generate(prompt_of(spec, VOCAB), spec.seed, n, VOCAB, (EOS,), eos)


@st.composite
def mtp_workloads(draw):
    """MTP's hazards in small pools: finishes inside accepted runs (EOS, stop tokens, more stop tokens than the runner
    knows, max_tokens), preemption and aborts with decode rows in flight, follow-up turns that resume from donations,
    decode checkpoints every few positions, hits on page boundaries (a drafter with KV copies the page below), NaNs
    sampled and written by prefill chunks (the cache suites' NaN dimension)."""
    k = draw(st.sampled_from([1, 2, 3, 3, 5]))
    page_size = draw(st.sampled_from([1, 2, 4, 8]))
    grid = draw(st.sampled_from([8, 16]))
    max_model_len = 160
    stems = [stem(i, draw(st.integers(8, 90))) for i in range(draw(st.integers(1, 3)))]
    n = draw(st.integers(1, 8))
    seeds = draw(st.lists(st.integers(1 << 32, (1 << 62) - 1), min_size=n, max_size=n, unique=True))
    specs: list[Spec] = []
    for rid in range(n):
        kind = draw(st.sampled_from(["stem", "turn", "turn", "random"]))
        if kind == "turn" and specs:
            prev = draw(st.sampled_from(specs))
            out = output_of(prev, prev.max_tokens)
            keep = draw(st.integers(0, len(out)))
            prompt = list(prompt_of(prev, VOCAB)) + out[:keep]
        elif kind in ("stem", "turn"):
            s = draw(st.sampled_from(stems))
            prompt = s[:draw(st.integers(1, len(s)))]
        else:
            prompt = []
        prompt = prompt + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=12))
        prompt = [t if t != EOS else 0 for t in prompt][:120]
        many = draw(st.booleans()) and draw(st.booleans())  # more stop tokens than the runner knows
        n_stops = (STOP_W, STOP_W) if many else (0, 2)
        stops = draw(st.lists(st.integers(0, VOCAB - 2), min_size=n_stops[0], max_size=n_stops[1]))
        if many:  # the last one (unknown to the runner) from the request's own predicted output, so it happens
            probe = Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=30, seed=seeds[rid])
            stops.append(draw(st.sampled_from(output_of(probe, 30))))
        eos = draw(st.one_of(st.none(), st.integers(0, 25)))
        specs.append(Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, max_tokens=draw(st.integers(1, 30)),
                          seed=seeds[rid], arrive=draw(st.integers(0, 30)), eos_after=eos,
                          ignore_eos=draw(st.booleans()) if eos is not None else False,
                          abort_at=draw(st.one_of(st.none(), st.integers(0, 50))), priority=draw(st.integers(0, 2)),
                          stop_ids=stops, hints=draw(st.lists(st.integers(1, len(prompt)), max_size=1)),
                          nonfinite_after=draw(st.sampled_from([None] * 10 + [0, 2, 5])),
                          **nan_write(draw, len(prompt))))
    bound = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True)
    return dict(specs=specs, page_size=page_size, grid=grid, max_model_len=max_model_len, mtp_k=k,
                report_unsampled=nan_report(draw),
                accept_p=draw(st.sampled_from([0.0, 0.5, 0.8, 0.95, 1.0])), max_num_seqs=draw(st.integers(1, 5)),
                max_tokens=draw(st.integers(4 * (1 + k), 96)), num_pages=bound + 1 + draw(st.integers(0, 60)),
                checkpoint_slots=draw(st.integers(1, 6)), policy=draw(st.sampled_from(["fcfs", "priority"])),
                chunk_cap=draw(st.sampled_from([0, 0, 9])), decode_checkpoint_interval=draw(st.sampled_from([8, 16])),
                hybrid=draw(st.sampled_from([True, True, False])),
                row_tokens=draw(st.sampled_from([0, 0, 1, 40, 160, 640])))


def with_counts(draw, w: dict) -> dict:
    """A workload with draft counts per step (`sched.draft_counts`): the policy over every count below k, or a schedule
    of counts cycled over the plans with decode rows (lowering and raising, `Scheduler.held`)."""
    k = w["mtp_k"]
    if k > 1 and draw(st.booleans()):
        w["adaptive"] = True
    else:
        w["count_schedule"] = draw(st.lists(st.integers(1, k), min_size=1, max_size=6))
    return w


@st.composite
def mtp_count_workloads(draw):
    """`mtp_workloads` with a draft count per step: every hazard of theirs while the verified and drafted counts
    change."""
    return with_counts(draw, draw(mtp_workloads()))


def run_mtp_workload(w: dict) -> None:
    from .workloads import run_cache_workload

    h = run_cache_workload(w)
    if h.sched.num_preemptions and h.runner.counts["folds"]:
        event("MTP: preempted with a fold")
    if any(len(r.tokens) and r.finish == FinishReason.LENGTH for r in h.results.values()):
        event("MTP: length finish")
    if w["accept_p"] == 1.0:
        event("every draft accepted")
    if any(len(s.stop_ids) > STOP_W for s in w["specs"]):
        event("more stop tokens than the runner knows")
