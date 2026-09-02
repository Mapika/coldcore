"""The multi-cell campaign runner: artefacts, resume, isolation."""
import json
import os

import numpy as np
import pytest

from coldcore.backends import (cell_label, make_backend, parse_cell,
                               split_cells)
from coldcore.campaign import normalise, run_campaign, run_job, summarise
from coldcore.io import read_solution


@pytest.mark.parametrize("spec,expect", [
    ("K3(4,1)", ("hamming", [3] * 4, 1, 1)),
    ("K2(6,1)mu3", ("hamming", [2] * 6, 1, 3)),
    ("hamming:2x3x4:2", ("hamming", [2, 3, 4], 2, 1)),
    ("covering:5x5:1", ("hamming", [5, 5], 1, 1)),
    ("lee:5x5:1", ("lee", [5, 5], 1, 1)),
    ("torus_linf:9x9:1", ("torus_linf", [9, 9], 1, 1)),
    ("grid_linf:8x8:1mu2", ("grid_linf", [8, 8], 1, 2)),
])
def test_cell_specs_round_trip(spec, expect):
    got = parse_cell(spec)
    assert got == expect
    assert parse_cell(cell_label(*got)) == got


def test_split_cells_respects_parentheses():
    assert split_cells("K3(4,1), lee:5x5:1 ,K2(6,1)mu2") == [
        "K3(4,1)", "lee:5x5:1", "K2(6,1)mu2"]


def test_bad_specs_rejected():
    for bad in ("nonsense", "K3(4,1", "lee:5x5"):
        with pytest.raises(ValueError):
            parse_cell(bad)
    with pytest.raises(ValueError):
        make_backend("quantum", [3] * 3, 1)


def test_job_validation_rejects_typos_and_needs_a_cell():
    with pytest.raises(ValueError):
        normalise({"cell": "K3(4,1)", "polsh": "lahc"})
    with pytest.raises(ValueError):
        normalise({"budget_s": 10})
    j = normalise({"cell": "K3(4,1)"})
    assert j["backend"] == "cpu" and j["label"] == "K3_4_1_"


def test_run_job_writes_solution_log_and_record(tmp_path):
    rec = run_job({"cell": "K3(4,1)", "budget_s": 20, "backend": "cpu"},
                  outdir=str(tmp_path))
    assert rec["error"] is None
    assert rec["uncov"] == 0 and rec["M"] == 9
    code = read_solution(rec["solution_file"], n=4)
    assert len(code) == rec["M"]
    b = make_backend("cpu", [3] * 4, 1)
    b.recount(b.word_index(code.astype(np.int64)))
    assert b.uncovered() == 0                  # the artefact is a real cover
    assert os.path.exists(rec["log_file"])
    saved = json.load(open(tmp_path / "K3_4_1_.json"))
    assert saved["M"] == rec["M"]


def test_a_crashing_cell_does_not_take_the_campaign_down(tmp_path):
    """torus_linf needs 2R+1 <= axis: that cell must fail alone."""
    recs = run_campaign([{"cell": "torus_linf:4x4:2"}, {"cell": "K3(4,1)"}],
                        outdir=str(tmp_path), workers=1, budget_s=20)
    assert recs[0]["error"] and "2R+1" in recs[0]["error"]
    assert recs[1]["error"] is None and recs[1]["uncov"] == 0
    assert "ERROR" in summarise(recs)


def test_campaign_runs_cells_in_parallel_and_collects_records(tmp_path):
    cells = ["K3(4,1)", "K2(6,1)", "lee:5x5:1", "torus_linf:9x9:1"]
    seen = []
    recs = run_campaign([{"cell": c} for c in cells], outdir=str(tmp_path),
                        workers=4, budget_s=20, backend="cpu",
                        on_done=seen.append)
    assert [r["label"] for r in recs] == [
        normalise({"cell": c})["label"] for c in cells]
    assert len(seen) == len(cells)
    assert all(r["error"] is None and r["uncov"] == 0 for r in recs)
    coll = json.load(open(tmp_path / "campaign.json"))
    assert len(coll["records"]) == len(cells)


def test_duplicate_labels_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_campaign([{"cell": "K3(4,1)"}, {"cell": "K3(4,1)"}],
                     outdir=str(tmp_path), workers=1)


def test_rerunning_resumes_from_the_checkpoint(tmp_path):
    first = run_job({"cell": "K2(6,1)", "budget_s": 20}, outdir=str(tmp_path))
    again = run_job({"cell": "K2(6,1)", "budget_s": 20}, outdir=str(tmp_path))
    assert "resumed from" in open(again["log_file"]).read()
    assert again["M"] <= first["M"]            # never worse than the artefact
    fresh = run_job({"cell": "K2(6,1)", "budget_s": 20}, outdir=str(tmp_path),
                    resume=False)
    assert "resumed from" not in open(fresh["log_file"]).read().split(
        "=== K2_6_1_")[-1]
