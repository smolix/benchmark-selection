"""Tests for the job definitions of facility location (2026-10-19) and of
the split-level worker processes (2026-10-20) in run_all.sh and
run_tabimpute.sh, and for the driver's scheduler.

Facility location (facloc) is the paper's headline method: the
selection-order jobs draw it (and greedy trace) next to entropy,
analyze_objectives declares its facility-location figures, and
run_tabimpute.sh's default job runs it (eval_tabimpute's default set)
while the extra job keeps hub, pca, trace.  The slow jobs run their
splits in WORKERS processes (--workers; estimator_<ds>: --jobs) and count
that many cores, the imputer shards at most their 10 splits; the short
ones keep one process.  Runs the drivers with --list and --dry-run
(nothing executed) on a data directory of empty placeholder matrices: job
outputs and commands only.  The scheduler test sources run_all.sh and
runs four `sleep` jobs through bs_run_selected (EASY backfilling, about
2 s).  Needs bash >= 4.4 (the drivers' own requirement); skipped
otherwise (macOS ships bash 3.2).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

CODE = pathlib.Path(__file__).resolve().parent
DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
EXTRA = ["olv2sub", "olv1items", "epoch", "helm_lite", "helm_mmlu",
         "helm_classic", "mteb2026"]


def _bash_ok():
    bash = shutil.which("bash")
    if not bash:
        return False
    out = subprocess.run([bash, "-c", 'echo "${BASH_VERSINFO[0]} '
                          '${BASH_VERSINFO[1]}"'], capture_output=True,
                         text=True).stdout.split()
    return len(out) == 2 and (int(out[0]), int(out[1])) >= (4, 4)


pytestmark = pytest.mark.skipif(not _bash_ok(), reason="needs bash >= 4.4")


def _run(tmp_path, *args, script="run_all.sh", **env_extra):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for ds in DATASETS + EXTRA + ["merged_v1"]:
        (data / f"{ds}.matrix.npz").touch()
    env = {k: v for k, v in os.environ.items()
           if k not in ("DATASETS", "DATASETS_EXTRA", "RUN_ONLY", "WORKERS")}
    env.update(BENCHSELECT_DATA_DIR=str(data), MAX_JOBS="64",
               BENCHSELECT_EXPERIMENT_ROOT=str(tmp_path / "root"))
    env.update(env_extra)
    res = subprocess.run(["bash", str(CODE / script), *args],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-2000:]
    return res.stdout


def _job_line(listing, name):
    lines = [ln for ln in listing.splitlines() if ln.split()[:1] == [name]]
    assert len(lines) == 1, name
    return lines[0]


class TestFaclocJobs:
    def test_selection_order_outputs(self, tmp_path):
        listing = _run(tmp_path, "--list")
        line = _job_line(listing, "selection_order")
        for ds in ("mmlu", "mteb", "merged"):
            for fig in (f"selection_order_{ds}.pdf",
                        f"selection_order_facloc_{ds}.pdf",
                        f"selection_order_trace_{ds}.pdf"):
                assert f"figures/{fig}" in line.split(), fig
        assert "selection_order_benchpress" not in line
        line = _job_line(listing, "selection_order_extra")
        for ds in EXTRA:
            assert f"figures/selection_order_facloc_{ds}.pdf" in line.split()

    def test_analyze_objectives_outputs(self, tmp_path):
        listing = _run(tmp_path, "--list")
        line = _job_line(listing, "analyze_objectives").split()
        for ds in DATASETS:
            assert f"figures/analyze_objectives_{ds}.pdf" in line
            assert f"figures/analyze_objectives_facloc_{ds}.pdf" in line
        line = _job_line(listing, "analyze_objectives_fulloracle").split()
        assert "figures/analyze_objectives_facloc_merged_fulloracle.pdf" \
            in line

    def test_selection_order_command(self, tmp_path):
        plan = _run(tmp_path, "--dry-run", "--repeats", "1")
        cmds = [ln for ln in plan.splitlines()
                if "plot_selection_order.py" in ln]
        assert len(cmds) == 2
        assert all("--repeats 1 --methods entropy facloc trace" in c
                   for c in cmds)
        assert not (tmp_path / "root").exists()

    def test_tabimpute_jobs_keep_their_method_sets(self, tmp_path):
        """run_tabimpute.sh: facloc joins the default job (eval_tabimpute's
        default set, no BENCHSELECT_METHODS), the extra job keeps hub, pca,
        trace."""
        plan = _run(tmp_path, "--dry-run", script="run_tabimpute.sh",
                    BENCHSELECT_DATASETS="mmlu,mteb",
                    BENCHSELECT_TABIMPUTE_DEVICE="cpu")
        # the plan quotes each argument with printf %q (hub\,pca\,trace)
        cmds = [ln.replace("\\", "") for ln in plan.splitlines()
                if "eval_tabimpute.py" in ln and not ln.startswith("#")]
        assert len(cmds) == 4
        for ds in ("mmlu", "mteb"):
            mine = [c for c in cmds if f"BENCHSELECT_DATASETS={ds} " in c]
            assert len(mine) == 2, ds
            assert sum("BENCHSELECT_METHODS" not in c for c in mine) == 1
            assert sum("BENCHSELECT_METHODS=hub,pca,trace " in c
                       for c in mine) == 1
        assert not (tmp_path / "root").exists()


# The jobs that run their splits in WORKERS processes (--workers; the
# estimator ablation: --jobs), apart from the imputer shards
PAR_BATTERY = ["epoch", "merged", "mteb2026", "mteb", "helm_classic"]
PAR_JOBS = (
    ["benchpress", "reviewer_checks", "analyze_objectives",
     "analyze_objectives_fulloracle", "baselines_merged_v1",
     "baselines_benchpress"]
    + [f"{job}_{ds}" for job in ("greedy_cv", "entropy_vs_mi",
                                 "mi_selection", "baselines")
       for ds in PAR_BATTERY]
    + [f"logit_{ds}" for ds in ("merged", "benchpress", "mteb2026", "mteb",
                                "helm_classic")]
    + [f"{job}_{ds}" for job in ("stability", "estimator")
       for ds in ("merged", "benchpress", "mteb")]
    + ["benchfam_merged", "benchfam_merged_sub", "benchfam_mteb",
       "benchfam_mteb_sub", "prospective_epoch", "prospective_epoch_bymodel",
       "prospective_mteb2026"])
# Short jobs that keep one process (spawning a pool would take longer)
SERIAL_JOBS = ["greedy_cv_mmlu", "greedy_cv_olv2sub", "baselines_olv2sub",
               "baselines_helm_lite", "logit_mmlu", "logit_helm_lite",
               "logit_olv2sub", "stability_mmlu", "estimator_mmlu",
               "family_epoch", "token_fixed", "imputers_merged", "numbers"]


def _commands(plan):
    """--dry-run plan -> {job: command line} (the job from its stdout
    file)."""
    cmds = {}
    for ln in plan.splitlines():
        if ln.startswith("(cd "):
            name = ln.rsplit("/stdout/", 1)[1].split(".txt")[0]
            cmds[name] = ln
    return cmds


def _cores(listing):
    """--list -> {job: CPU column}."""
    out = {}
    for ln in listing.splitlines()[2:]:
        tok = ln.split()
        i = next(i for i, t in enumerate(tok) if t in ("yes", "no"))
        out[tok[0]] = int(tok[i + 2])
    return out


class TestWorkers:
    @pytest.mark.parametrize("max_jobs,workers", [(64, 16), (60, 15),
                                                  (200, 16), (20, 5), (3, 1)])
    def test_default(self, tmp_path, max_jobs, workers):
        listing = _run(tmp_path, "--list", MAX_JOBS=str(max_jobs))
        assert listing.splitlines()[0].endswith(f"WORKERS {workers}")

    @pytest.mark.parametrize("workers,shard", [(4, 4), (16, 10)])
    def test_commands_and_cores(self, tmp_path, workers, shard):
        """Every parallel job passes WORKERS and counts that many cores;
        an imputer shard at most its 10 splits; the short jobs neither."""
        env = dict(WORKERS=str(workers), MAX_JOBS="64")
        cmds = _commands(_run(tmp_path, "--dry-run", "--repeats", "2",
                              **env))
        cores = _cores(_run(tmp_path, "--list", "--repeats", "2", **env))
        assert not (tmp_path / "root").exists()
        for name in PAR_JOBS:
            flag = "--jobs" if name.startswith("estimator_") else "--workers"
            assert f" {flag} {workers}) " in cmds[name], name
            assert cores[name] == workers, name
        shards = [n for n in cmds if "_rep" in n and n.startswith("imputers_")]
        assert len(shards) == 2 * len(DATASETS)
        for name in shards:
            assert f" --reps {name[-1]} --workers {shard}) " in cmds[name], \
                name
            assert cores[name] == shard, name
        for name in SERIAL_JOBS:
            assert "--workers" not in cmds[name] and \
                "--jobs" not in cmds[name], name
            assert cores[name] == 1, name
        assert set(PAR_JOBS) | set(shards) == {n for n, c in cores.items()
                                               if c > 1}

    def test_estimates_follow_cores(self, tmp_path):
        """BS_EST = core-seconds / cores: halving WORKERS doubles the
        estimate of a parallel job and leaves a serial one alone."""
        est = {}
        for w in (8, 16):
            listing = _run(tmp_path, "--list", WORKERS=str(w), MAX_JOBS="64")
            est[w] = {ln.split()[0]: ln for ln in listing.splitlines()[2:]}

        def e(w, name):
            tok = est[w][name].split()
            i = next(i for i, t in enumerate(tok) if t in ("yes", "no"))
            return int(tok[i + 1])
        assert e(8, "estimator_merged") == 31400 // 8
        assert e(16, "estimator_merged") == 31400 // 16
        assert e(16, "imputers_merged_rep0") == 5660 // 10
        assert e(8, "imputers_merged_rep0") == 5660 // 8
        assert e(8, "token_fixed") == e(16, "token_fixed") == 820


SCHED = r"""
set -euo pipefail
source "$1/run_all.sh"
BS_ROOT=$2/root
BS_RUN_ID=test
BS_STATE_DIR=$BS_ROOT/.state/test
BS_MAX_JOBS=4
mkdir -p "$BS_ROOT"/stdout "$BS_ROOT"/.state/locks "$BS_ROOT"/.state/last \
    "$BS_STATE_DIR"
bs_job L g "" "" sleep 1.0
bs_job M g "" "" sleep 0.3
bs_job S g "" "" sleep 0.2
bs_job T g "" "" sleep 0.2
BS_WEIGHT[L]=2 BS_WEIGHT[M]=4
BS_EST[L]=10 BS_EST[M]=9 BS_EST[S]=1 BS_EST[T]=20
BS_SELECTED=(L M S T)
for n in L M S T; do BS_IS_SEL[$n]=1; done
bs_run_selected > /dev/null
for n in L M S T; do cut -f1,3,5,6 "$BS_STATE_DIR/$n.status"; done
"""


class TestScheduler:
    def test_easy_backfill(self, tmp_path):
        """MAX_JOBS 4: L (2 cores, est 10 s) runs, M (4 cores) is held
        until L ends; S (est 1 s) ends before M could start and is
        backfilled at once; T (est 20 s) would delay M (no spare core) and
        waits until M is done."""
        (tmp_path / "sched.sh").write_text(SCHED)
        res = subprocess.run(["bash", str(tmp_path / "sched.sh"), str(CODE),
                              str(tmp_path)], capture_output=True, text=True,
                             timeout=120)
        assert res.returncode == 0, res.stderr[-2000:]
        rec = {}
        for ln in res.stdout.splitlines():
            name, status, t0, t1 = ln.split("\t")
            assert status == "ok", ln
            rec[name] = (float(t0), float(t1))
        assert set(rec) == {"L", "M", "S", "T"}
        assert rec["S"][0] < rec["L"][1]          # backfilled
        assert rec["M"][0] >= rec["L"][1]         # held for its 4 cores
        assert rec["T"][0] >= rec["M"][1]         # would have delayed M
        # never more than 4 cores busy
        cores = dict(L=2, M=4, S=1, T=1)
        for name, (t0, _) in rec.items():
            busy = sum(cores[n] for n, (a, b) in rec.items() if a <= t0 < b)
            assert busy <= 4, (name, busy)
