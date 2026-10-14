"""
test_final_test_guard.py
Unit tests for final_test_guard.py (runs/DECISION_RULES_FINAL_TEST.md's one-evaluation-per-checkpoint rule) and the
result-line detectors (eval_scenes.py / eval_env.py output), in a temporary directory: no data, no GPU, and the real
runs/TEST_EVALS.log is never touched.

Run in gpu-test-venv:  python test_final_test_guard.py
"""
import os
import sys
import tempfile
import traceback

import final_test_guard as g

scenes_result, env_result = g.is_scenes_result, g.is_env_result
RUN, CKPT = "runs/C_D_seed0", "runs/C_D_seed0/checkpoint_epoch12.pt"
LOG = "eval_C_D_seed0_test_final.log"


class Refused(Exception):
    pass


def refuse(msg):
    raise Refused(msg)


def in_tmp(fn):
    def wrapped():
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            os.makedirs("runs")
            try:
                fn()
            finally:
                sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
                os.chdir(old)
    wrapped.__name__ = fn.__name__
    return wrapped


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def entry(purpose=g.FINAL_TEST_PURPOSE, run=RUN, ckpt=CKPT):
    with open(g.TEST_EVALS_LOG, "a") as f:
        f.write(f"2026-10-14T10:00:00\trun_dir={run}\tgit_commit=x\tckpt={ckpt}\tpurpose={purpose}\tsplit_version=v2\n")


@in_tmp
def test_first_run_allowed_and_recorded():
    rerun = g.guard(LOG, RUN, CKPT, scenes_result, refuse)
    g.record(RUN, CKPT, rerun, "v2")
    print("Town05   100    10   1.000m")
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    assert rerun is False and os.path.exists(LOG)
    line = open(g.TEST_EVALS_LOG).read().strip().split("\t")
    assert f"run_dir={RUN}" in line and f"ckpt={CKPT}" in line and line[-2] == f"purpose={g.FINAL_TEST_PURPOSE}"
    assert line[-1] == "split_version=v2"
    assert "Town05   100" in open(LOG).read()


@in_tmp
def test_completed_attempt_is_final():
    entry()
    write(LOG, "Writing this output\nSanity: mean distance of first GT point from origin = 1.0 m\n")
    try:
        g.guard(LOG, RUN, CKPT, scenes_result, refuse)
        raise AssertionError("re-run after results must be refused")
    except Refused as e:
        assert "result lines" in str(e)
    assert os.path.exists(LOG) and not os.path.exists(LOG.replace(".log", "_attempt1.log"))


@in_tmp
def test_crash_without_results_allows_marked_rerun():
    entry()
    write(LOG, "Writing this output\nLoading metadata\nTraceback (most recent call last):\nRuntimeError: CUDA\n")
    rerun = g.guard(LOG, RUN, CKPT, scenes_result, refuse)
    g.record(RUN, CKPT, rerun, "v2")
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    assert rerun is True
    assert "Traceback" in open(LOG.replace(".log", "_attempt1.log")).read()
    assert "RE-RUN after crash" in open(LOG).read()
    lines = open(g.TEST_EVALS_LOG).read().splitlines()
    assert len(lines) == 2 and lines[1].split("\t")[-2] == f"purpose={g.FINAL_TEST_PURPOSE}{g.RERUN_SUFFIX}"


@in_tmp
def test_second_crash_numbering_and_any_attempt_with_results_blocks():
    entry(); entry()
    write(LOG.replace(".log", "_attempt1.log"), "crash\n")
    write(LOG, "crash again\n")
    rerun = g.guard(LOG, RUN, CKPT, scenes_result, refuse)
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    assert rerun and os.path.exists(LOG.replace(".log", "_attempt2.log"))
    # an attempt log with results anywhere blocks, even if the latest attempt crashed early
    write(LOG.replace(".log", "_attempt1.log"), "COND ALL     turning   10\n")
    write(LOG, "crash\n")
    try:
        g.guard(LOG, RUN, CKPT, scenes_result, refuse)
        raise AssertionError("must refuse")
    except Refused:
        pass


@in_tmp
def test_missing_log_refused():
    entry()  # logged, but its output log is gone: cannot verify that nothing was seen
    try:
        g.guard(LOG, RUN, CKPT, scenes_result, refuse)
        raise AssertionError("must refuse")
    except Refused as e:
        assert "cannot verify" in str(e)


@in_tmp
def test_other_purposes_and_checkpoints_do_not_count():
    entry(purpose="confirmatory condition evaluation (turning hypothesis from tuning)")
    entry(ckpt="runs/C_D_seed0/checkpoint_epoch11.pt")
    entry(run="runs/C_D_seed1", ckpt="runs/C_D_seed1/checkpoint_epoch12.pt")
    allowed, rerun, _ = g.check_rerun(LOG, RUN, CKPT, scenes_result)
    assert allowed and not rerun


def test_result_line_detectors():
    results = ["Sanity: mean CV error at step 1 = 0.1 m", "Town03     5   250   2.000m   4.000m",
               "ALL          2208   2.000m", "CV          2000 cvADE=", "    CV: n=  250 excluded=  0",
               "Median of per-scene minADE6: 2.0 m", "Scenes where top-1 ADE beats stay-put: 8/8",
               "SAFE ALL", "RANK ALL", "COND ALL     turning", "CONDITIONS  from ground truth"]
    not_results = ["Writing this output to x.log", "Using device: cuda", "PilotDataset: use_image_cache=True",
                   "Evaluating on the TEST split (2208 samples)", "Appended this test evaluation to runs/TEST_EVALS.log",
                   "map_transform=mirror, split_version=v2 (from config.json)", "ckpt=runs/C_D_seed0/x.pt  use_map=True",
                   "Town05 lanes loaded", "Traceback (most recent call last):"]
    assert all(scenes_result(l) for l in results), [l for l in results if not scenes_result(l)]
    assert not any(scenes_result(l) for l in not_results), [l for l in not_results if scenes_result(l)]
    assert env_result("ENV       ALL all") and env_result("COV       ALL samples=") and not env_result("EnvDataset: v2x=on")


@in_tmp
def test_cli_status():
    import subprocess
    cli = lambda: subprocess.run([sys.executable, g.__file__, LOG, RUN, CKPT, "scenes"], capture_output=True,
                                 text=True).stdout.strip()
    assert cli() == "fresh"
    entry(); write(LOG, "Traceback\n")
    assert cli() == "rerun"
    write(LOG, "ALL          2208   2.000m\n")
    assert cli().startswith("final: ")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
