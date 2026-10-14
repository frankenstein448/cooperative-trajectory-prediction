"""
final_test_guard.py
One evaluation per checkpoint on the split v2 TEST split (runs/DECISION_RULES_FINAL_TEST.md), shared by
eval_scenes.py --final-test (Part 1) and eval_env.py --split test (Part 2).

guard(): before anything is loaded. The evaluation's stdout and stderr go to its own log (opened with "x").
A re-run of a checkpoint is allowed only if EVERY earlier attempt crashed before printing any result line
(is_result(line) False for every line of every attempt log):
  - attempts = <stem>_attempt<N>.log (N = 1, 2, ... in order) plus the current <stem>.log;
  - if runs/TEST_EVALS.log has more final-test entries for (run_dir, ckpt) than there are attempt logs, the
    re-run is refused (a missing log cannot be checked);
  - on an allowed re-run the earlier <stem>.log is kept as <stem>_attempt<N>.log, and the entry is marked
    "(re-run after crash, no results seen)".
record(): appends the entry to runs/TEST_EVALS.log BEFORE the evaluation, so a crash is on record.
This is the turning test's rule (runs/DECISION_RULES_TURNING_TEST.md) with this study's purpose and logs.
"""
import os
import re
import sys
import subprocess
from datetime import datetime

TEST_EVALS_LOG = "runs/TEST_EVALS.log"
FINAL_TEST_PURPOSE = "final test evaluation (split v2)"
RERUN_SUFFIX = " (re-run after crash, no results seen)"
# lines printed only once predictions exist: eval_scenes.py (scene / ALL / CV rows, sanity, SAFE / RANK / COND
# blocks) and eval_env.py (ENV / COV tables)
SCENES_RESULT = re.compile(r"^(Sanity|Town0\d\s+\d+\s+\d+|ALL\s+\d|CV\s+\d|\s+CV: n=|Median|Scenes|SAFE|RANK|COND)")


def is_scenes_result(line):
    return bool(SCENES_RESULT.match(line))


def is_env_result(line):
    return line.startswith(("ENV", "COV"))


IS_RESULT = {"scenes": is_scenes_result, "env": is_env_result}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def logged_entries(path=TEST_EVALS_LOG):
    """(run_dir, ckpt, purpose) of every entry, paths normalised."""
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                fields = dict(kv.split("=", 1) for kv in line.rstrip("\n").split("\t")[1:] if "=" in kv)
                out.append((os.path.normpath(fields.get("run_dir", "")), os.path.normpath(fields.get("ckpt", "")),
                            fields.get("purpose", "")))
    return out


def attempt_logs(log_name, directory="."):
    stem = log_name[:-len(".log")]
    old = sorted((f for f in os.listdir(directory) if f.startswith(stem + "_attempt") and f.endswith(".log")),
                 key=lambda f: int(f[len(stem + "_attempt"):-len(".log")]))
    return old + ([log_name] if os.path.exists(os.path.join(directory, log_name)) else [])


def check_rerun(log_name, run_dir, ckpt, is_result, tests_log=TEST_EVALS_LOG, directory="."):
    """-> (allowed, rerun, reason). Pure check, no side effects."""
    key = (os.path.normpath(run_dir), os.path.normpath(ckpt))
    n_entries = sum(1 for r, c, p in logged_entries(tests_log) if (r, c) == key and p.startswith(FINAL_TEST_PURPOSE))
    logs = attempt_logs(log_name, directory)
    if n_entries > len(logs):
        return False, True, (f"{n_entries} final-test entries for run_dir={run_dir} ckpt={ckpt} in {tests_log} but "
                             f"only {len(logs)} attempt logs ({logs}); cannot verify that no result was seen")
    for f_old in logs:
        with open(os.path.join(directory, f_old)) as fh:
            if any(is_result(line) for line in fh):
                return False, True, f"{f_old} contains result lines; that attempt is final"
    return True, bool(logs) or n_entries > 0, ""


def guard(log_name, run_dir, ckpt, is_result, error):
    """Enforces the rule, then tees stdout / stderr into a fresh log_name. error(msg) must not return.
    -> rerun (bool)."""
    allowed, rerun, reason = check_rerun(log_name, run_dir, ckpt, is_result)
    if not allowed:
        error(f"final test: {reason} -- refusing a re-run")
    if os.path.exists(log_name):
        attempt = f"{log_name[:-len('.log')]}_attempt{len(attempt_logs(log_name))}.log"
        assert not os.path.exists(attempt), f"{attempt} already exists"
        os.rename(log_name, attempt)
    fh = open(log_name, "x")
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    print(f"Writing this output to {log_name}")
    if rerun:
        print(f"Final-test RE-RUN after crash: earlier attempt logs "
              f"{[f for f in attempt_logs(log_name) if f != log_name]} have no result line")
    return rerun


def record(run_dir, ckpt, rerun, split_version, tests_log=TEST_EVALS_LOG):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception as e:
        commit = f"<unavailable: {e}>"
    with open(tests_log, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\trun_dir={run_dir}\tgit_commit={commit}\t"
                f"ckpt={ckpt}\tpurpose={FINAL_TEST_PURPOSE}{RERUN_SUFFIX if rerun else ''}"
                f"\tsplit_version={split_version}\n")
    print(f"Appended this test evaluation to {tests_log}")


if __name__ == "__main__":
    # run_final_test.sh: python final_test_guard.py <log> <run_dir> <ckpt> scenes|env
    #   -> prints "fresh" (no attempt yet), "rerun" (allowed: every attempt crashed before any result) or
    #      "final: <reason>" (an attempt has results, or a logged attempt has no log)
    log, run_dir, ckpt, kind = sys.argv[1:5]
    allowed, rerun, reason = check_rerun(log, run_dir, ckpt, IS_RESULT[kind])
    print("rerun" if allowed and rerun else "fresh" if allowed else f"final: {reason}")
