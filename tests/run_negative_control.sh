#!/usr/bin/env bash
# Negative control: run every regression test, one process per test, against a given PRS-CSx code dir
# (normally the ORIGINAL, unfixed code). Each test is hard-killed (SIGKILL) at the deadline, because the
# original's bare `except:` swallows the exceptions pytest-timeout / Ctrl-C use to stop a hung test.
#
# Usage: run_negative_control.sh PYTHON CODE_DIR NODEIDS_FILE OUT_TSV [TIMEOUT_S]
# Output TSV columns: nodeid  status(PASS|FAIL|KILLED_TIMEOUT)  exit_code  seconds
PY=$1; CODE=$2; IDS=$3; OUT=$4; TO=${5:-90}
HERE=$(cd "$(dirname "$0")" && pwd)
: > "$OUT"
while read -r id; do
  t0=$(date +%s.%N)
  ( cd "$HERE/.." && PRSCSX_FIXED_DIR="$CODE" PRSCSX_REF_DIR="${PRSCSX_REF_DIR:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      timeout -s KILL "$TO" "$PY" -m pytest -q -p no:cacheprovider "$id" > /dev/null 2>&1 )
  rc=$?
  dt=$(echo "$(date +%s.%N) - $t0" | bc)
  case $rc in 0) st=PASS;; 137) st=KILLED_TIMEOUT;; *) st=FAIL;; esac
  printf '%s\t%s\t%d\t%.1f\n' "$id" "$st" "$rc" "$dt" >> "$OUT"
done < "$IDS"
echo DONE >> "$OUT.done"
