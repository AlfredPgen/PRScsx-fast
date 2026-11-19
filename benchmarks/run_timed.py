#!/usr/bin/env python
"""
Run a command as a child process and record wall time, CPU time and peak RSS.

Usage:
    python run_timed.py --json OUT.json --log OUT.log [--env KEY=VAL ...] -- cmd arg1 arg2 ...

Peak memory is taken from os.wait4() rusage of that specific child (ru_maxrss, KiB on Linux),
so it is not contaminated by other children of this harness.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--env', action='append', default=[], help='KEY=VAL added to the child env')
    ap.add_argument('--timeout', type=float, default=None)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd

    env = dict(os.environ)
    for kv in args.env:
        k, v = kv.split('=', 1)
        env[k] = v

    t0 = time.perf_counter()
    with open(args.log, 'w') as logf:
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        timed_out = False
        if args.timeout is not None:
            deadline = t0 + args.timeout
            while True:
                pid, status, ru = os.wait4(p.pid, os.WNOHANG)
                if pid != 0:
                    break
                if time.perf_counter() > deadline:
                    p.kill()
                    timed_out = True
                    pid, status, ru = os.wait4(p.pid, 0)
                    break
                time.sleep(0.05)
        else:
            pid, status, ru = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0

    rec = {
        'cmd': cmd,
        'env_overrides': args.env,
        'exit_code': os.waitstatus_to_exitcode(status),
        'timed_out': timed_out,
        'wall_s': wall,
        'user_s': ru.ru_utime,
        'sys_s': ru.ru_stime,
        'max_rss_mb': ru.ru_maxrss / 1024.0,
    }
    with open(args.json, 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ('exit_code', 'timed_out', 'wall_s', 'user_s', 'max_rss_mb')}))
    sys.exit(0 if rec['exit_code'] == 0 else 1)


if __name__ == '__main__':
    main()
