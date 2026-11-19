#!/usr/bin/env python
"""
Restart-safe job queue for long benchmarks (the sandbox can be restarted at any time).

  python jobs.py run --queue DIR [--slots 1]       run all pending jobs (re-run after a restart: finished jobs are skipped)
  python jobs.py status --queue DIR

A job is DIR/jobs/<id>.json: {"id", "cmd": [... "{OUT}" ...], "out_dir", "env": {}, "timeout": s}
"{OUT}" in cmd is replaced by a temporary directory that is renamed to out_dir only after success.
On completion DIR/done/<id>.json records exit code, wall time, CPU time and peak RSS (os.wait4 of the child).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time


def load_jobs(q):
    jobs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(q, 'jobs', '*.json')))]
    return sorted(jobs, key=lambda j: (j.get('priority', 0), j['id']))


def status(q):
    jobs = load_jobs(q)
    done = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(q, 'done', '*.json'))}
    failed = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(q, 'failed', '*.json'))}
    print('jobs=%d done=%d failed=%d pending=%d' % (len(jobs), len(done), len(failed - done), len(jobs) - len(done | failed)))
    return jobs, done, failed


def start(q, job):
    tmp = job['out_dir'] + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    cmd = [c.replace('{OUT}', tmp) for c in job['cmd']]
    env = dict(os.environ); env.update(job.get('env', {}))
    logf = open(os.path.join(q, 'logs', job['id'] + '.log'), 'w')
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return {'p': p, 'job': job, 'tmp': tmp, 't0': time.perf_counter(), 'log': logf}


def finish(q, r, status_, ru):
    job = r['job']; r['log'].close()
    rec = {'id': job['id'], 'exit_code': os.waitstatus_to_exitcode(status_) if status_ is not None else -9,
           'wall_s': time.perf_counter() - r['t0'], 'user_s': ru.ru_utime if ru else None,
           'sys_s': ru.ru_stime if ru else None, 'max_rss_mb': ru.ru_maxrss / 1024.0 if ru else None,
           'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
    if rec['exit_code'] == 0:
        shutil.rmtree(job['out_dir'], ignore_errors=True)
        os.rename(r['tmp'], job['out_dir'])
        json.dump(rec, open(os.path.join(job['out_dir'], 'job_timing.json'), 'w'), indent=1)
        json.dump(rec, open(os.path.join(q, 'done', job['id'] + '.json'), 'w'), indent=1)
    else:
        json.dump(rec, open(os.path.join(q, 'failed', job['id'] + '.json'), 'w'), indent=1)
    print('%s %-60s exit=%d wall=%.1fs rss=%.0fMB' % (time.strftime('%H:%M:%S'), job['id'], rec['exit_code'], rec['wall_s'],
                                                      rec['max_rss_mb'] or -1), flush=True)


def run(q, slots):
    for d in ('done', 'failed', 'logs'):
        os.makedirs(os.path.join(q, d), exist_ok=True)
    for f in glob.glob(os.path.join(q, 'failed', '*.json')):     # retry failures from a previous (killed) session
        os.remove(f)
    jobs, done, _ = status(q)
    pending = [j for j in jobs if j['id'] not in done]
    running = []
    while pending or running:
        while pending and len(running) < slots:
            j = pending.pop(0)
            if not os.path.exists(os.path.join(q, 'jobs', j['id'] + '.json')):   # job removed while queued
                continue
            running.append(start(q, j))
        time.sleep(0.5)
        for r in list(running):
            pid, st, ru = os.wait4(r['p'].pid, os.WNOHANG)
            if pid != 0:
                finish(q, r, st, ru); running.remove(r)
            elif time.perf_counter() - r['t0'] > r['job'].get('timeout', 1e9):
                r['p'].kill(); pid, st, ru = os.wait4(r['p'].pid, 0)
                finish(q, r, st, ru); running.remove(r)
    status(q)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'status']); ap.add_argument('--queue', required=True)
    ap.add_argument('--slots', type=int, default=1)
    a = ap.parse_args()
    run(a.queue, a.slots) if a.cmd == 'run' else status(a.queue)
