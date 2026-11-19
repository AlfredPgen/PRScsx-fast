#!/usr/bin/env python
"""
Per-stage wall-clock timing of the ORIGINAL PRS-CSx code (modules imported unchanged from --repo).
Mirrors PRScsx.main() for a single chromosome; only adds timers.

Usage: python stage_timing_original.py --repo /work/PRScsx -- <normal PRScsx.py arguments>
"""
import io
import json
import os
import sys
import time
import contextlib

repo = sys.argv[sys.argv.index('--repo') + 1]
out_json = sys.argv[sys.argv.index('--json') + 1]
sys.path.insert(0, repo)
prs_args = sys.argv[sys.argv.index('--') + 1:]

import parse_genet, mcmc_gtb  # noqa: E402
sys.argv = ['PRScsx.py'] + prs_args
import PRScsx  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    pd = PRScsx.parse_param()
n_pop = len(pd['pop'])
T = {}
t = time.perf_counter()
chrom = int(pd['chrom'][0])
ref_dict = parse_genet.parse_ref(pd['ref_dir'] + '/snpinfo_mult_1kg_hm3', chrom, '1kg')
T['parse_ref'] = time.perf_counter() - t; t = time.perf_counter()
vld_dict = parse_genet.parse_bim(pd['bim_prefix'], chrom)
T['parse_bim'] = time.perf_counter() - t; t = time.perf_counter()
sst_dict = {pp: parse_genet.parse_sumstats(ref_dict, vld_dict, pd['sst_file'][pp], pd['pop'][pp], pd['n_gwas'][pp]) for pp in range(n_pop)}
T['parse_sumstats'] = time.perf_counter() - t; t = time.perf_counter()
ld_blk, blk_size = {}, {}
for pp in range(n_pop):
    ld_blk[pp], blk_size[pp] = parse_genet.parse_ldblk(pd['ref_dir'], sst_dict[pp], pd['pop'][pp], chrom, '1kg')
T['parse_ldblk'] = time.perf_counter() - t; t = time.perf_counter()
snp_dict, beta_dict, frq_dict, idx_dict = parse_genet.align_ldblk(ref_dict, vld_dict, sst_dict, n_pop, chrom)
T['align_ldblk'] = time.perf_counter() - t; t = time.perf_counter()
mcmc_gtb.mcmc(pd['a'], pd['b'], pd['phi'], snp_dict, beta_dict, frq_dict, idx_dict, pd['n_gwas'], ld_blk, blk_size,
              pd['n_iter'], pd['n_burnin'], pd['thin'], pd['pop'], chrom, pd['out_dir'], pd['out_name'],
              pd['meta'], pd['write_pst'], pd['seed'])
T['mcmc_and_write'] = time.perf_counter() - t
T['total'] = sum(T.values())
T['n_snp_union'] = len(snp_dict['SNP'])
T['n_snp_per_pop'] = [len(beta_dict[pp]) for pp in range(n_pop)]
T['n_iter'] = pd['n_iter']
T['blk_size'] = {pd['pop'][pp]: [int(x) for x in blk_size[pp] if x > 0] for pp in range(n_pop)}
json.dump(T, open(out_json, 'w'), indent=1)
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in T.items() if k != 'blk_size'}))
