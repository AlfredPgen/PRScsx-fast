#!/usr/bin/env bash
# Baseline runs of the ORIGINAL PRS-CSx on the repo's example dataset (chr22, 1000 SNPs, EUR+EAS).
# Uses the README-recommended environment (Python 3.11, NumPy 1.26.4, SciPy 1.11.4, h5py 3.10.0).
set -u
PY=${PY:-/work/venv_ref/bin/python}
REPO=${REPO:-/work/PRScsx}
REF=${REF:-/work/ref/1kg}
OUT=${OUT:-/work/baseline}
T=/work/proj/benchmarks/run_timed.py
TD=$REPO/test_data
ONE="--env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1"

common="--ref_dir=$REF --bim_prefix=$TD/test --chrom=22"
two="--sst_file=$TD/EUR_sumstats.txt,$TD/EAS_sumstats.txt --n_gwas=200000,100000 --pop=EUR,EAS"

run () {  # name, threads-flag, extra args...
  local name=$1; shift; local thr=$1; shift
  mkdir -p $OUT/$name
  echo "== $name"
  $PY $T --json $OUT/$name/timing.json --log $OUT/$name/log.txt $thr -- \
      $PY $REPO/PRScsx.py $common --out_dir=$OUT/$name --out_name=test "$@"
}

cd $REPO
# 1) phi=1e-2, seeds 1..5, single-threaded BLAS (fixed, fair reference)
for s in 1 2 3 4 5; do run phi1e-2_seed${s}_t1 "$ONE" $two --phi=1e-2 --seed=$s; done
# 2) determinism: exact rerun of seed 1
run phi1e-2_seed1_t1_rerun "$ONE" $two --phi=1e-2 --seed=1
# 3) same seed, default (multi-threaded) BLAS -> does thread count change results?
run phi1e-2_seed1_tdefault "" $two --phi=1e-2 --seed=1
# 4) meta, meta+write_pst, phi auto, single-population
run meta_phi1e-2_seed1_t1 "$ONE" $two --phi=1e-2 --seed=1 --meta=TRUE
run metapst_phi1e-2_seed1_t1 "$ONE" $two --phi=1e-2 --seed=1 --meta=TRUE --write_pst=TRUE
run phiauto_meta_seed1_t1 "$ONE" $two --seed=1 --meta=TRUE
run eur_only_phi1e-2_seed1_t1 "$ONE" --sst_file=$TD/EUR_sumstats.txt --n_gwas=200000 --pop=EUR --phi=1e-2 --seed=1
echo ALL_DONE
