#!/usr/bin/env bash
# Clean, sequential timing: ORIGINAL (NumPy 1.26) vs FIXED (NumPy 1.26) vs FIXED (NumPy 2.4), example data,
# phi=1e-2, seed=1, 1 BLAS thread, interleaved repeats. Every run hard-killed after 600 s.
OUT=/work/fixcheck/timing; mkdir -p $OUT
T=/work/proj/benchmarks/run_timed.py
ORIG=/work/fixcheck/prscsx_orig; FIX=/work/proj/prscsx_fixed
declare -A PY=( [orig_np1.26]=/work/venv_ref/bin/python [fixed_np1.26]=/work/venv_ref/bin/python [fixed_np2.4]=/usr/local/bin/python3 )
declare -A CODE=( [orig_np1.26]=$ORIG [fixed_np1.26]=$FIX [fixed_np2.4]=$FIX )
for rep in 1 2 3; do
  for v in orig_np1.26 fixed_np1.26 fixed_np2.4; do
    d=$OUT/${v}_rep$rep; mkdir -p $d
    ( cd ${CODE[$v]} && timeout -s KILL 600 ${PY[$v]} $T --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
        --json $d/timing.json --log $d/log.txt -- ${PY[$v]} PRScsx.py --ref_dir=/work/ref/1kg --bim_prefix=test_data/test \
        --sst_file=test_data/EUR_sumstats.txt,test_data/EAS_sumstats.txt --n_gwas=200000,100000 --pop=EUR,EAS --chrom=22 \
        --phi=1e-2 --seed=1 --out_dir=$d --out_name=test ) > $d/driver.log 2>&1
    echo "$v rep$rep $(cat $d/driver.log)"
  done
done
echo ALL_DONE
