#!/bin/sh
# Download the PRS-CSx LD reference panels (the same files as in the original README) into one folder.
#
#   sh get_reference.sh OUT_DIR [1kg|ukbb] [POPS] [--broad] [--keep]
#     OUT_DIR  folder to create; pass it later as --ref_dir=OUT_DIR
#     1kg      1000 Genomes phase 3 panels (default); ukbb = UK Biobank panels
#     POPS     comma-separated ancestries (default EUR,EAS,AFR); any of AFR,AMR,EAS,EUR,SAS
#     --broad  download from the Broad mirror instead of Dropbox
#     --keep   keep the .tar.gz files after extraction
#   Sizes (1kg): AFR 4.4 GB, AMR 3.8, EAS 4.3, EUR 4.6, SAS 5.6, plus the SNP-info file 0.1 GB. About twice the
#   space is needed while a tarball is being extracted. Interrupted downloads resume when the command is rerun.
#
# Example: sh get_reference.sh ~/prscsx_ref 1kg EUR,EAS
set -e
OUT=$1
[ -n "$OUT" ] || { sed -n '2,13p' "$0"; exit 2; }
shift
PANEL=1kg; POPS=EUR,EAS,AFR; MIRROR=dropbox; KEEP=0
for a in "$@"; do
  case $a in
    1kg|ukbb) PANEL=$a ;;
    --broad) MIRROR=broad ;;
    --keep) KEEP=1 ;;
    *,*|AFR|AMR|EAS|EUR|SAS) POPS=$a ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done
command -v curl >/dev/null || { echo "curl is required"; exit 2; }

url() {   # url FILE -> download URL
  if [ "$MIRROR" = broad ]; then
    D=$(echo $PANEL | tr a-z A-Z)
    echo "https://personal.broadinstitute.org/hhuang/public/PRS-CSx/Reference/$D/$1"; return
  fi
  case $1 in
    snpinfo_mult_1kg_hm3)  echo "https://www.dropbox.com/s/rhi806sstvppzzz/snpinfo_mult_1kg_hm3?dl=1" ;;
    snpinfo_mult_ukbb_hm3) echo "https://www.dropbox.com/s/oyn5trwtuei27qj/snpinfo_mult_ukbb_hm3?dl=1" ;;
    ldblk_1kg_afr.tar.gz)  echo "https://www.dropbox.com/s/mq94h1q9uuhun1h/ldblk_1kg_afr.tar.gz?dl=1" ;;
    ldblk_1kg_amr.tar.gz)  echo "https://www.dropbox.com/s/uv5ydr4uv528lca/ldblk_1kg_amr.tar.gz?dl=1" ;;
    ldblk_1kg_eas.tar.gz)  echo "https://www.dropbox.com/s/7ek4lwwf2b7f749/ldblk_1kg_eas.tar.gz?dl=1" ;;
    ldblk_1kg_eur.tar.gz)  echo "https://www.dropbox.com/s/mt6var0z96vb6fv/ldblk_1kg_eur.tar.gz?dl=1" ;;
    ldblk_1kg_sas.tar.gz)  echo "https://www.dropbox.com/s/hsm0qwgyixswdcv/ldblk_1kg_sas.tar.gz?dl=1" ;;
    ldblk_ukbb_afr.tar.gz) echo "https://www.dropbox.com/s/dtccsidwlb6pbtv/ldblk_ukbb_afr.tar.gz?dl=1" ;;
    ldblk_ukbb_amr.tar.gz) echo "https://www.dropbox.com/s/y7ruj364buprkl6/ldblk_ukbb_amr.tar.gz?dl=1" ;;
    ldblk_ukbb_eas.tar.gz) echo "https://www.dropbox.com/s/fz0y3tb9kayw8oq/ldblk_ukbb_eas.tar.gz?dl=1" ;;
    ldblk_ukbb_eur.tar.gz) echo "https://www.dropbox.com/s/t9opx2ty6ucrpib/ldblk_ukbb_eur.tar.gz?dl=1" ;;
    ldblk_ukbb_sas.tar.gz) echo "https://www.dropbox.com/s/nto6gdajq8qfhh0/ldblk_ukbb_sas.tar.gz?dl=1" ;;
  esac
}

fetch() {   # fetch FILE: resumable download into OUT_DIR/FILE
  echo "downloading $1 ..."
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$OUT/$1" "$(url $1)" || {
    echo "* download of $1 failed; rerun to resume, or try --broad"; exit 1; }
}

mkdir -p "$OUT"
INFO=snpinfo_mult_${PANEL}_hm3
[ -s "$OUT/$INFO" ] && echo "$INFO already present" || fetch $INFO
for P in $(echo $POPS | tr ',' ' '); do
  p=$(echo $P | tr A-Z a-z)
  if [ -d "$OUT/ldblk_${PANEL}_$p" ] && [ "$(ls "$OUT/ldblk_${PANEL}_$p" | wc -l)" -ge 22 ]; then
    echo "ldblk_${PANEL}_$p already present"; continue
  fi
  T=ldblk_${PANEL}_$p.tar.gz
  fetch $T
  echo "extracting $T ..."
  tar -xzf "$OUT/$T" -C "$OUT"
  [ $KEEP = 1 ] || rm -f "$OUT/$T"
done
echo "done: use --ref_dir=$OUT"
ls "$OUT"
