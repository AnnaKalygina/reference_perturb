#!/usr/bin/env bash
#SBATCH --output=/cluster/work/boeva/reference_perturb/v1/logs/%x_%j.out
#SBATCH --cpus-per-task=2
set -eo pipefail
export PATH=/cluster/work/boeva/akalygina/conda/envs/vcell/bin:$PATH
export TMPDIR=/cluster/work/boeva/reference_perturb/v1/tmp
cd /cluster/work/boeva/reference_perturb/v1
/usr/bin/time -f "peak_rss_kb %M wall_s %e" python ingest/"$1".py "${@:2}"
