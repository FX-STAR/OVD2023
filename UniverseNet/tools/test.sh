CONFIG=$1
CHECKPOINT=$2
OUTDIR=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --format-only --show-dir $OUTDIR --show-score-thr 0.5