NUM_WORKERS=10
BATCH_SIZE=8

EVAL_METRIC=pck
PCK_ALPHA=0.1

DATASET=pf-pascal

MODEL_DIR=trained_models

W_MATCH=1.0
W_CYCLE=0.0
W_TRANS=0.0
W_COSEG=0.0
W_TASK=0.0

MODEL_PATH=$MODEL_DIR/best_match_${W_MATCH}_cycle_${W_CYCLE}_trans_${W_TRANS}_coseg_${W_COSEG}_task_${W_TASK}.pth.tar

python eval.py \
    --model $MODEL_PATH \
    --num-workers $NUM_WORKERS \
    --eval-dataset $DATASET \
    --pck-alpha $PCK_ALPHA \
    --eval-metric $EVAL_METRIC \
    --batch-size $BATCH_SIZE \
