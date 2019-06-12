NUM_WORKERS=10
BATCH_SIZE=8

EVAL_METRIC=pck
PCK_ALPHA=0.1

TPS_REG_FACTOR=0.0

DATASET=pf-pascal
#DATASET=pf-willow

MODEL_DIR=trained_models
MODEL_DIR=self-bd

W_MATCH=1.0
W_CYCLE=0.0
W_TRANS=0.0
W_GRID=0.0
W_ID=0.0

#MODEL_PATH=$MODEL_DIR/pami.pth.tar
#MODEL_PATH=$MODEL_DIR/best_match_${W_MATCH}_cycle_${W_CYCLE}_trans_${W_TRANS}_grid_${W_GRID}.pth.tar
MODEL_PATH=$MODEL_DIR/match_${W_MATCH}_cycle_${W_CYCLE}_trans_${W_TRANS}_grid_${W_GRID}_id_${W_ID}.pth.tar
#MODEL_PATH=$MODEL_DIR/pretrain.pth.tar
#MODEL_PATH=$MODEL_DIR/best_tt.pth.tar
#MODEL_PATH=$MODEL_DIR/weakalign.pth.tar

python eval.py \
    --model $MODEL_PATH \
    --num-workers $NUM_WORKERS \
    --eval-dataset $DATASET \
    --pck-alpha $PCK_ALPHA \
    --eval-metric $EVAL_METRIC \
    --batch-size $BATCH_SIZE \
    --tps-reg-factor $TPS_REG_FACTOR
