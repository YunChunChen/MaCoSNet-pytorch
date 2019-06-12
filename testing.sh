MODEL_DIR=trained_models

#MODEL_PATH=$MODEL_DIR/pretrain.pth.tar
MODEL_PATH=$MODEL_DIR/weakalign.pth.tar

GPU=0
NUM_WORKERS=4
BATCH_SIZE=8
LEARNING_RATE=5e-8
EPOCH=60

DATASET=pf-pascal

MATCH_LOSS=True
CYCLE_LOSS=True
TRANS_LOSS=False
GRID_LOSS=False
ID_LOSS=True

W_MATCH=1.0
W_CYCLE=1.0
W_TRANS=0.0
W_GRID=0.0

python train.py --model $MODEL_PATH --training-dataset $DATASET --num-epochs $EPOCH --lr $LEARNING_RATE --gpu $GPU --num-workers $NUM_WORKERS --batch-size $BATCH_SIZE --result-model-dir $MODEL_DIR --match-loss $MATCH_LOSS --cycle-loss $CYCLE_LOSS --trans-loss $TRANS_LOSS --grid-loss $GRID_LOSS --id-loss $ID_LOSS --w-match $W_MATCH --w-cycle $W_CYCLE --w-trans $W_TRANS --w-grid $W_GRID
