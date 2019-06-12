import os

THRESHOLD = 2

NUM_OF_COORD = 100

DATASET_DIR = '/work/cascades/ycchen918/PAMI/datasets'

PF_PASCAL_DIR = os.path.join(DATASET_DIR, 'proposal-flow-pascal')

PF_WILLOW_DIR = os.path.join(DATASET_DIR, 'proposal-flow-willow')

TSS_DIR = os.path.join(DATASET_DIR, 'TSS_CVPR2016')

INTERNET_DIR = os.path.join(DATASET_DIR, 'Internet')


CSV_DIR = '/home/ycchen918/Project/MaCoSNet-pytorch/data/csv'

PF_PASCAL_TRAIN_DATA = os.path.join(CSV_DIR, 'pf-pascal', 'train.csv')
PF_PASCAL_EVAL_DATA = os.path.join(CSV_DIR, 'pf-pascal', 'eval.csv')
PF_PASCAL_TEST_DATA = os.path.join(CSV_DIR, 'pf-pascal', 'test.csv')

PF_WILLOW_TEST_DATA = os.path.join(CSV_DIR, 'pf-willow', 'test.csv')

TSS_TRAIN_DATA = os.path.join(CSV_DIR, 'tss', 'data.csv')
TSS_EVAL_DATA = os.path.join(CSV_DIR, 'tss', 'data.csv')

INTERNET_TRAIN_DATA = os.path.join(CSV_DIR, 'internet', 'data.csv')
INTERNET_EVAL_DATA = os.path.join(CSV_DIR, 'internet', 'data.csv')
