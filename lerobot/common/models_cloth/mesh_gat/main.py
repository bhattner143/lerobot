import os
from pipeline.trainer import *
from pipeline.evaluator import *
from utils.configs_utils import get_args
from utils.train_utils import set_seed, is_distributed
from pathlib import Path

if __name__ == '__main__':
    # set current task
    mode = "test" # select mode from 'train', 'test', 'test_real'
    name_cloth = 't_shirt_l3'
    checkpoint_file = '2025-03-28/14-46-38/finalbestmodel_0299_0.01162.pt'

    # get address
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASET_DIR     = Path(f'/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat/{name_cloth}')
    PREDICT_DIR     = DATASET_DIR / 'predict'
    CHECKPOINT_DIR  = DATASET_DIR / 'checkpoints'
    CHECKPOINT_FILE = CHECKPOINT_DIR / checkpoint_file
    TEMPLATE_DIR    = DATASET_DIR / f'configs/template_{name_cloth}.pickle'

    # make dir
    if not os.path.exists(PREDICT_DIR):
        os.makedirs(PREDICT_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # load arguments
    cfg = get_args(DATASET_DIR, name_cloth)
    set_seed(cfg.main_seed)

    # append to cfg
    cfg.mode = mode
    cfg.name_cloth = name_cloth
    cfg.project_dir = PROJECT_DIR
    cfg.dataset_dir = DATASET_DIR
    cfg.predict_dir = PREDICT_DIR
    cfg.checkpoint_dir = CHECKPOINT_DIR
    cfg.checkpoint_file = CHECKPOINT_FILE
    cfg.template_dir = TEMPLATE_DIR

    # checking if use distributed training. and then print config and assert the backbone.
    cfg.distributed = False
    if is_distributed():
        cfg.distributed = True

    if mode == 'train':
        trainer(cfg)
    elif mode == 'test':
        evaluator(cfg)
    elif mode == 'test_real':
        evaluator_real(cfg)


