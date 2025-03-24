# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import copy
import random
import string
from functools import partial

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import ray
from ray.util.actor_pool import ActorPool
import torch
from contextlib import nullcontext
from mmengine.model import is_model_wrapper
from mmengine.runner.amp import autocast

import wandb

print(os.environ.get("CUDA_VISIBLE_DEVICES"))
NUM_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(","))

if NUM_GPUS < 1:
    raise Exception("NUM_GPUS is must be greater than 0")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

@ray.remote(num_cpus=1, num_gpus=1)
class Actor:
    def train(self, cfg, project):
        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        print("STARTING TRAINING RUN")

        # start training
        runner.train()

        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val') 

        metrics = dict(tag)

        box_map_50 = metrics["coco/bbox_mAP_50"]

        return box_map_50

def train_crossval(cfg):
    num_gpus = NUM_GPUS
    pool = ActorPool([Actor.remote() for i in range(num_gpus)])
    print(pool)
    n_folds = 5


    run = wandb.init()

    lr = wandb.config.lr
    momentum = wandb.config.momentum

    results = []

    for i in range(1, n_folds+1):
        fold_cfg = copy.deepcopy(cfg)

        data_root = '/workspace/dataset/'
        
        train_annotations = f'train/train_annotations_{i}_{n_folds}.json'
        test_annotations = f'train/test_annotations_{i}_{n_folds}.json'

        classes = ('microglia', )
        backend_args = None

        metainfo=dict(classes=classes, palette=[200,20,60])

        fold_cfg.train_dataloader.dataset.dataset.data_root = data_root
        fold_cfg.train_dataloader.dataset.dataset.metainfo = metainfo 
        fold_cfg.train_dataloader.dataset.dataset.ann_file = train_annotations 
        fold_cfg.train_dataloader.dataset.dataset.data_prefix = dict(img='train/') 

        fold_cfg.test_dataloader.dataset.data_root = data_root
        fold_cfg.test_dataloader.dataset.metainfo = metainfo 
        fold_cfg.test_dataloader.dataset.ann_file = test_annotations 
        fold_cfg.test_dataloader.dataset.data_prefix = dict(img='train/') 

        fold_cfg.val_dataloader.dataset.data_root = data_root
        fold_cfg.val_dataloader.dataset.metainfo = metainfo 
        fold_cfg.val_dataloader.dataset.ann_file = test_annotations 
        fold_cfg.val_dataloader.dataset.data_prefix = dict(img='train/') 

        fold_cfg.test_evaluator.ann_file = data_root + train_annotations
        fold_cfg.val_evaluator.ann_file = data_root + test_annotations

        fold_cfg.optim_wrapper.optimizer.lr = lr
        fold_cfg.optim_wrapper.optimizer.momentum = momentum

        # fold_name = cv_name + f'_fold_{i}'

        print("Submitting fold", i)
        pool.submit(lambda a, args: a.train.remote(*args), (fold_cfg, 'microglia'))

    for i in range(n_folds):
        result = pool.get_next_unordered()
        results.append(result)

    print("LOGGING BOX MAP")
    wandb.log({"box_map": sum(results) / len(results)})


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    group_name = ''.join(random.choice(string.ascii_lowercase) for x in range(5))
    cv_name = 'cv_' + group_name 

    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "box_map"},
        "parameters": {
            "lr": {"max": 0.01, "min": 0.0001},
            "momentum": {"max": 1.0, "min": 0.2}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="microglia")

    wandb.agent(sweep_id, function=partial(train_crossval, cfg), count=10)


if __name__ == '__main__':
    main()
