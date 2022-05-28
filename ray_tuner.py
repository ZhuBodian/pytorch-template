import train
import argparse
import torch
import numpy as np
from parse_config import ConfigParser
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import collections
from functools import partial
import ray
"""
注意如果要在pycharm中调试，要加上ray.init(local_mode=True)
"""


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def start_tune(tuner_config, config):
    config['ray_tune']['tune'] = True
    config['ray_tune']['args']['batch_size'] = tuner_config['batch_size']
    config['ray_tune']['args']['lr'] = tuner_config['lr']
    config['ray_tune']['args']['step_size'] = tuner_config['step_size']

    config['data_loader']['args']['batch_size'] = tuner_config['batch_size']
    config['optimizer']['args']['lr'] = tuner_config['lr']
    config['lr_scheduler']['args']['step_size'] = tuner_config['step_size']

    train.main(config)


def main(config, max_num_epochs):
    # ray.init(local_mode=True)  # 用于在pycharm中调试
    tuner_config = {
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "step_size": tune.sample_from(lambda _: 2**np.random.randint(5, 9))
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = tune.CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["val_loss", "val_accuracy", "training_iteration"])

    result = tune.run(
        partial(start_tune, config=config),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=tuner_config,
        scheduler=scheduler,
        progress_reporter=reporter,
        num_samples=8  # 超参数的采样组数
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)

    # max_num_epoch为每组超参数，单词收缩的最大epoch数
    main(config, max_num_epochs=10)