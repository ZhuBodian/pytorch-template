import argparse
import collections
import os
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from PrivateUtils import util, send_email, global_var
from parse_config import ConfigParser
from trainer import Trainer
from PrivateUtils.util import prepare_device
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
from ray import tune
from base.base_trainer import BaseTrainer
from torchvision import datasets


# fix random seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 自定义数据集获得targets
def get_split_targets():
    csv_path = os.path.join(config['data_loader']['args']['data_dir'], 'train.csv')
    csv_data = pd.read_csv(csv_path)
    targets = np.array(csv_data['label'])

    return targets


# 获得mnist数据集的targets
def get_split_mnist_targets():
    mnist = datasets.MNIST(config['data_loader']['args']['data_dir'], train=True, download=True)
    return mnist.targets


def single_fold_model(folds, fold_num, logger, train_index, valid_index):
    train_samples = SubsetRandomSampler(train_index)
    valid_samples = SubsetRandomSampler(valid_index)

    data_loader = config.init_obj('data_loader', module_data, train_val_samples=[train_samples, valid_samples],
                                  fold_num=fold_num)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    if fold_num == 0:
        logger.info(model)  # logging.info('输出信息')，而类model的输出信息为可训练参数
        global_var.get_value('email_log').add_log(model.__str__())

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    global_var.get_value('email_log').print_add(f'train on{device}')

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # config['loss']为'nll_loss'；criterion返回的实际是module_loss模块下的nll_loss函数的函数句柄，在special variables里面；
    criterion = getattr(module_loss, config['loss'])  # 损失函数的函数句柄
    metrics = [getattr(module_metric, met) for met in config['metrics']]  # tensorboard scalars中要跟踪的数据

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())  # 一个可迭代对象
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      folds=folds,
                      fold_num=fold_num)

    global_var.get_value('email_log').print_add(f'Fold: {fold_num}'.center(50, '*'))
    trainer.train()


def main(config):
    # 杂七杂八的输出与合理性验证
    global_var._init()
    if not config['ray_tune']['tune']:  # 不进行超参数搜索
        subject = '测试log'
    else:
        subject = str()
        for k, v in config['ray_tune']['args'].items():
            subject = subject + k + ':' + str(v) + ';  '
    global_var.set_value('email_log', send_email.Mylog(header='pytorch_template', subject=subject))

    logger = config.get_logger('train')  # 创建指定名字的日志文件，返回的为标准logging类

    folds = config['trainer']['folds']
    assert folds >= 1, 'par folds should >= 1'
    if folds == 1:
        global_var.get_value('email_log').print_add('Not run with K-fold'.center(100, '*'))
    else:
        global_var.get_value('email_log').print_add('Run with K-fold'.center(100, '*'))

    # 交叉验证并训练
    split = StratifiedShuffleSplit(n_splits=folds, test_size=config['data_loader']['args']['validation_split'])
    targets = get_split_mnist_targets()
    for fold_num, (train_index, valid_index) in enumerate(split.split(np.zeros_like(targets), targets)):
        single_fold_model(folds, fold_num, logger, train_index, valid_index)

    global_var.get_value('email_log').print_add(f'After {folds}-fold, average metric:'.center(100, '*'))
    for k, v in BaseTrainer.fold_best.items():
        BaseTrainer.fold_average[k] = sum(v) / len(v)
        global_var.get_value('email_log').print_add(f'{k}: {BaseTrainer.fold_average[k]}')

    if config['ray_tune']['tune']:
        tune.report(loss=BaseTrainer.fold_average['val_loss'], accuracy=BaseTrainer.fold_average['val_accuracy'])

    if not config['ray_tune']['tune']:
        global_var.get_value('email_log').send_mail()
        os.system('shutdown')


if __name__ == '__main__':
    util.seed_everything(42)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    # 这里专门把需要多次更改的超参数给列了出来，比如json中batch_size为为128，要想改为256，可以python train.py -c config.json --bs 256
    # 改epoch是我自己加的（方便快速调试），将原来的100改为5
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')  # config.json之外的，添加到args中的参数
    options = [
        # target是该超参数在config.json中的层级结构。如果要
        # CustomArgs(flags=['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        # CustomArgs(flags=['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(flags=['--ep', '--epochs'], type=int, target='trainer;epochs')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)