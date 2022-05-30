import argparse
import collections
import os
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils
from utils import send_email
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils import global_var
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def get_k_folds_data_loaders(config, folds):
    csv_path = os.path.join(config['data_loader']['args']['data_dir'], 'train.csv')
    csv_data = pd.read_csv(csv_path)
    targets = np.array(csv_data['label'])

    data_loaders = []
    valid_data_loaders = []
    rate = config['data_loader']['args']['validation_split']

    # 如果test_size不等于1/folds的话，那么不同folds之间，训练集的样本会有重叠
    split = StratifiedShuffleSplit(n_splits=folds, test_size=rate, random_state=42)
    for train_index, valid_index in split.split(np.zeros_like(targets), targets):
        first_data_loader = None if len(data_loaders) == 0 else data_loaders[0]

        train_samples = SubsetRandomSampler(train_index)
        valid_samples = SubsetRandomSampler(valid_index)
        data_loaders.append(config.init_obj('data_loader', module_data,
                                            train_val_samples=[train_samples, valid_samples],
                                            first_data_loader=first_data_loader))
        valid_data_loaders.append(data_loaders[-1].split_validation())

        module_data.TinyImageNetDataloader.fold_num += 1

    return data_loaders, valid_data_loaders


def init_model():
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    return model, device


def main(config):
    global_var._init()
    if not config['ray_tune']['tune']:  # 不进行超参数搜索
        global_var.set_value('email_log', send_email.Mylog(header='pytorch_template', subject='测试log'))
    else:
        subject = str()
        for k, v in config['ray_tune']['args'].items():
            subject = subject + k + ':' + str(v) + ';  '
        global_var.set_value('email_log', send_email.Mylog(header='pytorch_template', subject=subject))

    # config为parse_config中的ConfigParser类，且注意config.init_obj第二个变量均为某个模块，且变量在__name__ == '__main__'中的special variables中
    # 第一个变量为str类型，详细查看config.json文件
    logger = config.get_logger('train')  # 创建指定名字的日志文件，返回的为标准logging类

    # 生成dataloader
    folds = config['data_loader']['args']['folds']
    assert folds >= 1, 'par folds should >= 1'
    if folds == 1:
        # setup data_loader instances，实例化data_loader.data_loaders.MnistDataLoader（config.json中data_loader的type为这个）
        data_loader = config.init_obj('data_loader', module_data)  # 返回一个名为data_loader的，module_data实例
        valid_data_loader = data_loader.split_validation()
        data_loaders, valid_data_loaders = [None, None]
    else:
        data_loaders, valid_data_loaders = get_k_folds_data_loaders(config, folds)
        data_loader = data_loaders[0]
        valid_data_loader = valid_data_loaders[0]

    # 初始化模型
    model, device = init_model()
    logger.info(model)  # logging.info('输出信息')，而类model的输出信息为可训练参数
    global_var.get_value('email_log').add_log(model.__str__())

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
                      combine_data_loaders=[data_loaders, valid_data_loaders])

    trainer.train()

    if not config['ray_tune']['tune']:
        global_var.get_value('email_log').send_mail()
        # os.system('shutdown')


if __name__ == '__main__':
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
