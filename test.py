import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from sklearn.metrics import confusion_matrix
import pathlib
import utils
import os
import pandas as pd


def main(config):
    logger = config.get_logger('test')
    root = config['data_loader']['args']['data_dir']

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # autodl训练，另一台机器验证，需要更改以下路径，参考https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 在有gpu的机子训练，无gpu的机子测试，不带这个可能出错

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=map_location)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    all_target = []
    all_predict = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_target += list(target.numpy())
            all_predict += list(torch.argmax(output, dim=1).numpy())

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    cm = confusion_matrix(all_target, all_predict)

    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = cm / row_sums  # 为防止类别图片数量不均衡，计算错误率
    np.fill_diagonal(norm_cm, 0)  # 用0填充对角线，只保留错误

    # 保存混淆矩阵
    json_path = os.path.join(root, 'num2text_label.json')
    if os.path.exists(json_path):
        num2text_label = utils.read_json(json_path)
    else:
        print(f'file {json_path} not exist, test_norm_cm_df.csv will use num label')
        num2text_label = dict([(i, i) for i in range(len(row_sums))])
    text_label = [num2text_label[key] for key in num2text_label]
    norm_cm_df = pd.DataFrame(norm_cm, index=text_label, columns=text_label)
    temp = str(config.resume).split('\\')[-3:-1]
    log_saved_path = os.path.join(config['trainer']['save_dir'], 'log', temp[0], temp[1])
    norm_cm_df.to_csv(os.path.join(log_saved_path, 'test_norm_cm_df.csv'))

    plt.figure()
    plt.matshow(norm_cm, cmap=plt.cm.gray)
    plt.title('norm_confusion_matrix plot')
    plt.xlabel('true label')
    plt.ylabel('predict label')
    plt.savefig(os.path.join(log_saved_path, "norm_confusion_matrix.png"))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # 参数输入例子：python test.py --resume .\saved\models\Mnist_LeNet\0521_135049\model_best.pth
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
