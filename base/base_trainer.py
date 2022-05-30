import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import global_var
from ray import tune
import os
import model.model as module_arch
from utils import prepare_device


class BaseTrainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        """
        @param model: 实例化的torch模型，模型的类写在model.model文件下
        @param criterion: 自己编写损失函数的函数句柄，
        @param metric_ftns: 为list of 函数句柄，这里的函数句柄计算的数据会在tensorboard scalar中显示，函数写在model.metric下
        @param optimizer: torch.optim的诸多优化器类中的一个实例
        @param config: 实例化的parse_config.ConfugParser类
        """
        self.best = {
            "loss": None,
            "accuracy": None,
            "val_loss": None,
            "val_accuracy": None
        }
        self.fold_best = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        self.fold_average = {
            "loss": None,
            "accuracy": None,
            "val_loss": None,
            "val_accuracy": None
        }

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_non_optimum = cfg_trainer['save_non_optimum']

        if self.save_non_optimum:
            # 如果保存非最佳结果，那么根据save_period判断几个epoch保存一次
            assert self.save_period is not None, 'since save_non_optimum is true, save_period should not be None'
        else:
            # 如果不保存非最佳，那么save_period应为none
            assert self.save_period is None, 'since save_non_optimum is false, save_period should be None'

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        写了整个训练过程中的逻辑，其实就是重复运行单步训练逻辑，单步训练逻辑写在trainer.trainer.Trainer类中的方法_train_epoch中
        """
        def single_fold_train(fold_num):
            not_improved_count = 0
            for epoch in range(self.start_epoch, self.epochs + 1):
                result = self._train_epoch(epoch, fold_num)  # 运行单步

                # save logged informations into log dict

                log = {'fold': fold_num,
                       'epoch': epoch}
                log.update(result)

                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))
                    global_var.get_value('email_log').add_log('    {:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning(
                            f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled.")
                        global_var.get_value('email_log').add_log(
                            f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled.")
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info(
                            f"Validation performance didn\'t improve for {self.early_stop} epochs. Training stops.")
                        global_var.get_value('email_log').add_log(
                            f"Validation performance didn\'t improve for {self.early_stop} epochs. Training stops.")

                        global_var.get_value('email_log').print_add(f'fold{fold_num}, best Model:')
                        for k, v in self.best.items():
                            self.fold_best[k].append(v)
                            global_var.get_value('email_log').print_add(f'{k}: {v}')
                        break

                if best:
                    self._save_best_checkpoint(epoch, fold_num)
                    for k, v in result.items():
                        self.best[k] = v

                if self.save_non_optimum:
                    if epoch % self.save_period == 0:
                        self._save_non_optimum_opcheckpoint(epoch, fold_num)

        total_folds = self.config['data_loader']['args']['folds']

        if total_folds == 1:
            global_var.get_value('email_log').print_add.print('不使用k折交叉验证运行'.center(100, '*'))
            single_fold_train(0)

            if self.config['ray_tune']['tune']:
                tune.report(loss=self.best['val_loss'], accuracy=self.best['val_accuracy'])

        else:
            global_var.get_value('email_log').print_add('使用k折交叉验证运行'.center(100, '*'))
            for fold in range(total_folds):
                global_var.get_value('email_log').print_add(f'Fold: {fold}'.center(50, '*'))
                self.data_loader = self.data_loaders[fold]
                self.valid_data_loader = self.valid_data_loaders[fold]
                self.fold_init()

                single_fold_train(fold)


            global_var.get_value('email_log').print_add(f'After {total_folds}-fold, average metric:'.center(100, '*'))
            for k, v in self.fold_best:
                self.fold_average[k] = sum(v) / len(v)
                global_var.get_value('email_log').print_add(f'{k}: v')

            if self.config['ray_tune']['tune']:
                tune.report(loss=self.fold_average['val_loss'], accuracy=self.fold_average['val_accuracy'])


    def fold_init(self):
        # 重新生成新的tensorboard日志文件
        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.config['trainer'].get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.writer = TensorboardWriter(self.config.log_dir, self.logger, self.config['trainer']['tensorboard'])

        self.best = {
            "loss": None,
            "accuracy": None,
            "val_loss": None,
            "val_accuracy": None
        }

        self.model, _ = self.init_model()


    def init_model(self):
        # build model architecture, then print to console
        model = self.config.init_obj('arch', module_arch)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        return model, device

    def _save_non_optimum_opcheckpoint(self, epoch, fold_num):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        filename = f'checkpoint_fold_{fold_num}_epoch_{epoch}.pth'
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)
        self.logger.info(f"Saving checkpoint: {filename} ...")
        global_var.get_value('email_log').add_log(f"Saving checkpoint: {filename} ...")

    def _save_best_checkpoint(self, epoch, fold_num):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        file_name = f'model_best_fold_{fold_num}.pth'
        best_path = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, best_path)
        self.logger.info(f"Saving current best: {file_name} ...")
        global_var.get_value('email_log').add_log(f"Saving current best: {file_name} ...")


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        global_var.get_value('email_log').add_log(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
            global_var.get_value('email_log').add_log("Warning: Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
            global_var.get_value('email_log').add_log("Warning: Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
        global_var.get_value('email_log').add_log(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
