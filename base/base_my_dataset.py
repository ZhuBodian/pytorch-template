import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from base import BaseDataLoader
import os, json
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from PrivateUtils import global_var
"""
用作自建标准的图像文件问件夹结构为：
root/images(包含了所有图片)
root/train.csv（训练集csv文件，内部两列分别是filename与label（数字，连续，从0开始））
root/test.csv
可带：
root/train_short.csv（部分训练集，用于sanity check（图省事可以直接粘测试集））
root/dataset_att.json（包含数据集基本信息，如各类别数目，测试集比例）
root/num2text_label.json（（数字类别，文字类别）的映射）
"""


class BaseMyDataset(Dataset):
    def __init__(self, path, train, transform, split, load_all, save_as_pt, train_val_samples, fold_num):
        super().__init__()

        image_dir = os.path.join(path, "images")
        assert os.path.exists(image_dir), "dir:'{}' not found.".format(image_dir)

        self.target_transform = None
        self.image_folder = image_dir
        self.transform = transform
        self.root = path
        self.load_all_images_to_memories = load_all
        self.data_path = None

        csv_prefix = 'train' if train else 'test'

        if load_all:  # 一次性将所有图片导入内存
            if fold_num == 0:
                global_var.get_value('email_log').print_add(
                    'Running by storing all image tensor to memory'.center(100, '*'))

            if save_as_pt:
                first_run = not os.path.exists(os.path.join(os.path.join(path, csv_prefix + '_data.pt')))
                if first_run:  # 第一次运行
                    self.run_all_image(train, path, split, image_dir, transform, csv_prefix, train_val_samples)

                    torch.save(self.data, os.path.join(path, csv_prefix + '_data.pt'))
                    torch.save(self.targets, os.path.join(path, csv_prefix + '_targets.pt'))
                    torch.save(self.samples, os.path.join(path, csv_prefix + '_samples.pt'))
                    global_var.get_value('email_log').print_add('Saving data to pt......'.center(100, '*'))
                else:
                    self.data = torch.load(os.path.join(path, csv_prefix + '_data.pt'))
                    self.targets = torch.load(os.path.join(path, csv_prefix + '_targets.pt'))
                    self.samples = torch.load(os.path.join(path, csv_prefix + '_samples.pt'))

                    if fold_num == 0:
                        global_var.get_value('email_log').print_add(
                            f'Get tensor from pt; Train size + Val size: {len(self.data)}'.center(100, '*'))
        else:
            if fold_num == 0:
                global_var.get_value('email_log').print_add(
                    'Running by storing batch image tensor to memory'.center(100, '*'))
            self.run_batch_image(train, path, split, image_dir, transform, csv_prefix, train_val_samples)


    def run_all_image(self, train, path, split, image_dir, transform, csv_prefix, train_val_samples):
        # 根据布尔值train，来确定是生成训练集数据（如果是训练集，那么肯定也要生成验证集），还是测试集数据
        csv_name = csv_prefix + '.csv'
        csv_path = os.path.join(path, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found. please run creat_tiny.py firstly".format(csv_path)
        csv_data = pd.read_csv(csv_path)

        csv_image_name_list = list(csv_data['filename'])
        csv_image_label_list = list(set(csv_data['label']))  # set具有唯一性，但是无序
        csv_image_label_list.sort()  # 注意list的sort函数为就地sort

        # 生成self.target
        for idx, image_label in enumerate(csv_image_label_list):
            csv_data.replace(image_label, idx, inplace=True)
        self.targets = torch.from_numpy(np.array(csv_data['label'])).long()

        if train:
            self.samples = train_val_samples # 生成验证集
            csv_train_image_name_list = csv_data['filename'][self.samples[0].indices]
            csv_valid_image_name_list = csv_data['filename'][self.samples[1].indices]
            train_val_size = len(csv_data)
            train_size = len(csv_train_image_name_list)

            global_var.get_value('email_log').print_add(f'Mode: {csv_prefix};'.center(100, '*'))
            #global_var.get_value('email_log').print_add(f'Train dataset size: {len(csv_image_label_list)}(original: {train_size}; additional: {len(csv_image_label_list)*nums-train_size});')

            data = torch.empty((train_val_size, 3, 224, 224))

            # 处理训练集
            print('Process original train set'.center(100, '*'))
            for idx, image_name in enumerate(csv_train_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                temp = transform['train'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）
                data[self.samples[0].indices[idx]] = temp

                if (idx + 1) % 100 == 0:
                    print(f'There are {train_size} original train images. Processing image {idx + 1}')
            # 处理验证集
            print('Process original train set'.center(100, '*'))
            for idx, image_name in enumerate(csv_valid_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                image = transform['val'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）

                data[self.samples[1].indices[idx]] = image

                if (idx + 1) % 100 == 0:
                    print(f'There are {len(csv_valid_image_name_list)} val images. Processing image {idx + 1}')
        else:
            self.samples = None  # 虽然测试集不需要该变量，但是为了保证代码一致，取一个占位符
            global_var.get_value('email_log').print_add(f'Mode: {csv_prefix}; Test datasets: {len(csv_image_name_list)}'.center(100, '*'))
            data = torch.empty((len(csv_image_name_list), 3, 224, 224))
            for idx, image_name in enumerate(csv_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                image = transform['test'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）

                data[idx] = image

        self.data = data
        num_label2text_label = dict([(idx, text) for idx, text in enumerate(csv_image_label_list)])
        # util.write_json(num_label2text_label, os.path.join(path, 'num_label2text_label.json'))


    def run_batch_image(self, train, path, split, image_dir, transform, csv_prefix, train_val_samples):
        # 根据布尔值train，来确定是生成训练集数据（如果是训练集，那么肯定也要生成验证集），还是测试集数据
        csv_name = csv_prefix + '.csv'
        csv_path = os.path.join(path, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found. please run creat_tiny.py firstly".format(csv_path)
        csv_data = pd.read_csv(csv_path)

        csv_image_name_list = list(csv_data['filename'])
        csv_image_label_list = list(set(csv_data['label']))  # set具有唯一性，但是无序
        csv_image_label_list.sort()  # 注意list的sort函数为就地sort

        for idx, image_label in enumerate(csv_image_label_list):
            csv_data.replace(image_label, idx, inplace=True)
        self.targets = torch.from_numpy(np.array(csv_data['label'])).long()

        if train:
            self.samples = train_val_samples
        else:
            self.samples = None

        self.data_path = [os.path.join(image_dir, csv_image_name) for csv_image_name in csv_image_name_list]

        self.data = torch.arange(len(csv_image_name_list))  # 假的self.data，只是为了占位

        num_label2text_label = dict([(idx, text) for idx, text in enumerate(csv_image_label_list)])
        # util.write_json(num_label2text_label, os.path.join(path, 'num_label2text_label.json'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        if self.load_all_images_to_memories:
            return self.data[item, :], self.targets[item]
        else:  # 这里self.data仅是占位符
            return self.data[item], self.targets[item]

    """
        def add_additional_images(self, nums, csv_train_image_name_list, root, transform):
            # 这一部分代码为完成
            train_size = len(csv_train_image_name_list)
            image_dir = os.path.join(root, 'images')
            category_image_nums = utils.read_json(os.path.join(root, 'dataset_att.json'))['category_image_nums']

            additional_data = torch.empty((nums * train_size, 3, 224, 224))  # 用以记录数据增强的额外tenor
            additional_target = torch.empty((nums * train_size))  # 用以记录数据增强的额外tensor的标签
            addidional_idx = 0  # 当前的额外标签

            for idx, image_name in enumerate(csv_train_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                temp = transform['train'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）
                #data[self.samples[0].indices[idx]] = temp

                if (idx + 1) % 100 == 0:
                    print(f'There are {train_size} original images. Processing image {idx + 1}')

            # 添加额外图片
            for _ in range(nums):
                temp = transform['train'](image)  # 由于采用了transforms.RandAugment()，本来就有随机性
                additional_data[addidional_idx, :] = temp
                additional_target[addidional_idx] = self.targets[self.samples[0].indices[idx]]
                addidional_idx = addidional_idx + 1

            # 处理额外图片
            if nums == 0:
                print('Will not add additional pics'.center(100, '*'))
            else:
                print('Process additional train set'.center(100, '*'))
                addidional_idx, additional_data, additional_target = self.add_additional_images(nums,
                                                                                                csv_train_image_name_list,
                                                                                                path, transform)

                temp = np.array([i + train_size for i in range(addidional_idx)], dtype=np.int64)  # 索引只能为int，默认是float
                self.samples[0].indices = np.hstack((self.samples[0].indices, temp))
                data = torch.cat([data, additional_data], dim=0)
                self.targets = torch.cat([self.targets, additional_target], dim=0).long()

            return addidional_idx, additional_data, additional_target
    """