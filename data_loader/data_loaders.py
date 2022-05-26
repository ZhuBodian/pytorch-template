from torchvision import datasets, transforms
from base import BaseDataLoader
from base import base_my_dataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 assign_val_sample=False, augment_pics=0):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         assigned_val=assign_val_sample, samplers=None)


class TinyImageNetDataloader(BaseDataLoader):
    """
    MiniImageNet data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 assign_val_sample=False, load_all_images_to_memories=True, save_as_pt=True):
        # "train"归一化是为了加快收敛速度，但要注意，如果"train"归一化了，但是“test”没归一化，会严重影响测试集准确率
        # （可能是因为归一化会造成色彩失真，从而导致特征发生改变？）
        trsfm = {
            "train": transforms.Compose([transforms.Resize([224, 224]),
                                         # transforms.RandAugment(),  # 这一步在生成数据集那里完成
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize([224, 224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
        self.data_dir = data_dir

        """
        首先明确一点，MNIST数据集由于十分的小，把图片读取为tensorfloat64所占空间为：1*28*28*8*50000/1024/1024/1024=0.3gb（而且处理很快），
        他是把图片全部读取并transom之后全部放到内存中，需要用时再读取。

        对于大一点的数据集，比如3*224*224*8*50000/1024/1024/1024=11.2gb（处理很慢），虽然能完全放到内存中，但是每次重新训练网络，都要再
        读取读取并transform一遍数据集（慢，且重复性工作）。可以考虑将读取并transform一遍后，将全部图片储存为pickle，下次再次重新训练网络时，
        直接读取之前的pickle即可。

        但是对于再大一点的数据集，比如3*224*224*8*50000/1024/1024/1024=56gb（处理也很慢），这种就没办法把把图片全部读取并transom全部放到内存中，
        可以考虑只保存图片索引，计算到某个batch后，仅读取并transform那部分batch的的图片。

        load_all_images_to_memories: 如果为True，若数据集root文件夹中有相应pickle，就读取数据，若没有相应pickle，就从图片层面开始读取，
        读取处理之后，再保存为pickle；如果为False，那么不保存或读取pickle，仅仅在处理一个batch的数据
        
        （一个128*3*224*224的batch，从读取图片到导入gpu计算结束，要2s（但是直接从内存读取到导入gpu计算结束只用0.4s）
            2*10000/128=158，0.4*10000/128=31，还是差很多的）
            
        经过试验后发现，不管load_all_images_to_memories取值是true还是false，每个batch读取的图片顺序并不发生改变，但是如果transform添加
        transforms.RandAugment()方法后，epoch的loss会发生改变（可能是因为虽然train_epoch中的训练顺序不变，但是
        load_all_images_to_memories为true时，是顺序读取图片进行transform，之后再shuffle入batch；
        load_all_images_to_memories为false时，先shuffle入batch，再transform；
        虽然最终shuffle入batch中的顺序一样，但是实际上每个图片的transom顺序不一样，虽然固定了随机数种子，但是仅仅保证了不同次训练的同
        次transforms.RandAugment()的方法一样（即不同次训练的transform一样，但是没transform到同一张图片上））
        总之想要load_all_images_to_memories为true或者false均有同一输出，那就不要用transforms.RandAugment()
        
        虽然保存为pt文件可以节省导入data的时间，但是“保存”的这一步骤会将占用内存近乎翻倍，如果爆内存了，还则将其设置为false，可能还是可以用
        load_all_images_to_memories=True
        """

        self.dataset = base_my_dataset.BaseMyDataset(path=self.data_dir, train=training, transform=trsfm,
                                                     split=validation_split, load_all=load_all_images_to_memories,
                                                     save_as_pt=save_as_pt)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         assigned_val=assign_val_sample, samplers=self.dataset.samples)