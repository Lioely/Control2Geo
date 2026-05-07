from omegaconf import OmegaConf
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset.mixed_sampler import MixedBatchSampler
from data.dataset.train.hypersim import Hypersim_Dataset
from data.dataset.train.vkitti import VKitti_Dataset


train_dataset = {
    "hypersim": Hypersim_Dataset,
    "vkitti": VKitti_Dataset,
}


class TrainDataLoader(object):
    def __init__(self, args):
        config = OmegaConf.load(args.train_data_config)
        dataset_list = config["dataset_list"]
        dataset_ls = [train_dataset[name](**config[name]) for name in dataset_list]
        concat_dataset = ConcatDataset(dataset_ls)

        generator = torch.Generator()
        generator.manual_seed(getattr(args, "seed", 42))

        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=args.batch_size,
            drop_last=True,
            prob=config["mix_prod"],
            shuffle=True,
            generator=generator,
        )
        self.train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=4,
            pin_memory=True,
        )
