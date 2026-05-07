from omegaconf import OmegaConf
from data.dataset.eval.eval_dataset import Eval_Dataset
from torch.utils.data import DataLoader


class EvalDataLoader(object):
    def __init__(self, config_path):
        config = OmegaConf.load(config_path)
        self.dataset_list = config["dataset_list"]
        self.dataset_ls = [Eval_Dataset(**config[name]) for name in self.dataset_list]
        self.dataloader_ls = [DataLoader(dataset, 
                                         batch_size=1, 
                                         shuffle=False,
                                         num_workers=4) 
                              for dataset in self.dataset_ls]
