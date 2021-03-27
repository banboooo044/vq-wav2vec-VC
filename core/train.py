import hydra
from hydra import utils
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils import trainer
from dataset import WavDatasetCMU
from model import VQW2V_RNNDecoder

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

@hydra.main(config_path="config", config_name="train")
def main(cfg):
    fix_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.dataset == "cmu_arctic":
        tr_ds = WavDatasetCMU(
                root=cfg.dataset.root,
                speaker_list=cfg.dataset.speaker_list_path,
                data_list=cfg.dataset.train_list,
                hop_length=cfg.preprocessing.hop_length,
                sr=cfg.preprocessing.sr,
                sample_frames=cfg.training.sample_frames)

        va_ds = WavDatasetCMU(
                root=cfg.dataset.root,
                speaker_list=cfg.dataset.speaker_list_path,
                data_list=cfg.dataset.val_list,
                hop_length=cfg.preprocessing.hop_length,
                sr=cfg.preprocessing.sr,
                sample_frames=cfg.training.sample_frames)

    tr_dl = DataLoader(tr_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.n_workers,
            pin_memory=True,
            drop_last=True)

    va_dl = DataLoader(va_ds,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.n_workers,
            pin_memory=True,
            drop_last=False)

    model = VQW2V_RNNDecoder(**cfg.model)

    vocoder_params = model.vocoder.parameters()
    optimizer = optim.Adam(
                    vocoder_params,
                    lr=cfg.training.adam.lr,
                    betas=(float(cfg.training.adam.beta_0), float(cfg.training.adam.beta_1))
                )

    schedular = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=cfg.training.multi_step_lr.milestones,
                    gamma=cfg.training.multi_step_lr.gamma
                )

    score_f = {}
    checkpoint_dir = ""
    tensorboard_dir = ""
    trainer(model, optimizer, schedular, score_f, tr_dl, va_dl, checkpoint_dir, tensorboard_dir, device, \
                    early_stop=None, init_epochs=1, max_epochs=1000, validation_interval=2, save_interval=10000)

if __name__ == '__main__':
    main()