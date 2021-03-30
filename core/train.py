import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import hydra
from hydra import utils
from pathlib import Path
import numpy as np
import random

from utils import trainer
from dataset import WavDataset
from model import VQW2V_RNNDecoder, EarlyStopping

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

    if cfg.dataset.dataset == "cmu_arctic":
        root = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
        tr_ds = WavDataset(
                root=root,
                speaker_list=cfg.dataset.speaker_list_path,
                data_list=cfg.train_list,
                sr=cfg.preprocessing.sr,
                sample_frames=cfg.training.sample_frames,
                hop_length=cfg.preprocessing.hop_length,
                bits=cfg.preprocessing.bits,
                )

        va_ds = WavDataset(
                root=root,
                speaker_list=cfg.dataset.speaker_list_path,
                data_list=cfg.val_list,
                sr=cfg.preprocessing.sr,
                sample_frames=cfg.training.sample_frames,
                hop_length=cfg.preprocessing.hop_length,
                bits=cfg.preprocessing.bits)

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

    checkpoint_dir = utils.to_absolute_path(cfg.checkpoint_dir)
    tensorboard_dir = utils.to_absolute_path(cfg.tensorboard_dir)

    model = VQW2V_RNNDecoder(cfg.enc_checkpoint, **cfg.model)
    if cfg.resume:
        resume_path = Path(checkpoint_dir) / cfg.resume
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_model(checkpoint)

    elif cfg.pretrain_checkpoint:
        pretrain_path = Path(checkpoint_dir) / cfg.pretrain_checkpoint
        checkpoint = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
        model.load_decoder(checkpoint)

    vocoder_params = model.decoder.parameters()
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
    early_stop = EarlyStopping(cfg.training.es.monitor_value, cfg.training.es.direction)

    max_epochs = cfg.training.n_steps // len(tr_dl) + 1
    trainer(model, optimizer, schedular, score_f, tr_dl, va_dl, checkpoint_dir, tensorboard_dir, device, \
                    early_stop=early_stop, init_epochs=1, max_epochs=max_epochs, validation_interval=cfg.training.validation_interval, save_interval=cfg.training.save_interval)

if __name__ == '__main__':
    main()
