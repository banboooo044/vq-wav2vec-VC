import torch
from torch.utils.data import DataLoader

import hydra
import hydra.utils as utils
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import scipy.io.wavfile as sw
import pyloudnorm
from collections import OrderedDict

from dataset import ParallelWavDataset
from evaluation import mcd, get_log_f0_energy, get_pseudo_f0_energy, ffe, f0d, wer_cer, kl_distance, f0_dist_plot, normalize
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

@hydra.main(config_path="config/convert.yaml")
def convert(cfg):
    fix_seed(cfg.seed)
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    root = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    ds = ParallelWavDataset(root, out_dir, \
            speaker_list=cfg.dataset.speaker_list_path, synthesis_list=cfg.synthesis_list, \
            sr=cfg.preprocessing.sr)

    measures = cfg.measure
    model = VQW2V_RNNDecoder(cfg.enc_checkpoint, **cfg.model)
    checkpoint_dir = utils.to_absolute_path(cfg.checkpoint_dir)
    decoder_path = Path(checkpoint_dir) / cfg.dec_checkpoint
    checkpoint = torch.load(decoder_path, map_location=lambda storage, loc: storage)
    model.load_decoder(checkpoint)

    score= { measure : { 'sum' : 0, 'N' : 0 } for measure in measures }
    line = OrderedDict([ ('source', []), ('source_file', []), ('target', []), ('target_file', [])  ])
    for measure in measures:
        line[measure] = []
        line[measure + '-L'] = []

    if 'wer-cer' in measures:
        line['text'] = []
        line['true_text'] = []

    for source_path, target_path, source_speaker_id, source_audio, target_speaker_id, target_audio, converted_path, word_list, char_list in tqdm(ds):
        # Speech Transformation
        converted_audio = model.convert(source_audio, target_speaker_id)
        meter = pyloudnorm.Meter(cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(source_audio.cpu().detach().numpy())
        output_loudness = meter.integrated_loudness(converted_audio)
        output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
        converted_audio = normalize(converted_audio)
        # save
        sw.write(filename=converted_path, rate=cfg.preprocessing.sr, data=converted_audio)

        # load
        _, converted_audio = sw.read(converted_path)

        dir_name = Path(converted_path).parts[-2]
        [source_name, target_name] = dir_name.split('2')

        # source speaker name
        line['source'].append(source_name)
        # target speaker name
        line['target'].append(target_name)
        # source file name
        line['source_file'].append(source_path.name)
        # file name of target speaker
        line['target_file'].append(target_path.name)

        # evaluation
        if 'mcd' in measures:
            mcd_value, mcd_L  = mcd(target_audio.numpy(), converted_audio, cfg.preprocessing.sr)
            # MCD value
            line['mcd'].append(mcd_value)
            line['mcd-L'].append(mcd_L)

        if 'wer-cer' in measures:
            wer_value, wer_L, cer_value, cer_L, converted_text, true_text = wer_cer(converted_audio, word_list, char_list)
            line['wer'].append(wer_value)
            line['wer-L'].append(wer_L)
            line['cer'].append(cer_value)
            line['cer-L'].append(cer_L)
            line['text'].append(converted_text)
            line['true_text'].append(true_text)

        del source_speaker_id
        del source_audio
        del target_audio
        del target_speaker_id
        del converted_audio

    csv_file_name = out_dir / 'result.csv'
    df = pd.DataFrame.from_dict(line)
    df.to_csv(csv_file_name)

    # stats
    stats = OrderedDict([])
    for measure in measures:
        stats[measure + '-average'] = float((df[measure] * df[measure + '-L']).sum() / df[measure + '-L'].sum())

    csv_file_name = out_dir / 'stats.csv'
    df_stats = pd.DataFrame.from_dict(stats)
    df_stats.to_csv(csv_file_name)

if __name__ == "__main__":
    convert()
