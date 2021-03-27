import hydra
import hydra.utils as utils
import os
import json
from pathlib import Path
import torch
import numpy as np
import librosa
import scipy.io.wavfile as sw
import pandas as pd
from tqdm import tqdm
import random
import pyloudnorm

from torch.utils.data import DataLoader
from dataset import *
from preprocess import preemphasis
from model_vc import VQW2V_RnnVocoder
from evaluation import mcd, get_log_f0_energy, get_pseudo_f0_energy, ffe, f0d, wer_cer, kl_distance, f0_dist_plot, normalize
import scipy.stats as stats
from collections import OrderedDict

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
    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    #target_dir = Path(utils.to_absolute_path(cfg.target_dir))
    out_dir.mkdir(exist_ok=True, parents=True)
    #target_dir = Path(utils.to_absolute_path(cfg.target_dir))
    ds = ParallelWavDatasetCMU(in_dir, out_dir, \
            speaker_list="speakers.json", synthesis_list=cfg.synthesis_list, \
            sr=cfg.preprocessing.sr)

    dataset_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path

    model = VQW2V_RnnVocoder(cfg)
    model.setup()
    model.load_model()

    score_list = ['mcd',  'wer', 'cer']
    intra_sex = ["bdl2rms", "clb2slt", "bdl2slt", "clb2rms" ]
    extra_sex = ["bdl2aew", "clb2aew", "bdl2lnh", "clb2lnh", "bdl2awb", "clb2awb", "bdl2ljm", "clb2ljm" ]
    score_sum = { score_name : 0 for score_name in score_list }
    L_sum = { score_name : 0 for score_name in score_list }

    score_each = { }
    log_f0_each = { }
    L_each = { }

    source_path_list = [ Path(metadata['source_path']).parts[-3].replace('cmu_us_', '').replace('_arctic', '') + ":" + Path(metadata['source_path']).stem for metadata in ds.metadata ] + ['intra sex', 'extra sex', 'average']
    target_path_list = [ Path(metadata['target_path']).parts[-3].replace('cmu_us_', '').replace('_arctic', '') + ":" + Path(metadata['target_path']).stem for metadata in ds.metadata ] + ['intra sex', 'extra sex', 'average']
    df = OrderedDict([ ('source', source_path_list), ('target', target_path_list) ])
    for score in score_list:
        df[score] = []

    if 'wer' in score_list or 'cer' in score_list:
        df['text'] = []
        df['true_text'] = []

    for source_speaker_id, source_audio, target_speaker_id, target_audio, converted_path, word_list, char_list in tqdm(ds):
        if 'samples_per_sec' in score_list:
            converted_audio, samples_per_sec, sec = model.convert_time(source_audio, target_speaker_id)
        else:
            converted_audio = model.convert(source_audio, target_speaker_id)

        converted_audio = normalize(converted_audio)
        sw.write(filename=converted_path, rate=cfg.preprocessing.sr, data=converted_audio)

        _, converted_audio = sw.read(converted_path)

        dir_name = Path(converted_path).parts[-2]
        [source_name, target_name] = dir_name.split('2')

        if not dir_name in score_each:
            score_each[dir_name] = { score_name : 0 for score_name in score_list }
            L_each[dir_name] = { score_name : 0 for score_name in score_list }
            log_f0_each[dir_name] = { 'cv' : [ ], 'ref' : [] }

        if 'mcd' in score_list:
            mcd_value, mcd_L  = mcd(target_audio.numpy(), converted_audio, cfg.preprocessing.sr)
            df['mcd'].append(mcd_value)
            score_sum['mcd'] += mcd_value * mcd_L
            L_sum['mcd'] += mcd_L
            score_each[dir_name]['mcd'] += mcd_value * mcd_L
            L_each[dir_name]['mcd'] += mcd_L

        if 'ffe' in score_list or 'f0d' in score_list:
            log_f0_cv, energy_cv = get_log_f0_energy(converted_audio)
            log_f0_tar, energy_tar = get_log_f0_energy(target_audio.numpy())
            pseudo_log_f0_rf, energy_rf = get_pseudo_f0_energy(source_audio.numpy(), source_name, target_name, "f0_mu_std.json")

            log_f0_each[dir_name]['cv'].extend(list(log_f0_cv[energy_cv]))
            log_f0_each[dir_name]['ref'].extend(list(log_f0_tar[energy_tar]))

            if 'f0d' in score_list:
                f0d_value, f0d_L = f0d(log_f0_cv, pseudo_log_f0_rf, energy_cv, energy_rf)
                df['f0d'].append(str(f0d_value))
                score_sum['f0d'] += f0d_value * f0d_L
                L_sum['f0d'] += f0d_L
                score_each[dir_name]['f0d'] += f0d_value * f0d_L
                L_each[dir_name]['f0d'] += f0d_L

            if 'ffe' in score_list:
                ffe_value, ffe_L = ffe(log_f0_cv, pseudo_log_f0_rf, energy_cv, energy_rf)
                df['ffe'].append(str(ffe_value))
                score_sum['ffe'] += ffe_value * ffe_L
                L_sum['ffe'] += ffe_L
                score_each[dir_name]['ffe'] += ffe_value * ffe_L
                L_each[dir_name]['ffe'] += ffe_L


        if 'wer' in score_list or 'cer' in score_list:
            wer_value, wer_L, cer_value, cer_L, converted_text, true_text = wer_cer(converted_audio, word_list, char_list)

            if 'wer' in score_list:
                df['wer'].append(wer_value)
                score_sum['wer'] += wer_value * wer_L
                L_sum['wer'] += wer_L
                score_each[dir_name]['wer'] += wer_value * wer_L
                L_each[dir_name]['wer'] += wer_L

            if 'cer' in score_list:
                df['cer'].append(cer_value)
                score_sum['cer'] += cer_value * cer_L
                L_sum['cer'] += cer_L
                score_each[dir_name]['cer'] += cer_value * cer_L
                L_each[dir_name]['cer'] += cer_L


            df['text'].append(converted_text)
            df['true_text'].append(true_text)

        if 'samples_per_sec' in score_list:
            df['samples_per_sec'].append(samples_per_sec)
            score_sum['samples_per_sec'] += samples_per_sec * sec
            L_sum['samples_per_sec'] += sec
            score_each[dir_name]['samples_per_sec'] += samples_per_sec * sec
            L_each[dir_name]['samples_per_sec'] += sec

        del source_speaker_id
        del source_audio
        del target_audio
        del target_speaker_id

    for score_name in score_list:
        S = 0
        L = 0
        for dir_name in intra_sex:
            if not dir_name in score_each:
                continue
            S += score_each[dir_name][score_name]
            L += L_each[dir_name][score_name]
        if L == 0:
            df[score_name].append("")
        else:
            df[score_name].append(S/L)
        S = 0
        L = 0
        for dir_name in extra_sex:
            if not dir_name in score_each:
                continue
            S += score_each[dir_name][score_name]
            L += L_each[dir_name][score_name]
        if L == 0:
            df[score_name].append("")
        else:
            df[score_name].append(S/L)

    if 'wer' in score_list or 'cer' in score_list:
        df['text'].extend(["", ""])
        df['true_text'].extend(["", ""])

    # averageの行
    for score_name in score_list:
        df[score_name].append(score_sum[score_name] / L_sum[score_name])
    if 'wer' in score_list or 'cer' in score_list:
        df['text'].append("")
        df['true_text'].append("")

    for score in score_list:
        print(f"average score of {score} : {df[score][-1]}")

    csv_file_name = out_dir / 'result.csv'
    df = pd.DataFrame.from_dict(df)
    df.to_csv(csv_file_name)

    for dir_name in score_each.keys():
        root = out_dir / Path(dir_name)
        dic = { score_name : [ score_each[dir_name][score_name] / L_each[dir_name][score_name], L_each[dir_name][score_name] ] for score_name in score_list }
        if 'ffe' in score_list or 'f0d' in score_list:
            ref_speaker = dir_name.split("2")[1]
            f0_dist_plot(log_f0_each[dir_name]['cv'], ref_speaker, str(root / f"{dir_name}.png"))

        path = root / "result.csv"
        df_key = pd.DataFrame.from_dict(dic)
        df_key.to_csv(str(path))

if __name__ == "__main__":
    convert()
