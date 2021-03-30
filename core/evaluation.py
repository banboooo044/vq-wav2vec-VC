import pyworld
import pysptk
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
import re
import scipy.stats as stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from nnmnkwii.metrics import melcd
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import Levenshtein
import json
import warnings
from pathlib import Path
import hydra.utils as utils
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from utils import normalize
sns.set()

warnings.simplefilter('ignore')

d = ModelDownloader()
speech2text_en = Speech2Text(
    **d.download_and_unpack(task="asr", corpus="librispeech")
)

def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

def average_mcd(mc_ref, mc_cv):
    _, path = fastdtw(mc_cv, mc_ref, dist=euclidean)
    twf = np.array(path).T
    cvt_mcc_dtw = mc_cv[twf[0]]
    trg_mcc_dtw = mc_ref[twf[1]]
    # MCD
    diff2sum = np.sum((cvt_mcc_dtw - trg_mcc_dtw)**2, 1)
    mcd_value = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
    return mcd_value, len(cvt_mcc_dtw)

def audio2feature(x, sr, order):
    # Use WORLD vocoder to spectral envelope
    x = normalize(x)
    f0, _time = pyworld.harvest(x.astype(np.float64), sr)
    time_L = len(_time)
    time_delta = [ int((_time[i+1] - _time[i])*sr) for i in range(time_L-1)  ] + [ len(x) -  int(_time[-1]*sr) ]
    f0_expand = np.array([ v for v, delta in zip(f0, time_delta) for _ in range(delta) ])
    f0_expand = np.append(f0_expand, [f0[-1]]* (len(x)-len(f0_expand)) )
    sp = pyworld.cheaptrick(x.astype(np.float64), f0, _time, sr)
    # Extract MCEP features
    alpha = pysptk.util.mcepalpha(sr)
    mc = pysptk.sp2mc(sp, order=order, alpha=alpha)[:, 1:]
    return mc, f0_expand

def mcd(cv, ref, sr=16000, order=24):
    mc_cv, _ = audio2feature(cv, sr, order)
    mc_rf, _ = audio2feature(ref, sr, order)
    mcd_value, mcd_L = average_mcd(mc_rf, mc_cv)
    return mcd_value, mcd_L

def log_f0_mu_std_en(speaker, f0_mu_std_file):
    datasets_path = Path(utils.to_absolute_path("datasets"))
    with open(datasets_path / 'cmu_arctic' / f0_mu_std_file ) as f:
        dic = json.load(f)
    mu, std = dic[speaker]['mu'], dic[speaker]['std']
    return mu, std

def log_f0_mu_std_jp(speaker, f0_mu_std_file ):
    datasets_path = Path(utils.to_absolute_path("datasets"))
    with open(datasets_path / 'jvs' / f0_mu_std_file ) as f:
        dic = json.load(f)
    return dic[speaker]['mu'], dic[speaker]['std']


def get_log_f0_energy(audio, sr=16000):
    _, f0 = audio2feature(audio, sr, order=1)
    eps = 1e-7
    log_f0 = np.log(f0 + eps)
    return log_f0, (f0 > 0)

def get_pseudo_f0_energy(audio, src_speaker, tar_speaker, f0_mu_std_file, sr=16000):
    _, f0 = audio2feature(audio, sr, order=1)
    eps = 1e-7
    src_mu, src_std = log_f0_mu_std_en(src_speaker, f0_mu_std_file)
    tar_mu, tar_std = log_f0_mu_std_en(tar_speaker, f0_mu_std_file)
    pseudo_log_f0 = (np.log(f0 + eps) - src_mu) * tar_std / src_std + tar_mu
    return pseudo_log_f0, (f0 > 0)

def f0d(log_f0_cv, pseudo_log_f0_rf, energy_cv, energy_rf):
    voice_log_f0_cv = log_f0_cv[energy_cv]
    voice_pseudo_log_f0_rf = pseudo_log_f0_rf[energy_rf]
    _, path = fastdtw(voice_log_f0_cv,  voice_pseudo_log_f0_rf, dist=euclidean)
    twf = np.array(path).T
    log_f0_cv_dtw = voice_log_f0_cv[twf[0]]
    pseudo_log_f0_rf_dtw = voice_pseudo_log_f0_rf[twf[1]]
    return np.sqrt(np.sum(np.power(log_f0_cv_dtw - pseudo_log_f0_rf_dtw, 2))) / len(log_f0_cv_dtw), len(log_f0_cv_dtw)

def ffe(log_f0_cv, pseudo_log_f0_rf, energy_cv, energy_rf):
    f0_cv = np.exp(log_f0_cv)
    pseudo_f0_rf = np.exp(pseudo_log_f0_rf)
    _, path = fastdtw(f0_cv, pseudo_f0_rf, dist=euclidean)
    twf = np.array(path).T

    f0_cv_dtw = f0_cv[twf[0]]
    energy_cv_dtw = energy_cv[twf[0]]

    pseudo_f0_rf_dtw = pseudo_f0_rf[twf[1]]
    energy_rf_dtw = energy_rf[twf[1]]

    T = len(f0_cv_dtw)
    VDE = np.sum(energy_cv_dtw != energy_rf_dtw)
    GPE = np.sum((np.abs(f0_cv_dtw -  pseudo_f0_rf_dtw) > (0.2 * pseudo_f0_rf_dtw)) & (energy_cv_dtw == 1) & (energy_rf_dtw == 1))
    GPE_U = np.sum((energy_cv_dtw == 1) & (energy_rf_dtw == 1))
    FFE =  (VDE + GPE)
    return FFE/T, T

def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_distance(cv_f0_list, ref_f0_list, smooth=False, sigma=1):
    cv_hist, ref_hist = (np.histogram(cv_f0_list, density=True)[0],
                    np.histogram(ref_f0_list, density=True)[0])

    if smooth:
        cv_hist, ref_hist = (gaussian_filter(cv_hist, sigma),
                                gaussian_filter(ref_hist, sigma))
    return kl(ref_hist, cv_hist)

def f0_dist_plot(cv_f0_list, speaker, save_file_path):
    pickle_path = Path(utils.to_absolute_path("datasets/cmu_arctic/logf0")) / f'{speaker}.pickle'
    density_cv = stats.gaussian_kde(cv_f0_list)
    with open(pickle_path, 'rb') as f:
        density_ref = pickle.load(f)
    l = min(cv_f0_list) - 0.5
    r = max(cv_f0_list) + 0.5
    x = np.linspace(l, r, 1000)
    plt.figure()
    plt.plot(x, density_cv(x), label='converted')
    plt.plot(x, density_ref(x), label='target')
    plt.legend()
    plt.savefig(save_file_path, dpi=300)

def fix_abbreviation(s):
    s = s.replace('.', '').replace(',', '')
    s = re.sub('[-]+', ' ', s)
    s = s.replace('etc', 'et cetera')
    return s

def encode(a, b):
    word2symbol = {}
    i = 0
    A, B = "", ""
    symbol = [chr(i) for i in range(65, 65+26)] + [chr(i) for i in range(97, 97+26)] + [chr(i) for i in range(12353, 12439)]
    for w in a:
        if not w in word2symbol:
            word2symbol[w] = symbol[i]
            i += 1
        A += word2symbol[w]
    for w in b:
        if not w in word2symbol:
            word2symbol[w] = symbol[i]
            i += 1
        B += word2symbol[w]
    return A, B

def preprocess_text(text):
    return text

def wer(correct, predicted):
    correct, predicted = encode(correct, predicted)
    return Levenshtein.distance(correct, predicted) / len(correct), len(correct)

def cer(correct, predicted):
    correct = "".join(correct)
    predicted = "".join(predicted)
    return Levenshtein.distance(correct, predicted) / len(correct), len(correct)

def wer_cer(x, true_word_list, true_character_list):
    text, token, *_ = speech2text_en(x)[0]
    word_target = text.split()
    character_target = [ w.upper() for w in text.replace(' ', '') ]
    wer_score, wer_L = wer(true_word_list, word_target)
    cer_score, cer_L = cer(true_character_list, character_target)
    return wer_score, wer_L, cer_score, cer_L, text, " ".join(true_word_list)
