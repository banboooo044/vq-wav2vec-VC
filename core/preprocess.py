import hydra
from hydra import utils
from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import scipy.io.wavfile as sw
from pydub import AudioSegment

def mp3_save(x):
    rate, data = sw.read('outputs/full_ppg/bdl2rms/arctic_a0007.wav')
    shifted = data * (2 ** 31 - 1)   # Data ranges from -1.0 to 1.0
    ints = shifted.astype(np.int32)
    sound = AudioSegment(ints.tobytes(),  frame_rate=rate, sample_width=ints.dtype.itemsize, channels=1)
    sound.export("sample.mp3", format="mp3")

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def process_wav(wav_path, out_mulaw_path, out_mel_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):

    sr_, wav = sw.read(wav_path.with_suffix(".wav"))
    assert(sr_ == sr)

    wav = wav / np.abs(wav).max() * 0.999
    wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_mulaw_path, wav)
    return out_mulaw_path, out_mel_path

@hydra.main(config_path="config/preprocessing.yaml")
def preprocess_dataset(cfg):
    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path("datasets")) / str(cfg.dataset.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=cpu_count())

    with open(in_dir / "speakers.json") as file:
        speakers = sorted(json.load(file))

    speakers = [ "awb", "aew", "lnh", "ljm" ]
    for speaker in speakers:
        wav_dir = in_dir / Path(f"cmu_us_{speaker}_arctic") / "wav"
        mulaw_wav_dir = in_dir / Path(f"cmu_us_{speaker}_arctic") / "mulaw_wav"

        mulaw_wav_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting features for {} set".format(speaker))
        wav_list = list(wav_dir.glob('*.wav'))
        futures = []
        for wav_path in wav_list:
            file_name = wav_path.stem
            out_mulaw_path = mulaw_wav_dir / Path(file_name).with_suffix(".wav.npy")
            futures.append(executor.submit(partial(process_wav, wav_path, out_mulaw_path, **cfg.preprocessing)))

        results = [future.result() for future in tqdm(futures)]

if __name__ == "__main__":
    preprocess_dataset()
