import hydra
from hydra import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from fairseq.models.wav2vec import Wav2VecModel
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
import gc
import apex.amp as amp

from utils import mulaw_decode

class EarlyStopping(object):
    def __init__(self, monitor='loss', direction='min'):
        self.monitor = monitor
        self.direction = direction
        if direction == 'min':
            self.__monitor_values = { self.monitor : float('inf') }
        elif direction == 'max':
            self.__monitor_values = { self.monitor : -float('inf') }
        else:
            raise ValueError("args: [direction] must be min or max")
        self.__best_state = None

    def update(self, values):
        return (self.direction == 'min' and self.__monitor_values[self.monitor] > values[self.monitor]) \
                    or (self.direction == 'max' and self.__monitor_values[self.monitor] < values[self.monitor])

    @property
    def monitor_values(self):
        pass

    @monitor_values.setter
    def monitor_values(self, monitor_values):
        if not isinstance(monitor_values, dict):
            raise ValueError("[monitor_values] must be dict")
        self.__monitor_values = monitor_values

    @monitor_values.getter
    def monitor_values(self):
        return self.__monitor_values

    @property
    def best_state(self):
        pass

    @best_state.setter
    def best_state(self, best_state):
        if not isinstance(best_state, dict):
            raise ValueError("[best_state] must be dict")
        self.__best_state = best_state

    @best_state.getter
    def best_state(self):
        return self.__best_state

def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell

class VQ_Wav2Vec(nn.Module):
    def __init__(self, checkpoint_path):
        super(VQ_Wav2Vec, self).__init__()
        encoder_path = utils.to_absolute_path(checkpoint_path)
        checkpoint = torch.load(encoder_path, map_location=lambda storage, loc: storage)
        self.encoder = Wav2VecModel.build_model(checkpoint['args'], task=None)
        self.encoder.load_state_dict(checkpoint['model'])

    def forward(self, audio):
        self.encoder.train()
        return self.encoder(audio)

    def encode(self, audio):
        self.encoder.eval()
        z = self.encoder.feature_extractor(audio)
        _, idxs = self.encoder.vector_quantizer.forward_idx(z)
        del z
        return idxs

class RnnDecoder(nn.Module):
    def __init__(self, code_book_num, code_embedding_dim, n_speakers, speaker_embedding_dim, rnn_layers_num,
                 conditioning_channels, mu_embedding_dim, rnn_channels,
                 fc_channels, bits, hop_length):
        super(RnnDecoder, self).__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length
        self.rnn_layers_num = rnn_layers_num
        self.code_embedding_1 = nn.Embedding(code_book_num, code_embedding_dim)
        self.code_embedding_2 = nn.Embedding(code_book_num, code_embedding_dim)
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        self.rnn1 = nn.GRU(2*code_embedding_dim + speaker_embedding_dim, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True)
        self.rnn1A = nn.ModuleList([ nn.GRU(2*code_embedding_dim + 2*conditioning_channels, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True) for _ in range(rnn_layers_num) ])

        self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
        self.rnn2 = nn.GRU(mu_embedding_dim + 2*conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, idxs1, idxs2, speakers, audio_size):
        z1 = self.code_embedding_1(idxs1)
        z2 = self.code_embedding_2(idxs2)
        z = torch.cat((z1, z2), dim=2)
        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)
        z = torch.cat((z, speakers), dim=-1)

        # -- Conditioning sub-network
        z, _ = self.rnn1(z)
        ## GRU Block
        for i in range(self.rnn_layers_num):
            z = torch.cat((z, z1, z2), dim=2)
            z, _ = self.rnn1A[i](z)

        z = F.interpolate(z.transpose(1, 2), size=audio_size)
        z = z.transpose(1, 2)

        # -- Autoregressive model
        x = self.mu_embedding(x)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, idxs1, idxs2, speaker, audio_size):
        output = []
        cell = get_gru_cell(self.rnn2)
        z1 = self.code_embedding_1(idxs1)
        z2 = self.code_embedding_2(idxs2)
        z = torch.cat((z1, z2), dim=2)

        speaker = self.speaker_embedding(speaker)
        speaker = speaker.unsqueeze(1).expand(-1, z.size(1), -1)
        z = torch.cat((z, speaker), dim=-1)
        z, _ = self.rnn1(z)

        for i in range(self.rnn_layers_num):
            z = torch.cat((z, z1, z2), dim=2)
            z, _ = self.rnn1A[i](z)

        z = F.interpolate(z.transpose(1, 2), size=audio_size)
        z = z.transpose(1, 2)

        batch_size, sample_size, _ = z.size()
        h = torch.zeros(batch_size, self.rnn_channels, device=z.device)
        x = torch.zeros(batch_size, device=z.device).fill_(self.quantization_channels // 2).long()
        unbind = torch.unbind(z, dim=1)

        for m in tqdm(unbind, leave=False):
            x = self.mu_embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            dist = Categorical(logits=logits)
            x = dist.sample()
            output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)
        return output

class VQW2V_RNNDecoder(nn.Module):
    def __init__(self, enc_checkpoint_path, **model_args):
        super(VQW2V_RNNDecoder, self).__init__()
        self.encoder = VQ_Wav2Vec(enc_checkpoint_path)
        self.decoder = RnnDecoder(**model_args['decoder'])
        self.optimizer = None
        self.schedular = None
        self.step = 0
        self.device = 'cpu'

    def setup(self, optimizer, schedular, device, use_amp=False):
        self.encoder.to(device)
        self.decoder.to(device)
        self.optimizer = optimizer
        self.schedular = schedular
        self.decoder, self.optimizer = amp.initialize(self.decoder, self.optimizer, opt_level="O1")
        self.device = device

    def training_step(self, tr_dl):
        N = 0
        sum_loss = 0
        self.encoder.eval()
        self.decoder.train()
        with tqdm(tr_dl) as pbar:
            for i, (audio, mu_audio, speakers) in enumerate(pbar, 1):
                self.step += 1
                audio, mu_audio, speakers = audio.to(self.device), mu_audio.to(self.device), speakers.to(self.device)
                self.optimizer.zero_grad()
                with torch.no_grad():
                    idxs = self.encoder.encode(audio[:, :-1])
                    idxs1, idxs2 = idxs[:,:,0], idxs[:,:,1]
                output = self.decoder(mu_audio[:, :-1], idxs1, idxs2, speakers, mu_audio.size(1)-1)
                loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])

                pbar.set_description("[train]")
                pbar.set_postfix(OrderedDict(loss=round(loss.item(),2)))

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1)

                self.optimizer.step()

                batch_size = audio.size(0)
                N += batch_size
                sum_loss += loss.item() * batch_size

                del audio
                del mu_audio
                del speakers
                del idxs
                del idxs1
                del idxs2
                del output
                del loss

            gc.collect()
            self.schedular.step()

        tr_score = { 'loss' : sum_loss / N}
        return tr_score

    def validation_step(self, va_dl, score_f):
        self.encoder.eval()
        self.decoder.eval()
        N = 0
        sum_loss = 0
        with torch.no_grad():
            for i, (audio, mu_audio, speakers) in enumerate(tqdm(va_dl), 1):
                audio, mu_audio, speakers = audio.to(self.device), mu_audio.to(self.device), speakers.to(self.device)
                idxs = self.encoder.encode(audio[:, :-1])
                idxs1, idxs2 = idxs[:,:,0], idxs[:,:,1]
                output = self.decoder(mu_audio[:, :-1], idxs1, idxs2, speakers, mu_audio.size(1)-1)
                loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])
                batch_size = audio.size(0)
                N += batch_size
                sum_loss += loss.item() * batch_size
                # audio generate
                del audio
                del mu_audio
                del idxs
                del idxs1
                del idxs2
                del output
        va_score = { 'loss' : sum_loss / N }
        return va_score

    def convert(self, audio, speaker):
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3: #OK
            pass
        else:
            raise AssertionError("[audio] n_dim must be 2 or 3")

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            audio, speaker = audio.to(self.device), speaker.to(self.device)
            idxs = self.encoder.encode(audio)
            idxs1, idxs2 = idxs[:, :, 0], idxs[:, :, 1]
            output = self.vocoder.generate(idxs1, idxs2, speaker, audio.size(0))
        return output


    def checkpoint(self):
        best_checkpoint =  {
            "decoder": deepcopy(self.decoder.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "schedular": deepcopy(self.schedular.state_dict()),
            "step": self.step,
        }
        best_checkpoint['amp'] = deepcopy(amp.state_dict())
        return best_checkpoint

    def save_model(self, checkpoint_dir, state_dict=None):
        checkpoint_dir = Path(checkpoint_dir)
        if state_dict is None:
            checkpoint_path = checkpoint_dir / f"model-{self.step}.pt"
        else:
            checkpoint_path = checkpoint_dir / f"best_model-{state_dict['step']}.pt"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        if state_dict is None:
            state_dict = {
                "decoder": deepcopy(self.decoder.state_dict()),
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "schedular": deepcopy(self.schedular.state_dict()),
                "step": self.step,
            }
        torch.save(state_dict, checkpoint_path)

    def load_model(self, checkpoint):
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.schedular.load_state_dict(checkpoint["schedular"])
        self.step = checkpoint["step"]
        amp.load_state_dict(checkpoint["amp"])

    def load_decoder(self, checkpoint):
        self.decoder.load_state_dict(checkpoint["decoder"])
