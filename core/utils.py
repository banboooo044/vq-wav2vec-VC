from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def normalize(wav):
    return wav / np.abs(wav).max() * 0.999

def trainer(model, optimizer, schedular, score_f, tr_dl, va_dl, checkpoint_dir, tensorboard_dir, device, \
                    early_stop=None, init_epochs=1, max_epochs=1000, validation_interval=2, save_interval=10000):
    model.setup(optimizer,schedular, device)
    writer = SummaryWriter(tensorboard_dir)
    try:
        for epoch in range(init_epochs, max_epochs + 1):
            tr_score = model.training_step(tr_dl)

            print("[train] epoch:{}/{}, loss:{:.3E}".format(epoch, max_epochs, tr_score['loss']))
            writer.add_scalar("loss/train", tr_score['loss'], epoch)

            if epoch % validation_interval == 0:
                va_score = model.validation_step(va_dl, score_f)
                for measure, f in score_f.items():
                    writer.add_scalar(f"{measure}/val", va_score[measure], epoch)

                if early_stop.update(va_score):
                    best_checkpoint = model.checkpoint()
                    early_stop.best_state = best_checkpoint

                early_stop.monitor_value = va_score

            if epoch % save_interval == 0:
                model.save_model(checkpoint_dir)

    except KeyboardInterrupt:
        pass

    model.save_model(checkpoint_dir, early_stop.best_state)

def tester(model, te_dl):
    return model.test_step(te_dl)
