from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

def trainer(model, optimizer, schedular, score_f, tr_dl, va_dl, checkpoint_dir, tensorboard_dir, device, \
                    early_stop=None, init_epochs=1, max_epochs=1000, validation_interval=2, save_interval=10000):
    model.setup(device)
    writer = SummaryWriter(tensorboard_dir)
    try:
        for epoch in range(init_epochs, max_epochs + 1):
            tr_score = model.training_step(optimizer, schedular, tr_dl, va_dl)

            # print("[train] epoch:{}/{}, loss:{:.3E}".format(epoch, max_epochs, tr_score['loss']))
            writer.add_scalar("loss/train", tr_score['loss'], epoch)

            if epoch % validation_interval == 0:
                va_score = model.validation_step(va_dl, score_f)
                for measure, f in score_f.items():
                    writer.add_scalar(f"{measure}/val", va_score[measure], epoch)

                if early_stop.stop(va_score):
                    model.store_checkpoint(model, optimizer, schedular)
                    #best_checkpoints_state =  {
                    #    "vocoder": deepcopy(self.vocoder.state_dict()),
                    #    "optimizer": deepcopy(self.optimizer.state_dict()),
                    #    "scheduler": deepcopy(self.scheduler.state_dict()),
                    #    "step": self.step
                    #}
                    #best_checkpoints_state["amp"] = deepcopy(amp.state_dict())
                early_stop.update(va_score)

            if epoch % save_interval:
                model.save_model(checkpoint_dir)

    except KeyboardInterrupt:
        pass

    model.save_best_model(checkpoint_dir)

def tester(model, te_dl):
    return model.test_step(te_dl)