from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

def trainer(model, optimizer, schedular, tr_dl, va_dl=False, max_epochs=1000, early_stop=True):
    tensorboard_dir = f"tensorboard/{dir_name}"
    writer = SummaryWriter(tensorboard_dir)
    try:
        for epoch in range(start_epoch, n_epoch + 1):
            self.epoch = epoch
            tr_score = model.training_step(optimizer, schedular, tr_dl, va_dl)

            print("[train] epoch:{}/{}, loss:{:.3E}".format(epoch, n_epochs, average_loss))
            if epoch % self.config.training.log_interval == 0 and epoch != self.start_epoch:
                writer.add_scalar("loss/train", average_loss, epoch)

            if :
                va_score = model.validation_step(va_dl)
                if va_score['loss'] < best_va_score:
                    best_va_score = va_score['loss']
                    best_checkpoints_state =  {
                        "vocoder": deepcopy(self.vocoder.state_dict()),
                        "optimizer": deepcopy(self.optimizer.state_dict()),
                        "scheduler": deepcopy(self.scheduler.state_dict()),
                        "step": self.step
                    }
                    if AMP:
                        best_checkpoints_state["amp"] = deepcopy(amp.state_dict())
                writer.add_scalar("loss/val", va_score['loss'], epoch)

                print("[val] epoch:{}/{}, loss:{:.3E}".format(epoch, n_epochs, va_score['loss']))

            if :
                save_model()

        except KeyboardInterrupt:
            pass


def tester(model, te_dl):
    model.test_step()
    return 