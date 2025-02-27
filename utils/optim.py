import numpy as np


class StepLR_iter(object):
    def __init__(
        self,
        optimizer,
        decay_every_iter,
        lr_start=0.0001,
        lr_min=0.0,
        warm_up=0,
        decay_rate=1.0,
    ):
        super().__init__()

        self.warm_up = np.maximum(warm_up, 0)
        self.optimizer = optimizer
        self.decay_every_iter = decay_every_iter
        self.decay_rate = decay_rate
        self.lr_start = lr_start
        self.lr_min = lr_min

    def update(self, i_iter):
        # update the learning rate in optimizer
        if (self.warm_up > 0) and (i_iter < self.warm_up):
            lr = (i_iter + 1) / self.warm_up * self.lr_start
            # set learning rate
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        if i_iter >= self.warm_up:
            if (i_iter + 1 - self.warm_up) % self.decay_every_iter == 0:
                lr = self.lr_start * (
                    self.decay_rate
                    ** ((i_iter + 1 - self.warm_up) // self.decay_every_iter)
                )
                lr = np.maximum(lr, self.lr_min)
                # set learning rate
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

    def init(self, i_iter):
        # set the initial learning rate especially for model with pre-trained parameters
        if (self.warm_up > 0) and (i_iter < self.warm_up):
            lr = (i_iter + 1) / self.warm_up * self.lr_start
            # set learning rate
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        if i_iter >= self.warm_up:
            lr = self.lr_start * (
                self.decay_rate
                ** ((i_iter + 1 - self.warm_up) // self.decay_every_iter)
            )
            lr = np.maximum(lr, self.lr_min)
            # set learning rate
            for g in self.optimizer.param_groups:
                g["lr"] = lr


def on_load_checkpoint(checkpoint: dict, complie_mode=False):
    # keys_list = list(checkpoint["state_dict"].keys())
    keys_list = list(checkpoint.keys())
    if complie_mode is not True:
        for key in keys_list:
            if "orig_mod." in key:
                deal_key = key.replace("_orig_mod.", "")
                # checkpoint["state_dict"][deal_key] = checkpoint["state_dict"][key]
                # del checkpoint["state_dict"][key]
                checkpoint[deal_key] = checkpoint[key]
                del checkpoint[key]
    if complie_mode is True:
        for key in keys_list:
            if "orig_mod." not in key:
                deal_key = "_orig_mod." + key
                checkpoint[deal_key] = checkpoint[key]
                del checkpoint[key]
    return checkpoint
