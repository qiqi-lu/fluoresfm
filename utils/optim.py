import numpy as np


class StepLR_iter(object):
    """
    The learning rate is decayed by a factor every decay_every_iter iterations.

    ### Parameters:
        - optimizer (torch.optim.Optimizer): The optimizer to update the learning rate.
        - decay_every_iter (int): The number of iterations to decay the learning rate.
        - lr_start (float): The initial learning rate. Default is 0.0001.
        - lr_min (float): The minimum learning rate. Default is 0.0.
        - warm_up (int): The number of iterations to warm up the learning rate. Default is 0.
        - decay_rate (float): The decay rate. Default is 0.5.
    """

    def __init__(
        self,
        optimizer,
        decay_every_iter,
        lr_start=0.0001,
        lr_min=0.0,
        warm_up=0,
        decay_rate=0.5,
    ):
        super().__init__()

        self.warm_up = np.maximum(warm_up, 0)
        self.optimizer = optimizer
        self.decay_every_iter = decay_every_iter
        self.decay_rate = decay_rate
        self.lr_start = lr_start
        self.lr_min = lr_min

    def update(self, i_iter):
        """
        Update the learning rate in optimizer.

        ### Parameters:
            - i_iter (int): The current iteration.
        """
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
        """
        Initialize the learning rate based on the current iteration.
        Set the initial learning rate especially for model with pre-trained parameters.

        ### Parameters:
            - i_iter (int): The current iteration.
        """
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
    """
    This function is used to load the parameters from the checkpoint into the model.

    The keys of the parameters in the complied model are prefixed with "_orig_mod.",
    and the keys of the parameters in the uncomplied model are not prefixed with "_orig_mod.".
    So they are processed differently when loading the parameters.

    ### Parameters:
        - checkpoint (dict): The checkpoint to load the parameters from.
        - complie_mode (bool): Whether the model is compiled. Default is False.

    ### Returns:
        - checkpoint (dict): The checkpoint with the parameters loaded into the model.
    """
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
