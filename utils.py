import glob
import os
import logging

import torch


_checkpoint_prefix = "checkpoint"


def get_checkpoint_filepaths(model_dir):
    filepaths = glob.glob("{}/{}-*".format(model_dir, _checkpoint_prefix))
    filepaths.sort()
    return filepaths


def save_checkpoint(model_dir, step, states, keep_max=None):
    filepath = os.path.join(model_dir, "{}-{:08d}.pt".format(_checkpoint_prefix, step))
    torch.save(states, filepath)

    filepaths = get_checkpoint_filepaths(model_dir)
    if keep_max and len(filepaths) > keep_max:
        for filepath in filepaths[:len(filepaths) - keep_max]:
            os.remove(filepath)


def load_checkpoint(model_dir):
    filepaths = get_checkpoint_filepaths(model_dir)
    if not filepaths:
        return None
    latest_file = filepaths[-1]
    logging.info("Loading the checkpoint file: {}".format(latest_file))
    return torch.load(latest_file)
