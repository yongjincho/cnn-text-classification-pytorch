import argparse
import logging

import torch
from torch import autograd, optim, nn
from tensorboardX import SummaryWriter

import config
import utils
from data import Vocabulary, ClassificationDataset
from model import CnnTextClassifier


def train(args, states=None):
    vocab = Vocabulary(config.vocab_file)
    train_set = ClassificationDataset(config.train_file, vocab, config.train_batch_size)

    model = CnnTextClassifier(len(vocab))
    if torch.cuda.is_available():
        model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir=args.model_dir)

    if states:
        logging.info("Restoring the saved states...")
        epoch = states["epoch"]
        step = states["step"] + 1
        model.load_state_dict(states["model"])
        optimizer.load_state_dict(states["optimizer"])
    else:
        epoch = 0
        step = 0

    is_finished = False
    while (not config.num_epoches or epoch < config.num_epoches) and not is_finished:
        logging.info("==================== Epoch: {} ====================".format(epoch))
        running_losses = []
        for batch in train_set:
            if config.train_steps and step >= config.train_steps:
                is_finished = True
                break

            sequences = autograd.Variable(torch.LongTensor(batch["sequences"]))
            labels = autograd.Variable(torch.LongTensor(batch["labels"]))
            if torch.cuda.is_available():
                sequences, labels = sequences.cuda(), labels.cuda()

            # Predict
            probs, classes = model(sequences)

            # Backpropagation
            optimizer.zero_grad()
            losses = loss_function(probs, labels)
            losses.backward()
            optimizer.step()

            # Log summary
            running_losses.append(losses.data[0])
            if step % args.summary_interval == 0:
                loss = sum(running_losses) / len(running_losses)
                writer.add_scalar("train/loss", loss, step)
                logging.info("step = {}, loss = {}".format(step, loss))
                running_losses = []

            # Save a checkpoint
            if step % args.checkpoint_interval == 0:
                states = {
                    "epoch": epoch,
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                utils.save_checkpoint(args.model_dir, step, states, args.keep_checkpoint_max)

            step += 1

        epoch += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", default=[], nargs="*", help="A list of configuration files.")
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="The directory for a trained model is saved.")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # For train mode
    parser_train = subparsers.add_parser("train", help="train a model")
    parser_train.add_argument("--checkpoint_interval", default=1000, type=int, help="The period at which a checkpoint file will be created.")
    parser_train.add_argument("--keep_checkpoint_max", default=5, type=int, help="The number of checkpoint files to be preserved.")
    parser_train.add_argument("--summary_interval", default=100, type=int, help="The period at which summary will be saved.")

    # For predict mode
    parser_predict = subparsers.add_parser("predict", help="predict by using a trained model")

    args = parser.parse_args()

    if args.subcommand == "train":
        if config._load_from_model_dir(args.model_dir):
            config._load(args.config_files)
            states = utils.load_checkpoint(args.model_dir)
        else:
            config._load(args.config_files)
            config._save(args.model_dir)
            states = None
        train(args, states)

    elif args.subcommand == "predict":
        if not config._load_from_model_dir(args.model_dir):
            raise RuntimeError("Invalid model directory.")
        config._load(args.config_files)  # override
        states = utils.load_checkpoint(args.model_dir)
        if states is None:
            raise RuntimeError("There is no valid checkpoint file.")
        predict(args, states)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s\t%(asctime)s\t%(message)s", level=logging.INFO)
    main()
