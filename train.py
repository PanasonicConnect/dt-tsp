import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rl4co.utils.ops import gather_by_index
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader

from decision_transformer.envs import TSPEnvInit as TSPEnv
from decision_transformer.envs import TSPTrajectoryDataset
from decision_transformer.model import DecisionTransformer as DTModel


def parse():
    parser = argparse.ArgumentParser()
    # dataset and logging
    parser.add_argument(
        "--train_dataset", type=str, default="path/to/train_dataset.txt"
    )
    parser.add_argument(
        "--val_dataset", type=str, default="path/to/validation_dataset.txt"
    )
    parser.add_argument("--log_dir", type=str, default="dt_runs/")
    # model hyperparameters
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_enc_layer", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    # training hyperparameters
    parser.add_argument("--expectile", type=float, default=0.99)
    parser.add_argument("--exp_loss_weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--wt_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--save_iters", type=int, default=1)
    # other settings
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    cfg = OmegaConf.create({k: v for k, v in vars(args).items() if v is not None})
    return cfg


def preprocess(
    timesteps,
    states,
    actions,
    returns_to_go,
    nodefeatures,
    selected_masks,
):
    timesteps_u = timesteps.squeeze(-1)
    states_u = states.unsqueeze(-1)
    next_states_u = gather_by_index(nodefeatures, actions, squeeze=False)
    actions_u = actions.unsqueeze(-1)
    returns_to_go_u = returns_to_go.unsqueeze(-1)
    nodefeatures_u = nodefeatures

    selected_masks_u = selected_masks
    return (
        timesteps_u,
        states_u,
        next_states_u,
        actions_u,
        returns_to_go_u,
        nodefeatures_u,
        selected_masks_u,
    )


def cross_entropy_pointer(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(
        log_probs.permute(0, 2, 1), labels.squeeze(-1).long(), reduction="mean"
    )
    return loss


def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def calc_loss(
    model,
    batch,
    expectile,
    exp_loss_weight,
):
    # extract data from batch
    (
        timesteps,
        states,
        actions,
        returns_to_go,
        selected_masks,
        nodefeatures,
    ) = batch
    action_labels = actions
    # preprocess data
    (timesteps, states, _, actions, returns_to_go, nodefeatures, selected_masks) = (
        preprocess(
            timesteps,
            states,
            actions,
            returns_to_go,
            nodefeatures,
            selected_masks,
        )
    )
    # forward pass
    (
        action_preds,
        imp_return_preds,
    ) = model.forward(
        timesteps, states, actions, returns_to_go, nodefeatures, selected_masks
    )
    # action loss
    action_loss = cross_entropy_pointer(action_preds, action_labels)
    # return expectile loss
    imp_return_pred = imp_return_preds.reshape(-1, 1)
    imp_return_target = returns_to_go.reshape(-1, 1)
    imp_loss = expectile_loss((imp_return_target - imp_return_pred), expectile).mean()
    # merge losses
    merged_loss = action_loss + imp_loss * exp_loss_weight
    # return losses
    return {
        "loss": merged_loss,
        "action_loss": action_loss,
        "imp_loss": imp_loss,
    }


def logging(csv_path, line):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(line)


def train(args):
    # dataset paths
    dataset_path_train = args.train_dataset
    dataset_path_val = args.val_dataset
    # saves model and csv in this directory
    start_time = datetime.now().replace(microsecond=0)
    log_dir = "{}_{}".format(args.log_dir, start_time.strftime("%y%m%d%H%M%S"))
    args.log_dir = log_dir
    # model hyperparameters
    n_blocks = args.n_blocks  # num of transformer blocks
    embed_dim = args.embed_dim  # embedding (hidden) dim of transformer
    context_len = args.context_len  # K in decision transformer
    n_heads = args.n_heads  # num of transformer heads
    n_enc_layer = args.n_enc_layer  # num of graph attention layers
    dropout_p = args.dropout_p  # dropout probability
    # training hyperparameters
    expectile = args.expectile  # expectile value for expectile regression
    exp_loss_weight = args.exp_loss_weight  # weight for expectile loss
    lr = args.lr  # learning rate
    wt_decay = args.wt_decay  # weight decay
    warmup_steps = args.warmup_steps  # warmup steps for lr scheduler
    grad_clip = args.grad_clip  # gradient clipping value
    batch_size = args.batch_size  # training batch size
    num_epochs = args.num_epochs  # num of training epochs
    save_iters = args.save_iters  # save model every save_iters epochs
    # other settings
    device = torch.device(args.device)
    seed = args.seed

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # training device
    device = torch.device(args.device)

    # logging setup
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    iters_digit_cnt = len(str(args.num_epochs))
    save_model_name = "{}.pt".format(str(0).zfill(iters_digit_cnt))
    save_model_path = os.path.join(log_dir, save_model_name)
    log_csv_name = "log.csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)
    log_header = [
        "time_elapsed",
        "num_updates",
        "train_loss",
        "val_loss",
        "train_action_loss",
        "val_action_loss",
        "train_imp_loss",
        "val_imp_loss",
    ]
    logging(log_csv_path, log_header)
    os.makedirs(os.path.join(log_dir, "ckpts"), exist_ok=True)
    OmegaConf.save(args, os.path.join(log_dir, "config.yaml"))

    # output training start
    print("=" * 60)
    print("start training: " + start_time.strftime("%y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # load datasets
    env = TSPEnv(initial_node=0).to(device)
    train_dataset = TSPTrajectoryDataset(env, dataset_path_train)
    val_dataset = TSPTrajectoryDataset(env, dataset_path_val)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    # initialize model
    node_dim = train_dataset.node_dim
    num_actions = context_len
    model = DTModel(
        node_dim=node_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len - 1,
        n_heads=n_heads,
        n_enc_layer=n_enc_layer,
        drop_p=dropout_p,
    ).to(device)
    model = torch.compile(model)

    # initialize optimizer and scaler
    optimizer = AdamWScheduleFree(
        model.parameters(), warmup_steps=warmup_steps, lr=lr, weight_decay=wt_decay
    )
    scaler = torch.amp.GradScaler()

    # output settings
    print(
        "dataset and logging:\n"
        + "\ttrain dataset path: %s\n" % dataset_path_train
        + "\tvalidation dataset path: %s\n" % dataset_path_val
        + "\tlog dir: %s\n" % log_dir
        + "\tlog csv save path: %s\n" % log_csv_path
        + "\tmodel save path: %s\n" % save_model_path
        + "model hyperparameters:\n"
        + "\tnode_dim: %d\n" % node_dim
        + "\tnum_actions: %d\n" % num_actions
        + "\tn_blocks: %d\n" % n_blocks
        + "\th_dim: %d\n" % embed_dim
        + "\tcontext_len: %d\n" % (context_len - 1)
        + "\tn_heads: %d\n" % n_heads
        + "\tdrop_p: %f\n" % dropout_p
        + "other settings:\n"
        + "\tdevice set to: %s\n" % str(device)
        + "\tseed: %d\n" % seed
    )

    # training loop
    best_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        train_losses = {
            "loss": [],
            "action_loss": [],
            "imp_loss": [],
        }
        model.train()
        optimizer.train()

        for batch in train_dataloader:
            batch = [d.to(device) for d in batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # forward pass & calc loss
                loss_dic = calc_loss(
                    model,
                    batch,
                    expectile=expectile,
                    exp_loss_weight=exp_loss_weight,
                )
                # backward pass
                optimizer.zero_grad()
                scaler.scale(loss_dic["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                # logging
                train_losses["loss"].append(loss_dic["loss"].detach().cpu().item())
                train_losses["action_loss"].append(
                    loss_dic["action_loss"].detach().cpu().item()
                )
                train_losses["imp_loss"].append(
                    loss_dic["imp_loss"].detach().cpu().item()
                )

        # merge train losses
        mean_train_loss = {k: np.mean(v) for k, v in train_losses.items()}
        # validation
        val_losses = {
            "loss": [],
            "action_loss": [],
            "imp_loss": [],
        }
        for batch in val_dataloader:
            batch = [d.to(device) for d in batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # forward pass & calc loss
                loss_dic = calc_loss(
                    model,
                    batch,
                    expectile=expectile,
                    exp_loss_weight=exp_loss_weight,
                )
                # logging
                val_losses["loss"].append(loss_dic["loss"].detach().cpu().item())
                val_losses["action_loss"].append(
                    loss_dic["action_loss"].detach().cpu().item()
                )
                val_losses["imp_loss"].append(
                    loss_dic["imp_loss"].detach().cpu().item()
                )
        # merge val losses
        mean_val_loss = {k: np.mean(v) for k, v in val_losses.items()}
        # time elapsed
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        # output log
        log_str = (
            "time elapsed: "
            + time_elapsed
            + ", \t"
            + "num of epochs: "
            + str(epoch)
            + ", \t"
            + "train loss: "
            + format(mean_train_loss["loss"], ".5f")
            + ", \t"
            + "val loss: "
            + format(mean_val_loss["loss"], ".5f")
        )
        print(log_str)
        logging(
            log_csv_path,
            [
                time_elapsed,
                epoch,
                mean_train_loss["loss"],
                mean_val_loss["loss"],
                mean_train_loss["action_loss"],
                mean_val_loss["action_loss"],
                mean_train_loss["imp_loss"],
                mean_val_loss["imp_loss"],
            ],
        )

        if epoch % save_iters == 0:
            filename = os.path.join(log_dir, "ckpts", "{}.pt".format(epoch))
            torch.save(model.state_dict(), filename)

        if mean_val_loss["loss"] < best_loss:
            best_loss = mean_val_loss["loss"]
            filename = os.path.join(log_dir, "ckpts", "best.pt")
            torch.save(model.state_dict(), filename)

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d %H:%M:%S")
    print("started training at: " + start_time.strftime("%y-%m-%d %H:%M:%S"))
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("=" * 60)


if __name__ == "__main__":
    args = parse()
    train(args)
