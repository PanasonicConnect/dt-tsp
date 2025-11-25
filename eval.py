import argparse
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
from tqdm import tqdm

from decision_transformer.envs import TSPEnvInit as TSPEnv
from decision_transformer.envs import TSPTrajectoryDataset
from decision_transformer.model import DecisionTransformer as DTModel
from train import preprocess


def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, default="path/to/test_dataset.txt")
    parser.add_argument("--test_batch_size", type=int, default=2000)
    parser.add_argument("--fixed_rtg", type=float, default=None)
    parser.add_argument("--model_dir", type=str, default="dt_runs/")
    parser.add_argument("--chk_pt_dir", type=str, default=None)
    parser.add_argument("--chk_pt_name", type=str, nargs="+")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--eval_optimal_gap", action="store_true")
    parser.add_argument("--optimal_dataset", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_num", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    return args


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def load_model(model_class, chk_pt_dir, chk_pt_name, dataset, device, params):
    eval_chk_pt_path = os.path.join(chk_pt_dir, chk_pt_name)
    ckpt = torch.load(eval_chk_pt_path, map_location=device)
    in_state_dict = ckpt
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt = out_state_dict
    model = model_class(
        node_dim=dataset.node_dim,
        n_blocks=params["n_blocks"],
        h_dim=params["embed_dim"],
        context_len=params["context_len"] - 1,
        n_heads=params["n_heads"],
        n_enc_layer=params["n_enc_layer"],
        drop_p=params["dropout_p"],
    ).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def predict_rtg(
    model, timesteps, states, actions, returns_to_go, nodefeatures, selected_masks
):
    # 1st pred -> get predicted r_0 from s_0
    (
        _,
        imp_return_preds,
    ) = model(timesteps, states, actions, returns_to_go, nodefeatures, selected_masks)
    # 2nd pred -> get predicted r_t (t>=1) from s_{0:t} and a_{0:t-1}
    (
        _,
        imp_return_preds,
    ) = model(
        timesteps, states, actions, imp_return_preds, nodefeatures, selected_masks
    )
    return imp_return_preds


def predict_action(
    model, timesteps, states, actions, returns_to_go, nodefeatures, selected_masks
):
    # pred -> get proper a_1 - a_t
    (
        action_preds,
        _,
    ) = model(timesteps, states, actions, returns_to_go, nodefeatures, selected_masks)
    action_probs = F.softmax(action_preds, dim=-1)
    _, action_preds = torch.max(action_probs, dim=-1)
    return action_preds


@torch.no_grad()
def rollout(env, td, model, target_return, max_steps: int = None):
    max_steps = float("inf") if max_steps is None else max_steps
    if target_return is None:
        pred_rtg = True
        target_return = -999
    else:
        pred_rtg = False
    # gathered data
    nodefeatures = td["locs"]
    timesteps = torch.empty(
        (td["i"].size(0), 0, td["i"].size(1)), dtype=torch.int64, device=td.device
    )
    target_returns = (
        torch.tensor([target_return], device=td.device)
        .expand(td["reward"].size(0))
        .unsqueeze(-1)
    )
    returns_to_go = torch.empty((td["reward"].size(0), 0), device=td.device)
    states = torch.empty(
        (td["current_node"].size(0), 0), dtype=torch.int64, device=td.device
    )
    selected_masks = torch.empty(
        (td["action_mask"].size(0), 0, td["action_mask"].size(1)),
        dtype=torch.bool,
        device=td.device,
    )
    actions = torch.empty(
        (td["current_node"].size(0), 0), dtype=torch.int64, device=td.device
    )
    actions_padding = torch.zeros(
        (td["current_node"].size(0), 1), dtype=torch.int64, device=td.device
    )
    indices = []

    # simulate
    steps = 0
    while not td["done"].all():
        # get current states
        timesteps = torch.cat((timesteps, td["i"].unsqueeze(1)), dim=1)
        returns_to_go = torch.cat((returns_to_go, target_returns), dim=1)
        states = torch.cat((states, td["current_node"].unsqueeze(1)), dim=1)
        selected_masks = torch.cat(
            (selected_masks, td["action_mask"].unsqueeze(1)), dim=1
        )
        actions = torch.cat((actions, actions_padding), dim=1)
        # preprocess
        (
            timesteps_u,
            states_u,
            next_states_u,
            actions_u,
            rtgs_u,
            nodefeatures_u,
            selected_masks_u,
        ) = preprocess(
            timesteps, states, actions, returns_to_go, nodefeatures, selected_masks
        )
        # get action
        i = 0
        if pred_rtg:
            rtgs_input = predict_rtg(
                model,
                timesteps=timesteps_u[:, i:],
                states=states_u[:, i:],
                actions=actions_u[:, i:],
                returns_to_go=rtgs_u[:, i:],
                nodefeatures=nodefeatures_u,
                selected_masks=selected_masks_u[:, i:],
            )
        else:
            rtgs_input = rtgs_u[:, i:]
        action_preds = predict_action(
            model,
            timesteps=timesteps_u[:, i:],
            states=states_u[:, i:],
            actions=actions_u[:, i:],
            returns_to_go=rtgs_input,
            nodefeatures=nodefeatures_u,
            selected_masks=selected_masks_u[:, i:],
        )[:, -1]
        indices.append(
            torch.ones(timesteps.size(0), dtype=torch.int64, device=td.device) * i
        )
        td.set("action", action_preds)
        actions[:, -1] = action_preds
        # update step
        td = env.step(td)["next"]
        steps += 1
        if steps > max_steps:
            print("Max steps reached")
            break
    reward = env.get_reward(td, actions)
    indices = torch.stack(indices, dim=-1)

    return (
        reward,
        states,
        actions,
        selected_masks_u,
        timesteps,
        nodefeatures,
        indices,
    )


def evaluate(model, env, traj_dataset, batch_size=200, target_return=None):
    pred_rewards = []
    pred_states = []
    pred_actions = []
    best_indices = []
    device = next(model.parameters()).device
    st = time.time()
    for batch_i in tqdm(range(0, traj_dataset.__len__(), batch_size), desc="[batch]"):
        batch_en = min(batch_i + batch_size, traj_dataset.__len__())
        td_val = TensorDict(
            {
                "locs": torch.stack(
                    [
                        traj_dataset.trajectories[i]["locs"]
                        for i in range(batch_i, batch_en)
                    ]
                )
            },
            batch_size=batch_en - batch_i,
        )
        td_val = env.reset(td_val).to(device)
        (
            r,
            s,
            a,
            _,
            _,
            _,
            ind,
        ) = rollout(env, td_val.copy(), model, target_return)
        pred_rewards.append(r)
        pred_states.append(s)
        pred_actions.append(a)
        best_indices.append(ind)
    en = time.time()
    return {
        "reward": torch.cat(pred_rewards).cpu(),
        "state": torch.cat(pred_states).cpu(),
        "action": torch.cat(pred_actions).cpu(),
        "index": torch.cat(best_indices).cpu(),
        "reward_mean": torch.cat(pred_rewards).mean().item(),
        "reward_std": torch.cat(pred_rewards).std().item(),
        "elapsed_time": str(timedelta(seconds=en - st)),
    }


def render(result, env, dataset, result_dir, render_num=None):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print("Start render")
    save_dir = os.path.join(result_dir, "render")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    res_acts = torch.cat(
        [
            torch.zeros([result["action"].size(0), 1]),
            result["action"].to("cpu").detach(),
        ],
        dim=1,
    ).to(int)
    td = TensorDict(
        {
            "locs": torch.stack(
                [dataset.trajectories[i]["locs"] for i in range(dataset.__len__())]
            )
        },
        batch_size=dataset.__len__(),
        device="cpu",
    )
    td = env.reset(td)
    render_num = dataset.__len__() if render_num is None else render_num
    for i in tqdm(range(render_num), desc="[render done]"):
        fig, ax = plt.subplots()
        env.render(td[i], res_acts[i], ax=ax)
        ax.set_title("{}".format(i))
        fig_path = os.path.join(save_dir, "{}.png".format(i))
        fig.savefig(fig_path)
        plt.clf()
        plt.close()


def run(input_args):
    ndigits = 2
    # load args
    test_dataset_path = input_args.test_dataset
    test_batch_size = input_args.test_batch_size
    fixed_rtg = input_args.fixed_rtg
    model_dir = input_args.model_dir
    if input_args.chk_pt_dir is None:
        eval_chk_pt_dir = os.path.join(model_dir, "ckpts")
    else:
        eval_chk_pt_dir = input_args.chk_pt_dir
    if input_args.save_dir is None:
        save_dir = model_dir
    else:
        save_dir = input_args.save_dir
    if input_args.eval_optimal_gap:
        assert (
            input_args.optimal_dataset is not None
        ), "optimal_dataset must be specified when eval_optimal_gap is set"
        optimal_dataset_path = input_args.optimal_dataset
    else:
        optimal_dataset_path = None
    device = torch.device(input_args.device)
    args = OmegaConf.load(os.path.join(model_dir, "config.yaml"))

    # create dataset
    env = TSPEnv(initial_node=0).to(device)
    test_dataset = TSPTrajectoryDataset(env, test_dataset_path)

    model_results = {
        "name": [],
        "cost_mean": [],
        "cost_std": [],
        "gap_mean": [],
        "gap_std": [],
        "elapsed_time": [],
    }
    for eval_chk_pt_name in input_args.chk_pt_name:
        # load model
        model_name = os.path.splitext(eval_chk_pt_name)[0]
        model = load_model(
            DTModel, eval_chk_pt_dir, eval_chk_pt_name, test_dataset, device, args
        )

        # evaluate
        res = evaluate(
            model,
            env,
            test_dataset,
            batch_size=test_batch_size,
            target_return=fixed_rtg,
        )
        pred_costs = res["reward"] * -1.0
        pred_costs_mean = round(pred_costs.mean().item(), ndigits)
        pred_costs_std = round(pred_costs.std().item(), ndigits)
        elapsed_time = res["elapsed_time"]
        model_results["name"].append(model_name)
        model_results["cost_mean"].append(pred_costs_mean)
        model_results["cost_std"].append(pred_costs_std)
        model_results["elapsed_time"].append(elapsed_time)
        print(
            "model: %s\n\telapsed time:\t%s\n\tcost:\t%s +- %s"
            % (model_name, elapsed_time, pred_costs_mean, pred_costs_std)
        )

        # eval optimality gap
        if input_args.eval_optimal_gap:
            optimal_dataset = (
                None
                if optimal_dataset_path is None
                else TSPTrajectoryDataset(env, optimal_dataset_path)
            )
            pred_costs = res["reward"] * -1.0
            optimal_costs = (
                torch.stack(
                    [traj["returns_to_go"][0] for traj in optimal_dataset.trajectories]
                )
                * -1.0
            )
            optimal_gap = (pred_costs - optimal_costs) / optimal_costs * 100.0
            optimal_gap_mean = round(optimal_gap.mean().item(), ndigits)
            optimal_gap_std = round(optimal_gap.std().item(), ndigits)
            model_results["gap_mean"].append(optimal_gap_mean)
            model_results["gap_std"].append(optimal_gap_std)
            print("\toptimality gap:\t%s +- %s" % (optimal_gap_mean, optimal_gap_std))

        # save
        result_dir = os.path.join(save_dir, "result_{}".format(model_name))

        # render
        if input_args.render:
            render(res, env, test_dataset, result_dir, input_args.render_num)

    model_results = pd.DataFrame.from_dict(model_results)
    model_results.to_csv(os.path.join(save_dir, "model_results.csv"), index=False)


if __name__ == "__main__":
    args = eval_parse()
    run(args)
