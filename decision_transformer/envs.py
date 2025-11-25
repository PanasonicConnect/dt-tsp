from typing import Optional

import matplotlib.pyplot as plt
import torch
from rl4co.envs import TSPEnv
from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class TSPEnvInit(TSPEnv):
    name = "tsp"

    def __init__(
        self,
        initial_node: int = 0,
        **kwargs,
    ):
        self.initial_node = initial_node
        super().__init__(**kwargs)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        td = super()._reset(td, batch_size)
        td["initial_node"] = (
            torch.tensor(self.initial_node, dtype=torch.int64, device=td.device)
            .unsqueeze(0)
            .expand((td.batch_size))
        )
        td["current_node"] = td["initial_node"]
        td["first_node"] = td["initial_node"]
        td["action_mask"][:, self.initial_node] = False

        return td

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        actions = (
            actions
            if (actions == self.initial_node).any()
            else torch.cat((td["initial_node"].unsqueeze(-1), actions), dim=1)
        )
        return super()._get_reward(td, actions)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are visited exactly once"""
        actions = (
            actions
            if (actions == td["initial_node"][0]).any()
            else torch.cat((td["initial_node"].unsqueeze(-1), actions), dim=1)
        )
        return TSPEnv.check_solution_validity(td, actions)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        if ax is None:
            # Create a plot of the nodes
            _, ax = plt.subplots()

        # plot point
        TSPEnv.render(td, actions, ax)
        # change color
        for obj in ax.get_children():
            if "quiver" in str(obj.__class__):
                obj.set_color("lightblue")
        # add node index
        locs = td["locs"]
        for i, loc in enumerate(locs):
            ax.text(loc[0], loc[1], i, fontsize=8, ha="right", va="bottom")
        # add grid
        ax.grid(True)


class TSPTrajectoryDataset(Dataset):
    def __init__(self, env, dataset_path, context_len=None, node_dim=2, data_ratio=1.0):

        self.context_len = context_len
        self.node_dim = node_dim

        # load dataset
        with open(dataset_path, "r") as f:
            # read file
            lines = f.readlines()
            line = lines[0].split(" ")
            self.num_nodes = (
                line.index("output") // self.node_dim
                if "output" in line
                else len(line) // self.node_dim
            )
            self.context_len = (
                self.num_nodes - 1 if self.context_len is None else self.context_len
            )
            locs, tours = self.encodelines(lines)
            # create dataset
            size = len(lines)
            timesteps = (
                torch.arange(self.num_nodes - 1)
                .unsqueeze(0)
                .expand((size, self.num_nodes - 1))
                .unsqueeze(-1)
            )
            states = tours[:, :-1]
            actions = tours[:, 1:]
            self.trajectories = TensorDict(
                {"initial_node": states[:, 0], "locs": locs},
                batch_size=size,
                device="cpu",
            )
            returns_to_go = (
                env.get_reward(self.trajectories, actions)
                .unsqueeze(-1)
                .expand((size, self.num_nodes - 1))
            )
            selected_masks = torch.ones(
                (size, self.num_nodes - 1, self.num_nodes), dtype=bool
            )
            batch_indexes = torch.arange(size)
            for t in range(self.num_nodes - 1):
                selected_masks[batch_indexes, t, states[:, t]] = False
            if self.num_nodes > 50:
                for i in range(len(selected_masks)):
                    selected_masks[i] = torch.cumprod(selected_masks[i], dim=0).bool()
            else:
                selected_masks = torch.cumprod(selected_masks, dim=1).bool()
            self.trajectories.update(
                {
                    "timesteps": timesteps,
                    "observations": states,
                    "actions": actions,
                    "returns_to_go": returns_to_go,
                    "nodefeatures": locs,
                    "action_mask": selected_masks,
                }
            )
            # cut data
            if data_ratio < 1.0:
                new_size = int(size * data_ratio)
                self.trajectories = self.trajectories[:new_size]

            traj_d = self.trajectories.to_dict()
            self.trajectories = [
                {k: v[i] for k, v in traj_d.items()}
                for i in range(len(self.trajectories))
            ]

    def encodelines(self, lines):
        def enc(line):
            loc = torch.tensor(
                [float(x) for x in line[: self.num_nodes * self.node_dim]]
            ).reshape(-1, self.node_dim)
            tour = torch.tensor(
                [
                    int(x) - 1
                    for x in line[
                        self.num_nodes * self.node_dim
                        + 1 : self.num_nodes * self.node_dim
                        + self.num_nodes
                        + 1
                    ]
                ]
            )
            return loc, tour

        locs, tours = [], []
        for line in lines:
            line = line.split(" ")
            loc, tour = enc(line)
            locs.append(loc)
            tours.append(tour)
        locs = torch.stack(locs, dim=0)
        tours = torch.stack(tours, dim=0)
        return locs, tours

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = traj["observations"]
        actions = traj["actions"]
        returns_to_go = traj["returns_to_go"]
        timesteps = traj["timesteps"]
        # all ones since no padding
        traj_mask = traj["action_mask"]
        # nodefeatures
        nodefeatures = traj["nodefeatures"]

        return timesteps, states, actions, returns_to_go, traj_mask, nodefeatures
