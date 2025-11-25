import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, att_mask=None, num_inputs=3):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T
        self.num_inputs = num_inputs

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        if att_mask is not None:
            mask = att_mask
        else:
            ones = torch.ones((max_T, max_T))
            mask = torch.tril(ones).view(1, 1, max_T, max_T)
            # need to mask the return except for the first return entry
            # this is the default practice used by their notebook
            # for every inference, we first estimate the return value for the first return
            # then we estimate the action for at timestamp t
            # it is actually not mentioned in the paper. (ref: ret_sample_fn, single_return_token)
            # mask other ret entries (s, R, a, s, R, a)
            period = num_inputs
            ret_order = 2
            ret_masked_rows = torch.arange(period + ret_order - 1, max_T, period).long()
            # print(ret_masked_rows)
            # print(max_T, ret_masked_rows, mask.shape)
            mask[:, :, :, ret_masked_rows] = 0

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = (
            self.n_heads,
            C // self.n_heads,
        )  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, att_mask=None, num_inputs=3):
        super().__init__()
        self.num_inputs = num_inputs
        self.attention = MaskedCausalAttention(
            h_dim, max_T, n_heads, drop_p, att_mask=att_mask, num_inputs=num_inputs
        )
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 3 * h_dim),
            nn.GELU(),
            nn.Linear(3 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        h_dim,
        n_heads,
        drop_p,
        batch_first=False,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(h_dim, n_heads, batch_first=batch_first)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        att, _ = self.attention(x, x, x)
        x = x + att  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        node_dim,
        n_blocks,
        h_dim,
        context_len,
        n_heads,
        n_enc_layer,
        drop_p,
        max_timestep=4096,
        num_inputs=3,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.h_dim = h_dim
        self.num_inputs = num_inputs

        ### transformer blocks
        input_seq_len = num_inputs * context_len
        blocks = [
            Block(
                h_dim,
                input_seq_len,
                n_heads,
                drop_p,
                num_inputs=num_inputs,
            )
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_reward = torch.nn.Linear(1, h_dim)
        self.embed_nodefeature = torch.nn.Linear(node_dim, h_dim)
        self.encoder = nn.Sequential(
            *(
                [
                    GraphAttentionLayer(h_dim, n_heads, drop_p, batch_first=True)
                    for _ in range(n_enc_layer)
                ]
            )
        )

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)

    def norm(x):
        return x / x.norm(p=2, dim=-1, keepdim=True)

    def indexing(self, input, dim, index):
        return torch.gather(
            input, dim, index.expand(index.size(0), index.size(1), input.size(-1))
        )

    def forward(
        self, timesteps, states, actions, returns_to_go, nodefeatures, selected_masks
    ):

        B, T, _ = states.shape

        returns_to_go = returns_to_go.float()
        time_embeddings = self.embed_timestep(timesteps)
        nodefeature_embeddings = self.embed_nodefeature(nodefeatures)
        nodefeature_embeddings = self.encoder(nodefeature_embeddings)
        # input embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        state_embeddings = (
            self.indexing(nodefeature_embeddings, 1, states) + time_embeddings
        )
        action_embeddings = (
            self.indexing(nodefeature_embeddings, 1, actions) + time_embeddings
        )

        # stack rtg, states and actions and reshape sequence as
        # (s_0, r_0, a_0, s_1, r_1, a_1, ..., s_{T-1}, r_{T-1}, a_{T-1})
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        # predict next rtg with implicit loss
        return_preds = self.predict_rtg(h[:, 0])
        # predict action given s, R
        action_heads = h[:, 1]
        action_pointers = (
            action_heads
            @ nodefeature_embeddings.transpose(1, 2)
            / math.sqrt(action_heads.size(-1))
        )
        action_masked_pointers = action_pointers.masked_fill(
            ~selected_masks, float("-inf")
        )
        action_preds = action_masked_pointers

        return (
            action_preds,
            return_preds,
        )
