"""
Parallel (on-policy) PPO for probe-driven 15-min offset control.

K worker processes each run a CorridorEnv. Per iteration the learner broadcasts the
current policy, the workers collect a total of `--traj-per-update` full episodes
(trajectories) with THAT policy, and the learner does one PPO update (GAE + clipped
objective) on all of them. This fixes the single-process PPO's data-starved updates
(it used ~48 transitions/update); 16 trajectories ≈ 350+ transitions -> a far more
stable gradient.

On-policy correctness: workers don't update between episodes, so all
`traj-per-update` trajectories in an iteration come from the same broadcast policy.
Reuses ActorCritic / gae / ppo_update from train_centralized unchanged; same MDP,
expected-delay difference reward, and GA warm-start as the other trainers.

Run:
  python -m src.rl.train_ppo_parallel --workers 8 --traj-per-update 16 --iters 100
Outputs: outputs/rl/<backbone>/ppo_parallel_ckpt_*.pt and training CSV files.
"""
import os, csv, time, argparse, io
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from . import config as C
from .corridor_env import CorridorEnv
if C.RL_METHOD == "diaz":
    from .diaz_style_env import DiazCorridorEnv as TrainingEnv
elif C.CONTROL_LAYOUT == "all10":
    from .corridor_env_all10 import CorridorEnvAll10 as TrainingEnv
else:
    TrainingEnv = CorridorEnv
from . import demand
from . import scenarios
from .traci_ports import resolve_port_base
from .train_centralized import (
    ActorCritic,
    GLOBAL_LOG_STD_MAX,
    LOG_STD_MIN,
    STATE_LOG_STD_MAX,
    evaluate_policy_action,
    gae,
    ppo_update,
    sample_policy_action,
    OUT,
)

# state layout (from segment_state.build_state):
#   [ int_0(FEAT) | int_1(FEAT) | ... | int_N(FEAT) | tod(2) ]
#   per intersection FEAT = 4 dirs x 7 probe features + 4 approach-mask ctx
N_INT = C.N_AGENTS
FEAT = len(C.DIRECTIONS) * C.N_DIR_FEATURES + C.N_CTX_PER_AGENT
TOD = C.N_TOD


def _policy_action_mask():
    """Mask only the new all10 environment; legacy8 behavior stays reproducible."""
    if C.CONTROL_LAYOUT != "all10" or not C.SPLIT_CONTROL:
        return np.ones(C.ACTION_DIM, dtype=np.float32)
    from .signal_phases import (
        build_policy_action_mask,
        build_split_layout,
        load_signal_phases,
    )
    phases = load_signal_phases(C.SIG_ACTUATED)
    _kmax, layout = build_split_layout(phases, k_max=C.K_MAX_SPLIT)
    return build_policy_action_mask(layout)


ACTION_MASK = _policy_action_mask()
SQUASH_ACTIONS = C.CONTROL_LAYOUT == "all10"
STATE_DEPENDENT_STD = C.CONTROL_LAYOUT == "all10"
# Opt in so historical v3 checkpoints retain their exact parameter structure.
# LayerNorm is applied per intersection because the last tensor dimension is the
# hidden feature dimension, not the corridor/intersection dimension.
USE_SPATIOTEMPORAL_LAYER_NORM = (
    os.environ.get("RL_SPATIOTEMPORAL_LAYER_NORM", "0") == "1")

# Attention and Attention-GRU use direct projections from the per-intersection
# input into a 128-dimensional attention space. Four heads therefore operate on
# 32 features each when --lstm-hid=128. Attention dropout stays at zero so the
# rollout and PPO-update policies remain deterministic conditional on the state.
ATTENTION_HEADS = 4
ATTENTION_DROPOUT = 0.0
ATTENTION_POSITION_ENCODING = "learned_per_intersection_input_v1"
ATTENTION_CRITIC_POOLING = "learned_query_mha_v1"


def _initialize_std_head(head, value=-2.0):
    """Start a state-dependent std head as the previous constant log-std."""
    last = head[-1] if isinstance(head, nn.Sequential) else head
    nn.init.zeros_(last.weight)
    nn.init.constant_(last.bias, value)


# --------------------------- BiLSTM actor-critic ---------------------------
class BiLSTMActorCritic(nn.Module):
    """Treat the selected intersections as a spatial sequence (corridor W->E) and fuse
    their features with a bidirectional LSTM, so each intersection's offset action
    is informed by its upstream AND downstream neighbours (relative-offset
    coordination). Same act()/evaluate() interface as the MLP ActorCritic.

    state (..,STATE_DIM) -> seq (..,N_INT,FEAT) + tod (..,2)
    BiLSTM over the corridor intersections -> (..,N_INT,2H); LayerNorm is
    applied independently to each intersection's 2H-dimensional spatial feature;
    the shared per-intersection actor head then emits one offset action per node."""
    def __init__(self, hid=128, head=64, proj=16, action_mask=ACTION_MASK,
                 squash_actions=SQUASH_ACTIONS,
                 state_dependent_std=STATE_DEPENDENT_STD,
                 use_layer_norm=USE_SPATIOTEMPORAL_LAYER_NORM):
        super().__init__()
        self.lstm = nn.LSTM(FEAT, hid, batch_first=True, bidirectional=True)
        self.spatial_norm = (
            nn.LayerNorm(2 * hid) if use_layer_norm else nn.Identity())
        self.use_layer_norm = bool(use_layer_norm)
        # per-intersection actor head -> N_ACT_PER_INT actions (offset [+ K split])
        self.mu = nn.Sequential(nn.Linear(2 * hid + TOD, head), nn.Tanh(),
                                nn.Linear(head, C.N_ACT_PER_INT))
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Sequential(
                nn.Linear(2 * hid + TOD, head), nn.Tanh(),
                nn.Linear(head, C.N_ACT_PER_INT))
            _initialize_std_head(self.log_std_head)
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
            self.log_std = nn.Parameter(torch.full((C.ACTION_DIM,), -2.0))
        # critic: project each intersection's BiLSTM output 2H->proj (shared), then
        # CONCAT all 8 (keeps per-intersection identity, unlike mean pooling) + tod
        self.v_proj = nn.Linear(2 * hid, proj)
        self.v = nn.Sequential(nn.Linear(N_INT * proj + TOD, head), nn.Tanh(),
                               nn.Linear(head, 1))
        self.register_buffer(
            "action_mask",
            torch.as_tensor(action_mask, dtype=torch.float32).reshape(C.ACTION_DIM),
            persistent=False,
        )
        self.squash_actions = bool(squash_actions)

    def forward(self, s):
        single = (s.dim() == 1)
        if single:
            s = s.unsqueeze(0)
        B = s.shape[0]
        seq = s[:, :N_INT * FEAT].reshape(B, N_INT, FEAT)        # (B,N_INT,FEAT)
        tod = s[:, N_INT * FEAT:]                                # (B,2)
        h, _ = self.lstm(seq)                                    # (B,N_INT,2H)
        h = self.spatial_norm(h)                                  # per intersection
        tod_b = tod.unsqueeze(1).expand(-1, N_INT, -1)           # (B,N_INT,2)
        hcat = torch.cat([h, tod_b], dim=-1)                     # (B,N_INT,2H+2)
        mu = torch.tanh(self.mu(hcat)).reshape(B, -1)            # (B, N*N_ACT)
        mu = mu * self.action_mask
        if self.state_dependent_std:
            log_std = self.log_std_head(hcat).reshape(B, -1).clamp(
                LOG_STD_MIN, STATE_LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                LOG_STD_MIN, GLOBAL_LOG_STD_MAX).expand_as(mu)
        std = log_std.exp()
        vp = torch.tanh(self.v_proj(h)).reshape(B, -1)       # (B,N_INT*proj) concat
        v = self.v(torch.cat([vp, tod], dim=-1)).squeeze(-1)     # (B,)
        if single:
            mu, std, v = mu[0], std[0], v[0]
        return mu, std, v

    def act(self, s):
        mu, std, v = self.forward(s)
        a, logp = sample_policy_action(
            mu, std, self.action_mask, self.squash_actions)
        return a, logp, v

    def evaluate(self, s, a):
        mu, std, v = self.forward(s)
        logp, entropy = evaluate_policy_action(
            mu, std, a, self.action_mask, self.squash_actions)
        return logp, entropy, v


# ----------------------- GRU (deep-recurrent) actor-critic -----------------------
class GRUActorCritic(nn.Module):
    """Deep-recurrent policy for the POMDP. A GRU integrates the history of
    (observation, previous action) into a belief hidden state h_t, which the actor
    and critic read from. This addresses partial observability: at 10% probe
    penetration many per-segment readings are stale/missing within a 15-min step, so
    a single observation is a noisy projection of the true corridor state; carrying
    h_t across the policy-controlled 15-min steps recovers a belief.

    Per step  x_t = [ obs_t (STATE_DIM) | a_{t-1} (ACTION_DIM) ].
    forward() runs on padded (B,T,.) sequences -> whole-episode BPTT (episodes are
    short, so no truncated-BPTT / burn-in needed). step()/act_mean() are the
    single-step rollout carrying the hidden. Warm-start contract unchanged: zero-init
    self.mu -> a=0 -> GA plan regardless of h_t (so compute_baseline still holds)."""
    def __init__(self, obs_dim=C.STATE_DIM, act_dim=C.ACTION_DIM, gru_hid=128,
                 enc=128, action_mask=ACTION_MASK,
                 squash_actions=SQUASH_ACTIONS,
                 state_dependent_std=STATE_DEPENDENT_STD):
        super().__init__()
        self.act_dim = act_dim
        self.enc = nn.Sequential(nn.Linear(obs_dim + act_dim, enc), nn.Tanh())
        self.gru = nn.GRU(enc, gru_hid, batch_first=True)
        self.mu = nn.Linear(gru_hid, act_dim)
        self.v = nn.Linear(gru_hid, 1)
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(gru_hid, act_dim)
            _initialize_std_head(self.log_std_head)
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
            self.log_std = nn.Parameter(torch.full((act_dim,), -2.0))
        self.register_buffer(
            "action_mask",
            torch.as_tensor(action_mask, dtype=torch.float32).reshape(act_dim),
            persistent=False,
        )
        self.squash_actions = bool(squash_actions)

    def forward(self, obs_seq, aprev_seq, h0=None):
        """obs_seq (B,T,obs), aprev_seq (B,T,act) -> mu/std (B,T,act),
        v (B,T), hT. h0=None -> zeros (episode start; matches rollout)."""
        x = self.enc(torch.cat([obs_seq, aprev_seq], dim=-1))   # (B,T,enc)
        out, hT = self.gru(x, h0)                               # (B,T,H)
        mu = torch.tanh(self.mu(out)) * self.action_mask        # (B,T,act) in [-1,1]
        if self.state_dependent_std:
            log_std = self.log_std_head(out).clamp(
                LOG_STD_MIN, STATE_LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                LOG_STD_MIN, GLOBAL_LOG_STD_MAX).expand_as(mu)
        v = self.v(out).squeeze(-1)                             # (B,T)
        return mu, log_std.exp(), v, hT

    def step(self, obs_t, aprev_t, h):
        """Stochastic rollout step: -> sampled a (act,), logp (), v (), h_next."""
        mu, std, v, h = self.forward(obs_t.view(1, 1, -1), aprev_t.view(1, 1, -1), h)
        a, logp = sample_policy_action(
            mu[0, 0], std[0, 0], self.action_mask, self.squash_actions)
        return a, logp, v[0, 0], h

    def act_mean(self, obs_t, aprev_t, h):
        """Deterministic (eval) step: action = mu, no exploration noise."""
        mu, _std, v, h = self.forward(obs_t.view(1, 1, -1), aprev_t.view(1, 1, -1), h)
        return mu[0, 0], v[0, 0], h


# ------------------- BiLSTM (spatial) + GRU (temporal) actor-critic -------------------
class BiLSTMGRUActorCritic(nn.Module):
    """Full spatiotemporal model. Each step a BiLSTM fuses the intersections along
    the corridor (W->E neighbours -> green-wave / relative-offset coupling); then a
    per-intersection GRU (shared weights) carries that spatially-fused feature over
    time as the POMDP belief. So both axes are modeled: SPACE (who's my up/downstream
    neighbour now) and TIME (what have I been seeing / doing).

    Per intersection i, step t:
        sp_{i,t} = BiLSTM_i( [feat_{i,t} | a_{i,t-1}]  over the 8 intersections )
        h_{i,t}  = GRU( sp_{i,t}, h_{i,t-1} )                    # belief
        mu_{i,t} = Head_a( [h_{i,t} | tod_t] )                   # 1 offset
    Critic concatenates the intersection belief projections (+ tod). Warm-start:
    zero-init Head_a's last
    layer -> a=0 -> GA plan. forward() is whole-episode BPTT; step()/act_mean() carry
    the per-intersection hidden (shape (1, B*N_INT, Hg))."""
    def __init__(self, sp_hid=128, gru_hid=128, head=64, proj=16,
                 action_mask=ACTION_MASK, squash_actions=SQUASH_ACTIONS,
                 state_dependent_std=STATE_DEPENDENT_STD,
                 use_layer_norm=USE_SPATIOTEMPORAL_LAYER_NORM):
        super().__init__()
        self.lstm = nn.LSTM(FEAT + C.N_ACT_PER_INT, sp_hid, batch_first=True,
                            bidirectional=True)
        self.spatial_norm = (
            nn.LayerNorm(2 * sp_hid) if use_layer_norm else nn.Identity())
        self.gru = nn.GRU(2 * sp_hid, gru_hid, batch_first=True)
        self.belief_norm = (
            nn.LayerNorm(gru_hid) if use_layer_norm else nn.Identity())
        self.use_layer_norm = bool(use_layer_norm)
        self.mu = nn.Sequential(nn.Linear(gru_hid + TOD, head), nn.Tanh(),
                                nn.Linear(head, C.N_ACT_PER_INT))
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Sequential(
                nn.Linear(gru_hid + TOD, head), nn.Tanh(),
                nn.Linear(head, C.N_ACT_PER_INT))
            _initialize_std_head(self.log_std_head)
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
            self.log_std = nn.Parameter(torch.full((C.ACTION_DIM,), -2.0))
        self.v_proj = nn.Linear(gru_hid, proj)
        self.v = nn.Sequential(nn.Linear(N_INT * proj + TOD, head), nn.Tanh(),
                               nn.Linear(head, 1))
        self.register_buffer(
            "action_mask",
            torch.as_tensor(action_mask, dtype=torch.float32).reshape(C.ACTION_DIM),
            persistent=False,
        )
        self.squash_actions = bool(squash_actions)

    def forward(self, obs_seq, aprev_seq, h0=None):
        """obs_seq (B,T,STATE_DIM), aprev_seq (B,T,ACTION_DIM) -> mu, std, value,
        hT. h0=None -> zeros (episode start)."""
        B, T, _ = obs_seq.shape
        xint = obs_seq[..., :N_INT * FEAT].reshape(B, T, N_INT, FEAT)
        tod = obs_seq[..., N_INT * FEAT:]                              # (B,T,2)
        ap = aprev_seq.reshape(B, T, N_INT, -1)                  # (B,T,N_INT,N_ACT)
        xin = torch.cat([xint, ap], dim=-1)               # (B,T,N_INT,FEAT+N_ACT)
        # spatial BiLSTM over the corridor intersections, per (B*T)
        sp, _ = self.lstm(xin.reshape(B * T, N_INT, -1))        # (B*T,N_INT,2Hs)
        sp = self.spatial_norm(sp)                               # per intersection
        sp = sp.reshape(B, T, N_INT, -1)                        # (B,T,N_INT,2Hs)
        # temporal GRU per intersection: intersections are the batch dim
        g_in = sp.permute(0, 2, 1, 3).reshape(B * N_INT, T, -1)
        g_out, hT = self.gru(g_in, h0)                          # (B*N_INT,T,Hg)
        g_out = g_out.reshape(B, N_INT, T, -1).permute(0, 2, 1, 3)
        belief = self.belief_norm(g_out)                         # per intersection
        tod_b = tod.unsqueeze(2).expand(-1, -1, N_INT, -1)      # (B,T,N_INT,2)
        head_input = torch.cat([belief, tod_b], -1)
        mu = torch.tanh(self.mu(head_input))
        mu = mu.reshape(B, T, -1) * self.action_mask                  # (B,T,N*N_ACT)
        if self.state_dependent_std:
            log_std = self.log_std_head(head_input).reshape(B, T, -1)
            log_std = log_std.clamp(LOG_STD_MIN, STATE_LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                LOG_STD_MIN, GLOBAL_LOG_STD_MAX).expand_as(mu)
        vp = torch.tanh(self.v_proj(belief)).reshape(B, T, -1)        # (B,T,N*proj)
        v = self.v(torch.cat([vp, tod], -1)).squeeze(-1)             # (B,T)
        return mu, log_std.exp(), v, hT

    def step(self, obs_t, aprev_t, h):
        mu, std, v, h = self.forward(obs_t.view(1, 1, -1), aprev_t.view(1, 1, -1), h)
        a, logp = sample_policy_action(
            mu[0, 0], std[0, 0], self.action_mask, self.squash_actions)
        return a, logp, v[0, 0], h

    def act_mean(self, obs_t, aprev_t, h):
        mu, _std, v, h = self.forward(obs_t.view(1, 1, -1), aprev_t.view(1, 1, -1), h)
        return mu[0, 0], v[0, 0], h


# ------------------- self-attention (spatial) + GRU (temporal) -------------------
class ProjectedResidualSelfAttention(nn.Module):
    """Self-attention without a separate feature-embedding layer.

    The raw per-intersection input is projected directly to Q/K/V.  The total
    attention dimension may therefore differ from the input dimension (33 for
    the current all10 offset-only policy).  A learned per-intersection positional
    term preserves node identity, while a matching linear projection supplies
    the residual path required for the 33 -> 128 dimensional change.
    """

    def __init__(self, input_dim, attention_dim=128, num_heads=ATTENTION_HEADS,
                 n_nodes=N_INT):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError(
                f"attention_dim={attention_dim} must be divisible by "
                f"num_heads={num_heads}")
        self.input_dim = int(input_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.n_nodes = int(n_nodes)

        self.position = nn.Parameter(
            torch.zeros(1, self.n_nodes, self.input_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.q_proj = nn.Linear(self.input_dim, self.attention_dim)
        self.k_proj = nn.Linear(self.input_dim, self.attention_dim)
        self.v_proj = nn.Linear(self.input_dim, self.attention_dim)
        self.out_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.residual_proj = nn.Linear(self.input_dim, self.attention_dim)
        self.norm = nn.LayerNorm(self.attention_dim)

    def forward(self, x, return_weights=False):
        """Encode ``(..., N_INT, input_dim)`` and optionally return head weights."""
        if x.shape[-2:] != (self.n_nodes, self.input_dim):
            raise ValueError(
                f"expected (..., {self.n_nodes}, {self.input_dim}), got "
                f"{tuple(x.shape)}")
        leading = x.shape[:-2]
        x = x.reshape(-1, self.n_nodes, self.input_dim) + self.position
        batch = x.shape[0]

        def split_heads(projection):
            return projection.reshape(
                batch, self.n_nodes, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        value = split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous()
        context = context.reshape(batch, self.n_nodes, self.attention_dim)
        encoded = self.norm(self.residual_proj(x) + self.out_proj(context))
        encoded = encoded.reshape(*leading, self.n_nodes, self.attention_dim)
        if return_weights:
            weights = weights.reshape(
                *leading, self.num_heads, self.n_nodes, self.n_nodes)
            return encoded, weights
        return encoded


class AttentionActorCritic(nn.Module):
    """Memoryless spatial-attention actor-critic.

    The current per-intersection observations are projected directly into four
    residual self-attention heads. The actor emits one action per intersection;
    a time-of-day-aware learned query pools the node representations for the
    centralized critic. This is the spatial-only counterpart of Attention-GRU.
    """

    def __init__(self, attention_dim=128, head=64,
                 num_heads=ATTENTION_HEADS, action_mask=ACTION_MASK,
                 squash_actions=SQUASH_ACTIONS,
                 state_dependent_std=STATE_DEPENDENT_STD):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError(
                f"attention_dim={attention_dim} must be divisible by "
                f"num_heads={num_heads}")
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.spatial_attention = ProjectedResidualSelfAttention(
            FEAT, attention_dim=self.attention_dim,
            num_heads=self.num_heads, n_nodes=N_INT)

        self.mu = nn.Sequential(
            nn.Linear(self.attention_dim + TOD, head), nn.Tanh(),
            nn.Linear(head, C.N_ACT_PER_INT))
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Sequential(
                nn.Linear(self.attention_dim + TOD, head), nn.Tanh(),
                nn.Linear(head, C.N_ACT_PER_INT))
            _initialize_std_head(self.log_std_head)
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
            self.log_std = nn.Parameter(torch.full((C.ACTION_DIM,), -2.0))

        self.critic_query = nn.Parameter(
            torch.zeros(1, 1, self.attention_dim))
        nn.init.normal_(self.critic_query, mean=0.0, std=0.02)
        self.critic_tod_proj = nn.Linear(TOD, self.attention_dim, bias=False)
        self.critic_pool = nn.MultiheadAttention(
            embed_dim=self.attention_dim, num_heads=self.num_heads,
            dropout=ATTENTION_DROPOUT, batch_first=True)
        self.critic_pool_norm = nn.LayerNorm(self.attention_dim)
        self.v = nn.Sequential(
            nn.Linear(self.attention_dim + TOD, head), nn.Tanh(),
            nn.Linear(head, 1))

        self.register_buffer(
            "action_mask",
            torch.as_tensor(action_mask, dtype=torch.float32).reshape(
                C.ACTION_DIM),
            persistent=False,
        )
        self.squash_actions = bool(squash_actions)

    def forward(self, state):
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        nodes = state[:, :N_INT * FEAT].reshape(batch, N_INT, FEAT)
        tod = state[:, N_INT * FEAT:]
        spatial = self.spatial_attention(nodes)

        tod_nodes = tod.unsqueeze(1).expand(-1, N_INT, -1)
        actor_input = torch.cat([spatial, tod_nodes], dim=-1)
        mu = torch.tanh(self.mu(actor_input)).reshape(batch, -1)
        mu = mu * self.action_mask
        if self.state_dependent_std:
            log_std = self.log_std_head(actor_input).reshape(batch, -1)
            log_std = log_std.clamp(LOG_STD_MIN, STATE_LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                LOG_STD_MIN, GLOBAL_LOG_STD_MAX).expand_as(mu)

        query = self.critic_query.expand(batch, -1, -1)
        query = query + self.critic_tod_proj(tod).unsqueeze(1)
        pooled, _weights = self.critic_pool(
            query, spatial, spatial, need_weights=False)
        corridor = self.critic_pool_norm(query + pooled).squeeze(1)
        value = self.v(torch.cat([corridor, tod], dim=-1)).squeeze(-1)
        if single:
            mu, log_std, value = mu[0], log_std[0], value[0]
        return mu, log_std.exp(), value

    def act(self, state):
        mu, std, value = self.forward(state)
        action, logp = sample_policy_action(
            mu, std, self.action_mask, self.squash_actions)
        return action, logp, value

    def evaluate(self, state, action):
        mu, std, value = self.forward(state)
        logp, entropy = evaluate_policy_action(
            mu, std, action, self.action_mask, self.squash_actions)
        return logp, entropy, value


class AttentionGRUActorCritic(nn.Module):
    """Spatial self-attention followed by an intersection-level temporal GRU.

    Per intersection, the current 32-dimensional probe observation and previous
    local action are projected directly into four attention heads.  A shared GRU
    then maintains one temporal belief per intersection.  The actor emits one
    residual offset per node.  The centralized critic uses a learned, TOD-aware
    query to pool all intersection beliefs without first compressing each belief
    to a fixed 16-dimensional vector.
    """

    def __init__(self, attention_dim=128, gru_hid=128, head=64,
                 num_heads=ATTENTION_HEADS, action_mask=ACTION_MASK,
                 squash_actions=SQUASH_ACTIONS,
                 state_dependent_std=STATE_DEPENDENT_STD,
                 use_layer_norm=USE_SPATIOTEMPORAL_LAYER_NORM):
        super().__init__()
        if gru_hid % num_heads:
            raise ValueError(
                f"gru_hid={gru_hid} must be divisible by num_heads={num_heads} "
                "for critic attention pooling")
        self.input_dim = FEAT + C.N_ACT_PER_INT
        self.attention_dim = int(attention_dim)
        self.gru_hid = int(gru_hid)
        self.num_heads = int(num_heads)
        self.spatial_attention = ProjectedResidualSelfAttention(
            self.input_dim, attention_dim=self.attention_dim,
            num_heads=self.num_heads, n_nodes=N_INT)
        self.gru = nn.GRU(self.attention_dim, self.gru_hid, batch_first=True)
        self.belief_norm = (
            nn.LayerNorm(self.gru_hid) if use_layer_norm else nn.Identity())
        self.use_layer_norm = bool(use_layer_norm)

        self.mu = nn.Sequential(
            nn.Linear(self.gru_hid + TOD, head), nn.Tanh(),
            nn.Linear(head, C.N_ACT_PER_INT))
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Sequential(
                nn.Linear(self.gru_hid + TOD, head), nn.Tanh(),
                nn.Linear(head, C.N_ACT_PER_INT))
            _initialize_std_head(self.log_std_head)
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
            self.log_std = nn.Parameter(torch.full((C.ACTION_DIM,), -2.0))

        # A single learned query asks which intersection beliefs matter for the
        # corridor-level return.  TOD shifts that query across the daily profile.
        self.critic_query = nn.Parameter(torch.zeros(1, 1, self.gru_hid))
        nn.init.normal_(self.critic_query, mean=0.0, std=0.02)
        self.critic_tod_proj = nn.Linear(TOD, self.gru_hid, bias=False)
        self.critic_pool = nn.MultiheadAttention(
            embed_dim=self.gru_hid, num_heads=self.num_heads,
            dropout=ATTENTION_DROPOUT, batch_first=True)
        self.critic_pool_norm = nn.LayerNorm(self.gru_hid)
        self.v = nn.Sequential(
            nn.Linear(self.gru_hid + TOD, head), nn.Tanh(),
            nn.Linear(head, 1))

        self.register_buffer(
            "action_mask",
            torch.as_tensor(action_mask, dtype=torch.float32).reshape(C.ACTION_DIM),
            persistent=False,
        )
        self.squash_actions = bool(squash_actions)

    def forward(self, obs_seq, aprev_seq, h0=None):
        """Run whole-episode spatial attention and per-node temporal BPTT."""
        B, T, _ = obs_seq.shape
        xint = obs_seq[..., :N_INT * FEAT].reshape(B, T, N_INT, FEAT)
        tod = obs_seq[..., N_INT * FEAT:]
        aprev = aprev_seq.reshape(B, T, N_INT, C.N_ACT_PER_INT)
        xin = torch.cat([xint, aprev], dim=-1)

        spatial = self.spatial_attention(
            xin.reshape(B * T, N_INT, self.input_dim))
        spatial = spatial.reshape(B, T, N_INT, self.attention_dim)
        temporal_input = spatial.permute(0, 2, 1, 3).reshape(
            B * N_INT, T, self.attention_dim)
        temporal, hT = self.gru(temporal_input, h0)
        temporal = temporal.reshape(B, N_INT, T, self.gru_hid).permute(
            0, 2, 1, 3)
        belief = self.belief_norm(temporal)

        tod_nodes = tod.unsqueeze(2).expand(-1, -1, N_INT, -1)
        actor_input = torch.cat([belief, tod_nodes], dim=-1)
        mu = torch.tanh(self.mu(actor_input)).reshape(B, T, -1)
        mu = mu * self.action_mask
        if self.state_dependent_std:
            log_std = self.log_std_head(actor_input).reshape(B, T, -1)
            log_std = log_std.clamp(LOG_STD_MIN, STATE_LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                LOG_STD_MIN, GLOBAL_LOG_STD_MAX).expand_as(mu)

        beliefs_flat = belief.reshape(B * T, N_INT, self.gru_hid)
        tod_flat = tod.reshape(B * T, TOD)
        query = self.critic_query.expand(B * T, -1, -1)
        query = query + self.critic_tod_proj(tod_flat).unsqueeze(1)
        pooled, _weights = self.critic_pool(
            query, beliefs_flat, beliefs_flat, need_weights=False)
        corridor_belief = self.critic_pool_norm(query + pooled).squeeze(1)
        value = self.v(torch.cat([corridor_belief, tod_flat], dim=-1))
        value = value.squeeze(-1).reshape(B, T)
        return mu, log_std.exp(), value, hT

    def step(self, obs_t, aprev_t, h):
        mu, std, value, h = self.forward(
            obs_t.view(1, 1, -1), aprev_t.view(1, 1, -1), h)
        action, logp = sample_policy_action(
            mu[0, 0], std[0, 0], self.action_mask, self.squash_actions)
        return action, logp, value[0, 0], h

    def act_mean(self, obs_t, aprev_t, h):
        mu, _std, value, h = self.forward(
            obs_t.view(1, 1, -1), aprev_t.view(1, 1, -1), h)
        return mu[0, 0], value[0, 0], h


def build_net(arch, lstm_hid):
    if arch == "bilstm":
        return BiLSTMActorCritic(
            hid=lstm_hid, action_mask=ACTION_MASK,
            squash_actions=SQUASH_ACTIONS,
            state_dependent_std=STATE_DEPENDENT_STD,
            use_layer_norm=USE_SPATIOTEMPORAL_LAYER_NORM)
    if arch == "gru":
        return GRUActorCritic(
            gru_hid=lstm_hid, action_mask=ACTION_MASK,
            squash_actions=SQUASH_ACTIONS,
            state_dependent_std=STATE_DEPENDENT_STD)
    if arch == "bilstm_gru":
        return BiLSTMGRUActorCritic(
            sp_hid=lstm_hid, gru_hid=lstm_hid, action_mask=ACTION_MASK,
            squash_actions=SQUASH_ACTIONS,
            state_dependent_std=STATE_DEPENDENT_STD,
            use_layer_norm=USE_SPATIOTEMPORAL_LAYER_NORM)
    if arch == "attention":
        return AttentionActorCritic(
            attention_dim=lstm_hid, num_heads=ATTENTION_HEADS,
            action_mask=ACTION_MASK, squash_actions=SQUASH_ACTIONS,
            state_dependent_std=STATE_DEPENDENT_STD)
    if arch == "attention_gru":
        return AttentionGRUActorCritic(
            attention_dim=lstm_hid, gru_hid=lstm_hid,
            num_heads=ATTENTION_HEADS, action_mask=ACTION_MASK,
            squash_actions=SQUASH_ACTIONS,
            state_dependent_std=STATE_DEPENDENT_STD,
            use_layer_norm=USE_SPATIOTEMPORAL_LAYER_NORM)
    return ActorCritic(
        C.STATE_DIM, C.ACTION_DIM, action_mask=ACTION_MASK,
        squash_actions=SQUASH_ACTIONS,
        state_dependent_std=STATE_DEPENDENT_STD)


# archs whose policy is recurrent (need sequence rollout + per-episode BPTT update)
RECURRENT_ARCHS = ("gru", "bilstm_gru", "attention_gru")


def warmstart_net(net, arch, log_std_init=-2.5, warmstart_mean=True):
    """Initialize the mean and exploration scale independently.

    State-dependent std heads start as a constant function, then learn variation
    from the policy representation. log_std_init=-2.5 gives about 2.5 s of offset
    jitter near the center of the tanh transform for a +/-30 s residual range.
    """
    with torch.no_grad():
        if warmstart_mean:
            if arch in ("bilstm", "bilstm_gru", "attention",
                        "attention_gru"):
                nn.init.zeros_(net.mu[-1].weight)
                nn.init.zeros_(net.mu[-1].bias)
            else:
                nn.init.zeros_(net.mu.weight)
                nn.init.zeros_(net.mu.bias)
        if net.state_dependent_std:
            _initialize_std_head(net.log_std_head, log_std_init)
        else:
            net.log_std.fill_(log_std_init)


def ppo_update_recurrent(net, opt, episodes, epochs=10, clip=0.2, ent_coef=0.01,
                         max_grad_norm=0.5):
    """PPO update for the recurrent policy. `episodes` is a list of per-episode dicts
    with ordered arrays obs (T,obs), act (T,act), old_lp (T,), adv (T,), ret (T,).

    Sequences (NOT shuffled transitions): each episode is padded to Tmax, the GRU is
    rolled forward from h0=0 over the whole sequence (exact hidden reconstruction, no
    burn-in), and the clipped PPO loss is averaged over the valid (unpadded) steps.
    aprev_seq is reconstructed from act (a_{-1}=0), matching what the rollout fed."""
    E = len(episodes)
    Tmax = max(ep["obs"].shape[0] for ep in episodes)
    od, ad = C.STATE_DIM, C.ACTION_DIM
    obs = torch.zeros(E, Tmax, od); aprev = torch.zeros(E, Tmax, ad)
    act = torch.zeros(E, Tmax, ad); oldlp = torch.zeros(E, Tmax)
    adv = torch.zeros(E, Tmax); ret = torch.zeros(E, Tmax); mask = torch.zeros(E, Tmax)
    for i, ep in enumerate(episodes):
        T = ep["obs"].shape[0]
        a = torch.as_tensor(ep["act"], dtype=torch.float32)
        obs[i, :T] = torch.as_tensor(ep["obs"], dtype=torch.float32)
        act[i, :T] = a
        aprev[i, 1:T] = a[:T - 1]                       # a_{t-1}; a_{-1}=0
        oldlp[i, :T] = torch.as_tensor(ep["old_lp"], dtype=torch.float32)
        adv[i, :T] = torch.as_tensor(ep["adv"], dtype=torch.float32)
        ret[i, :T] = torch.as_tensor(ep["ret"], dtype=torch.float32)
        mask[i, :T] = 1.0
    # normalize advantages over the valid steps only
    valid = mask.bool()
    adv = (adv - adv[valid].mean()) / (adv[valid].std() + 1e-8)
    m = mask
    denom = m.sum()
    for _ in range(epochs):
        mu, std, v, _ = net.forward(obs, aprev)             # (E,Tmax,act),(act,),(E,Tmax)
        action_mask = net.action_mask
        newlp, ent = evaluate_policy_action(
            mu, std, act, action_mask, net.squash_actions)
        ratio = (newlp - oldlp).exp()
        l1 = ratio * adv
        l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        pol = -torch.min(l1, l2)
        vloss = (v - ret) ** 2
        loss = ((pol + 0.5 * vloss - ent_coef * ent) * m).sum() / denom
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm); opt.step()


# ------------------------ stochastic-demand bank (CRN) ----------------------
def _compute_one_baseline(task):
    """One GA-plan rollout for a given demand realization. Module-level so it is
    picklable for the spawn Pool. Each pool worker uses node-local pid-tagged
    runtime files, so the N baselines run fully in parallel."""
    backbone, seed, episode_end, params, idx, port_base = task
    env = TrainingEnv(
        backbone=backbone, seed=seed, episode_end=episode_end,
        port=port_base + idx)
    env.set_demand(params)
    bl, ref = env.compute_baseline()
    env.close()
    return idx, params, bl, ref.get("vmt_per_vht", 0.0)


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path):
    """Content digest for the geometry files that define observation ownership."""
    import hashlib
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for filename in sorted(files):
            full_path = os.path.join(root, filename)
            relative = os.path.relpath(full_path, path)
            digest.update(relative.encode("utf-8"))
            digest.update(_sha256(full_path).encode("ascii"))
    return digest.hexdigest()


def _bank_metadata(args):
    """Configuration fingerprint required for a reusable difference-reward bank."""
    signal_path = C.signal_path(args.backbone)
    _points, scenario_metadata = scenarios.load_scenario_file(
        args.train_scenario_file, expected_n=args.demand_bank_size)
    metadata = {
        "schema_version": 9 if demand.explicit_demand_enabled() else 8,
        "control_layout": C.CONTROL_LAYOUT,
        "agent_order": list(C.AGENT_ORDER),
        "backbone": args.backbone,
        "seed": int(args.seed),
        "episode_end": int(args.episode_end),
        "warmup_s": int(C.WARMUP_S),
        "decision_interval_s": int(C.DECISION_INTERVAL_S),
        "expected_policy_steps": int(max(
            0, (args.episode_end - C.WARMUP_S) // C.DECISION_INTERVAL_S)),
        "demand_bank_size": int(args.demand_bank_size),
        "demand_sampler": scenario_metadata["sampler"],
        "demand_dimensions": list(scenarios.DIMENSIONS),
        "direction_allocation": "eb=2*d;wb=2*(1-d)",
        "train_scenario_file": os.path.relpath(
            args.train_scenario_file, C.ROOT),
        "train_scenario_sha256": scenarios.file_sha256(
            args.train_scenario_file),
        # JSON-canonical form prevents tuple->list conversion from making a bank
        # reject itself the next time it is loaded.
        "demand_ranges": {
            key: [float(value[0]), float(value[1])]
            for key, value in demand.DEFAULT_RANGES.items()
        },
        "collector": os.environ.get("RL_COLLECTOR", "fast"),
        "use_expected_delay": bool(C.USE_EXPECTED_DELAY),
        "speed_percentiles": list(C.SPEED_PERCENTILES),
        "expected_delay_quadrature": C.EXPECTED_DELAY_QUADRATURE,
        "trips_sha256": _sha256(C.TRIPS),
        "signal_sha256": _sha256(signal_path),
        "detectors_sha256": _sha256(C.DETECTORS),
        "approach_geometry_sha256": _sha256_tree(C.APPROACH_DIR),
    }
    if C.RL_METHOD == "diaz":
        metadata["comparison_method"] = "diaz_dual_loop_queue_v1"
    if demand.explicit_demand_enabled():
        metadata.update({
            "explicit_demand": True,
            "probe_penetration": float(demand.probe_penetration()),
            "explicit_demand_seed": int(demand.explicit_demand_seed()),
            "probe_membership_seed": int(demand.probe_membership_seed()),
        })
    return metadata


def _validate_bank_entries(entries, metadata, path):
    if len(entries) != metadata["demand_bank_size"]:
        raise RuntimeError(
            f"Demand bank {path} has {len(entries)} entries; expected "
            f"{metadata['demand_bank_size']}. Rebuild it with --rebuild-bank.")
    expected_steps = metadata["expected_policy_steps"]
    bad = [
        (i, len(entry.get("baseline", [])))
        for i, entry in enumerate(entries)
        if len(entry.get("baseline", [])) != expected_steps
    ]
    if bad:
        raise RuntimeError(
            f"Demand bank {path} has baseline-length mismatches "
            f"(expected {expected_steps}, first mismatches: {bad[:5]}). "
            "Rebuild it with --rebuild-bank.")


def build_or_load_demand_bank(args, out, tag, port_base):
    """Pre-sample N demand realizations and, for EACH, run the GA plan once to
    cache its per-step delay -> the paired (common-random-numbers) difference-
    reward baseline. The N rollouts run on a parallel Pool (the PPO workers aren't
    started yet, so the CPUs are free). Cached to the SHARED demand_bank.json
    (no tag) so concurrent MLP/BiLSTM runs use identical demand+baselines."""
    import json
    default_bank = (
        f"baseline_cache_{'diaz_' if C.RL_METHOD == 'diaz' else ''}"
        f"{C.CONTROL_LAYOUT}_{args.backbone}_"
        f"lhs_n{args.demand_bank_size}.json")
    path = args.bank_path or os.path.join(out, default_bank)
    metadata = _bank_metadata(args)
    if os.path.exists(path) and not args.rebuild_bank:
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, list):
            if C.CONTROL_LAYOUT == "all10":
                raise RuntimeError(
                    f"Demand bank {path} uses the unversioned legacy format. "
                    "Rebuild it with --rebuild-bank before all10 training.")
            entries = payload
        else:
            entries = payload.get("entries", [])
            actual = payload.get("metadata", {})
            mismatches = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in metadata.items()
                if actual.get(key) != value
            }
            if mismatches:
                preview = dict(list(mismatches.items())[:6])
                raise RuntimeError(
                    f"Demand bank {path} does not match this run: {preview}. "
                    "Use the correct bank or rebuild it with --rebuild-bank.")
        _validate_bank_entries(entries, metadata, path)
        bank = [
            (tuple(d["params"]), d["baseline"], d.get("ga_mph", 0.0))
            for d in entries
        ]
        print(f"loaded demand bank: {len(bank)} realizations from {path}", flush=True)
        if any(b[2] == 0.0 for b in bank):
            print("  NOTE: cached bank has no ga_mph -> use --rebuild-bank to record it",
                  flush=True)
        return bank
    # The scenario design is generated once and shared by every experiment.
    param_list, _scenario_metadata = scenarios.load_scenario_file(
        args.train_scenario_file, expected_n=args.demand_bank_size)
    tasks = [(args.backbone, args.seed, args.episode_end, p, j, port_base)
             for j, p in enumerate(param_list)]
    print(f"building demand bank: {args.demand_bank_size} realizations on "
          f"{args.workers} parallel workers (one GA rollout each)...", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_compute_one_baseline, tasks)
    results.sort(key=lambda r: r[0])                  # restore sampling order
    bank = [(p, bl, mph) for _idx, p, bl, mph in results]   # (params, per-step baseline, GA mph)
    for idx, p, bl, mph in results:
        print(f"  demand {idx:3d}  {demand.param_key(p)}  "
              f"GA baseline mean_delay={np.mean(bl):.3f} veh-h  mph={mph:.2f}", flush=True)
    entries = [
        {"params": list(p), "baseline": bl, "ga_mph": mph}
        for p, bl, mph in bank
    ]
    _validate_bank_entries(entries, metadata, path)
    payload = {"metadata": metadata, "entries": entries}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
    print(f"wrote {path}", flush=True)
    return bank


def select_nested_demand_subset(bank, limit, seed):
    """Select a reproducible nested subset without rebuilding its baselines.

    ``bank`` is the master cache (normally 2,000 LHS scenarios).  For a fixed
    seed, the first 200 selected entries are contained in the first 500, which
    are contained in the first 1,000.  Sorting the retained master indices keeps
    their on-disk order stable while preserving that nesting property.
    """
    master_size = len(bank)
    effective_size = master_size if int(limit) == 0 else int(limit)
    if not 1 <= effective_size <= master_size:
        raise ValueError(
            f"demand-bank-limit must be 0 or in [1, {master_size}], "
            f"got {limit}")
    if effective_size == master_size:
        return bank, list(range(master_size))
    rng = np.random.default_rng(int(seed))
    selected = sorted(
        int(index) for index in rng.permutation(master_size)[:effective_size])
    subset = [bank[index] for index in selected]
    print(
        f"selected nested demand subset: {effective_size}/{master_size} "
        f"master scenarios (seed={seed}, first master indices={selected[:10]})",
        flush=True)
    return subset, selected


def build_eval_set(args, n_eval, port_base):
    """A FIXED set of n_eval HELD-OUT demand realizations (distinct RNG from the
    training bank) + their GA baselines. Re-used every iteration so the eval curve
    is comparable across updates and measures GENERALIZATION (these triples are not
    in the training bank). Returns [(params, baseline), ...]."""
    all_params, _metadata = scenarios.load_scenario_file(
        args.eval_scenario_file)
    if n_eval > len(all_params):
        raise ValueError(
            f"requested {n_eval} online-eval scenarios, but "
            f"{args.eval_scenario_file} contains only {len(all_params)}")
    param_list = all_params[:n_eval]
    tasks = [(args.backbone, args.seed, args.episode_end, p, j, port_base)
             for j, p in enumerate(param_list)]
    print(f"building eval set: {n_eval} held-out scenarios (one GA rollout each)...",
          flush=True)
    with mp.get_context("spawn").Pool(min(n_eval, args.workers)) as pool:
        res = pool.map(_compute_one_baseline, tasks)
    res.sort(key=lambda r: r[0])
    eval_set = [(p, bl, mph) for _i, p, bl, mph in res]    # (params, per-step baseline, GA mph)
    for i, p, bl, mph in res:
        print(f"  eval {i}: {demand.param_key(p)}  GA baseline mph={mph:.2f}", flush=True)
    return eval_set


# ----------------------------- worker (sampler) -----------------------------
def worker_loop(wid, base, seed, episode_end, baseline, arch, lstm_hid, demand_bank,
                tag, port, no_diff, plain_reward_scale, weight_q, result_q,
                eval_result_q, error_q):
    """Per round receives (state_dict, n_train_eps, eval_list). Rolls out n_train_eps
    STOCHASTIC training episodes (-> result_q) then, for each (params, baseline) in
    eval_list, one DETERMINISTIC eval episode (action = mu, no exploration) on that
    held-out scenario (-> eval_result_q). If no_diff, the TRAINING reward is plain
    -delay (no incumbent baseline subtracted); demand is still randomized and the
    held-out evaluation still reports mph against the incumbent's mph. The plain
    reward scale is an optimization-only numerical scale and does not alter those
    physical evaluation metrics."""
    torch.set_num_threads(1)
    # Spawned workers otherwise receive OS-random torch seeds. Fixed per-worker
    # seeds make architecture ablations and reruns reproducible.
    worker_seed = 1_000_003 * int(seed) + int(wid)
    torch.manual_seed(worker_seed)
    env = TrainingEnv(base=base, seed=seed, episode_end=episode_end, port=port)
    env.baseline = None if no_diff else baseline
    env.plain_reward_scale = plain_reward_scale
    net = build_net(arch, lstm_hid)
    try:
        while True:
            msg = weight_q.get()
            if msg == "STOP":
                break
            state_payload, n_eps, eval_list, iteration, scenario_indices = msg
            _load_broadcast_state_dict(net, state_payload)
            for episode_index in range(n_eps):
                # Training only needs aggregate monitoring metrics; reward/state are
                # probe-based. This replaces a multi-MB tripinfo with a ~KB file.
                env.set_metrics_mode("statistics")
                # Episode-local seeds make a resumed iteration bitwise repeatable
                # without serializing worker RNG state.
                episode_seed = (
                    worker_seed
                    + 100_003 * int(iteration)
                    + 1_009 * int(episode_index)
                )
                torch.manual_seed(episode_seed)
                episode_rng = np.random.default_rng(episode_seed)
                ga_mph = 0.0
                if demand_bank:                      # randomized demand from the shared bank
                    if scenario_indices is None:
                        scenario_index = int(
                            episode_rng.integers(len(demand_bank)))
                    else:
                        scenario_index = int(scenario_indices[episode_index])
                    params, bl, ga_mph = demand_bank[scenario_index]
                    env.set_demand(params)           # same realization the baseline was built on
                    # paired (difference) reward subtracts the incumbent baseline (CRN);
                    # the no_diff ablation instead uses plain -delay (baseline=None)
                    env.baseline = None if no_diff else bl
                s = env.reset()
                traj, ret, probe_delay_sum, offset_penalty_sum, offset_delta_sum, steps, info, done = (
                    [], 0.0, 0.0, 0.0, 0.0, 0, {}, False)
                h = None                                     # GRU belief (recurrent)
                a_prev = np.zeros(C.ACTION_DIM, dtype=np.float32)
                while not done:
                    st = torch.as_tensor(s, dtype=torch.float32)
                    with torch.no_grad():
                        if arch in RECURRENT_ARCHS:
                            a, logp, v, h = net.step(
                                st, torch.as_tensor(a_prev), h)
                        else:
                            a, logp, v = net.act(st)
                    a_np = a.numpy()
                    ns, r, done, info = env.step(a_np)
                    traj.append((s, a_np, float(logp), float(v), r, float(done)))
                    s = ns; a_prev = a_np; ret += r
                    probe_delay_sum += info["total_delay_vehh"]; steps += 1
                    offset_penalty_sum += info["offset_deviation_penalty"]
                    offset_delta_sum += info["mean_abs_offset_adjustment_s"]
                result_q.put((wid, traj, {
                    "ret": ret,
                    "probe_delay_vehh": probe_delay_sum / max(steps, 1),
                    "offset_penalty": offset_penalty_sum / max(steps, 1),
                    "offset_delta_s": offset_delta_sum / max(steps, 1),
                    "time_loss": info.get("timeloss_s_per_veh", 0.0),
                    "steps": steps,
                    "vmt": info.get("vmt_mi", 0.0), "vht": info.get("vht_h", 0.0),
                    "ratio": info.get("vmt_per_vht", 0.0), "ga_mph": ga_mph,
                    "wait": info.get("wait_s_per_veh", 0.0)}))
            # deterministic evaluation on the assigned held-out scenario(s)
            for sidx, eparams, ebl, ega_mph in eval_list:
                # Keep exact per-vehicle metrics for held-out/final comparisons.
                env.set_metrics_mode("tripinfo")
                env.set_demand(eparams); env.baseline = ebl
                s = env.reset()
                eret, eprobe_delay_sum, esteps, einfo, done = (
                    0.0, 0.0, 0, {}, False)
                h = None
                a_prev = np.zeros(C.ACTION_DIM, dtype=np.float32)
                while not done:
                    st = torch.as_tensor(s, dtype=torch.float32)
                    with torch.no_grad():
                        if arch in RECURRENT_ARCHS:       # action = mean, carry belief
                            mu, _v, h = net.act_mean(
                                st, torch.as_tensor(a_prev), h)
                        else:
                            mu, _std, _v = net.forward(st)   # action = mean -> no noise
                    a_np = mu.numpy()
                    ns, r, done, einfo = env.step(a_np)
                    s = ns; a_prev = a_np; eret += r
                    eprobe_delay_sum += einfo["total_delay_vehh"]; esteps += 1
                eval_result_q.put({
                    "sidx": sidx,
                    "ret": eret,
                    "probe_delay_vehh": (
                        eprobe_delay_sum / max(esteps, 1)),
                    "time_loss": einfo.get("timeloss_s_per_veh", 0.0),
                    "ratio": einfo.get("vmt_per_vht", 0.0),
                    "ga_mph": ega_mph,
                    "wait": einfo.get("wait_s_per_veh", 0.0),
                })
    except BaseException:
        import traceback
        error_q.put({
            "worker": wid,
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        env.close()


def _training_signature(args):
    """Fields that must match before optimizer/model state can be resumed."""
    enhanced_sampling_or_norm = (
        USE_SPATIOTEMPORAL_LAYER_NORM
        or args.demand_sampling != "with_replacement")
    signature = {
        "training_schema_version": (
            (13 if demand.explicit_demand_enabled() else 12)
            if enhanced_sampling_or_norm
            else (11 if demand.explicit_demand_enabled() else 10)),
        "monitoring_metrics": "sumo_time_loss_v1",
        "control_layout": C.CONTROL_LAYOUT,
        "agent_order": list(C.AGENT_ORDER),
        "state_schema": C.STATE_SCHEMA,
        "direction_features": list(C.DIR_FEATURES),
        "delay_ratio_saturation": float(C.DELAY_RATIO_SATURATION),
        "state_dim": C.STATE_DIM,
        "action_dim": C.ACTION_DIM,
        "active_action_dim": int(ACTION_MASK.sum()),
        "action_mask": [float(value) for value in ACTION_MASK],
        "squash_actions": bool(SQUASH_ACTIONS),
        "state_dependent_std": bool(STATE_DEPENDENT_STD),
        "policy_distribution": (
            "tanh_normal_state_std_v1"
            if SQUASH_ACTIONS and STATE_DEPENDENT_STD
            else "tanh_normal_v1"
            if SQUASH_ACTIONS
            else "legacy_normal"),
        "log_std_min": float(LOG_STD_MIN),
        "log_std_max": float(
            STATE_LOG_STD_MAX
            if STATE_DEPENDENT_STD else GLOBAL_LOG_STD_MAX),
        "split_control": bool(C.SPLIT_CONTROL),
        "residual_action": bool(C.RESIDUAL_ACTION),
        "backbone": args.backbone,
        "arch": args.arch,
        "lstm_hid": int(args.lstm_hid),
        "workers": int(args.workers),
        "traj_per_update": int(args.traj_per_update),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "clip": float(args.clip),
        "max_grad_norm": float(args.max_grad_norm),
        "optimizer": "adamw",
        "weight_decay": float(args.weight_decay),
        "lr": float(args.lr),
        "gamma": float(args.gamma),
        "lam": float(args.lam),
        "episode_end": int(args.episode_end),
        "warmup_s": int(C.WARMUP_S),
        "decision_interval_s": int(C.DECISION_INTERVAL_S),
        "seed": int(args.seed),
        "distinct_seeds": bool(args.distinct_seeds),
        "random_demand": bool(args.random_demand),
        # ``demand_bank_size`` is the cached master size.  Bank-size ablations
        # vary only the effective nested subset and therefore reuse all master
        # incumbent trajectories.
        "demand_bank_size": int(args.demand_bank_size),
        "effective_demand_bank_size": int(
            args.demand_bank_limit or args.demand_bank_size),
        "demand_subset_method": "seeded_nested_permutation_v1",
        "demand_subset_seed": int(args.demand_subset_seed),
        "train_scenario_sha256": scenarios.file_sha256(
            args.train_scenario_file),
        "eval_scenario_sha256": scenarios.file_sha256(
            args.eval_scenario_file),
        "demand_ranges": {
            key: [float(value[0]), float(value[1])]
            for key, value in demand.DEFAULT_RANGES.items()
        },
        "demand_dimensions": list(scenarios.DIMENSIONS),
        "direction_allocation": "eb=2*d;wb=2*(1-d)",
        "no_baseline": bool(args.no_baseline),
        "no_diff_reward": bool(args.no_diff_reward),
        "plain_reward_scale": float(args.plain_reward_scale),
        "no_warmstart": bool(args.no_warmstart),
        "log_std_init": float(args.log_std_init),
        "ent_coef0": float(args.ent_coef0),
        "ent_coef_final": float(args.ent_coef_final),
        "ent_decay_iters": int(args.ent_decay_iters),
        "eval_every": int(args.eval_every),
        "eval_scenarios": int(args.eval_scenarios),
        "collector": os.environ.get("RL_COLLECTOR", "fast"),
        "use_expected_delay": bool(C.USE_EXPECTED_DELAY),
        "expected_delay_quadrature": C.EXPECTED_DELAY_QUADRATURE,
        "reward_scale": float(C.REWARD_SCALE),
        "reward_min_speed_kph": float(C.REWARD_MIN_SPEED_KPH),
        "action_half_range": float(C.ACTION_HALF_RANGE),
        "offset_deviation_penalty": float(C.OFFSET_DEVIATION_PENALTY),
        "split_half_range": float(C.SPLIT_HALF_RANGE),
        "trips_sha256": _sha256(C.TRIPS),
        "signal_sha256": _sha256(C.signal_path(args.backbone)),
        "detectors_sha256": _sha256(C.DETECTORS),
        "approach_geometry_sha256": _sha256_tree(C.APPROACH_DIR),
    }
    if C.RL_METHOD == "diaz":
        signature["comparison_method"] = "diaz_near_stopbar_queue_step_presence_v1"
    # Omit these keys under the exact historical setup so v3 training states can
    # still be resumed when the legacy options are selected explicitly.
    if enhanced_sampling_or_norm:
        signature.update({
            "spatiotemporal_layer_norm": bool(
                USE_SPATIOTEMPORAL_LAYER_NORM),
            "layer_norm_scope": (
                "per_intersection_hidden_v2"
                if USE_SPATIOTEMPORAL_LAYER_NORM else "none"),
            "demand_sampling": args.demand_sampling,
        })
    if args.arch in ("attention", "attention_gru"):
        signature.update({
            "attention_dim": int(args.lstm_hid),
            "attention_heads": int(ATTENTION_HEADS),
            "attention_dropout": float(ATTENTION_DROPOUT),
            "attention_position_encoding": ATTENTION_POSITION_ENCODING,
            "critic_pooling": ATTENTION_CRITIC_POOLING,
        })
    if demand.explicit_demand_enabled():
        signature.update({
            "explicit_demand": True,
            "probe_penetration": float(demand.probe_penetration()),
            "explicit_demand_seed": int(demand.explicit_demand_seed()),
            "probe_membership_seed": int(demand.probe_membership_seed()),
        })
    return signature


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_save(obj, path):
    import json
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _load_torch_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before weights_only=
        return torch.load(path, map_location="cpu")


def _serialize_broadcast_state_dict(net):
    """Serialize one policy snapshot before broadcasting it to rollout workers.

    Sending a tensor-valued ``state_dict`` through every ``torch.multiprocessing``
    queue makes PyTorch transfer each tensor through shared-memory file
    descriptors.  With a recurrent policy and dozens of worker queues, those
    descriptors can exceed an HPCC process's ``RLIMIT_NOFILE`` before the first
    rollout begins.  A single in-memory byte payload uses ordinary queue data and
    therefore keeps the worker count independent of the number of model tensors.
    """
    buffer = io.BytesIO()
    torch.save(
        {key: value.detach().cpu() for key, value in net.state_dict().items()},
        buffer,
    )
    return buffer.getvalue()


def _load_broadcast_state_dict(net, payload):
    """Load a serialized worker broadcast; accept old dict payloads in tests."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        buffer = io.BytesIO(bytes(payload))
        try:
            state_dict = torch.load(
                buffer, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions predating ``weights_only``
            buffer.seek(0)
            state_dict = torch.load(buffer, map_location="cpu")
    else:
        state_dict = payload
    net.load_state_dict(state_dict)


def _latest_train_state(tag, base, expected_signature=None):
    """Return the newest compatible state, skipping checkpoints from old setups."""
    import glob
    import re
    pattern = os.path.join(
        OUT, f"ppo_parallel_trainstate_{tag}_{base}_*.pt")
    candidates = []
    for path in glob.glob(pattern):
        match = re.search(r"_(\d+)\.pt$", path)
        if match:
            candidates.append((int(match.group(1)), path))
    candidates.sort(reverse=True)
    if expected_signature is None:
        return candidates[0][1] if candidates else ""
    skipped = []
    for _iteration, path in candidates:
        try:
            payload = _load_torch_file(path)
            actual_signature = payload.get("signature", {})
        except Exception as exc:
            skipped.append((path, f"unreadable: {exc}"))
            continue
        mismatches = {
            key: {"expected": value, "actual": actual_signature.get(key)}
            for key, value in expected_signature.items()
            if actual_signature.get(key) != value
        }
        if not mismatches:
            if skipped:
                print(
                    f"auto-resume: skipped {len(skipped)} newer incompatible "
                    f"state(s); using {path}", flush=True)
            return path
        skipped.append((path, dict(list(mismatches.items())[:4])))
    if skipped:
        print(
            f"auto-resume: found {len(skipped)} state(s), but none match the "
            f"current training signature; starting fresh. Newest mismatch: "
            f"{skipped[0]}", flush=True)
    return ""


def _collect_worker_results(result_q, count, procs, error_q, label,
                            stall_timeout):
    """Collect queue items while surfacing worker crashes instead of deadlocking."""
    items = []
    deadline = time.monotonic() + stall_timeout
    while len(items) < count:
        try:
            items.append(result_q.get(timeout=min(30.0, stall_timeout)))
            deadline = time.monotonic() + stall_timeout
            continue
        except queue.Empty:
            pass
        try:
            failure = error_q.get_nowait()
        except queue.Empty:
            failure = None
        if failure is not None:
            raise RuntimeError(
                f"Worker {failure['worker']} failed while collecting {label}:\n"
                f"{failure['traceback']}")
        dead = [
            (idx, proc.exitcode)
            for idx, proc in enumerate(procs)
            if proc.exitcode not in (None, 0)
        ]
        if dead:
            raise RuntimeError(
                f"Workers exited while collecting {label}: {dead}")
        if time.monotonic() >= deadline:
            alive = [idx for idx, proc in enumerate(procs) if proc.is_alive()]
            raise TimeoutError(
                f"No {label} result arrived for {stall_timeout:.0f}s "
                f"({len(items)}/{count} received; alive workers={alive}).")
    return items


def _without_replacement_indices(bank_size, batch_size, seed, iteration):
    """Deterministic unique demand-bank indices for one PPO update.

    The mapping depends only on the training seed and update number, so resumed
    runs reproduce the same scenario batch without serializing sampler state.
    """
    if batch_size > bank_size:
        raise ValueError("batch_size cannot exceed bank_size")
    rng = np.random.default_rng(
        int(seed) + 7_919 * int(iteration) + 104_729)
    return rng.choice(bank_size, size=batch_size, replace=False).tolist()


def _open_training_log(path, header, start_iter):
    """Open a resumable CSV, discarding rows newer than the restored checkpoint."""
    if start_iter > 0 and os.path.exists(path):
        with open(path, newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or rows[0] != header:
            raise RuntimeError(
                f"Training log schema does not match this run: {path}")
        kept = [rows[0]]
        for row in rows[1:]:
            if not row:
                continue
            try:
                iteration = int(row[0])
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid iteration in training log {path}: {row[0]!r}") from exc
            if iteration < start_iter:
                kept.append(row)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", newline="") as handle:
            csv.writer(handle).writerows(kept)
        os.replace(tmp_path, path)
        log = open(path, "a", newline="")
        return log, csv.writer(log)
    log = open(path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(header)
    log.flush()
    return log, writer


# ----------------------------- learner (main) -----------------------------
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone", choices=C.BACKBONE_CHOICES, default=None,
        help="canonical signal controller loaded beneath the residual policy")
    ap.add_argument(
        "--base", choices=["fixed", "actuated"], default=None,
        help="deprecated compatibility alias: fixed=coord_fixed, "
             "actuated=coord_actuated")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--traj-per-update", type=int, default=16,
                    help="full episodes collected per PPO update (across all workers)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="global actor-critic gradient-norm clipping threshold")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW decoupled weight-decay coefficient")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--episode-end", type=int, default=C.EPISODE_END_S)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distinct-seeds", action="store_true")
    ap.add_argument("--arch", choices=[
                        "mlp", "bilstm", "attention", "gru", "bilstm_gru",
                        "attention_gru"],
                    default="bilstm",
                    help="mlp: memoryless flat. bilstm: spatial fusion over the "
                         "selected corridor nodes. attention: projected residual "
                         "self-attention without temporal memory. "
                         "gru: temporal belief (=mlp+gru). bilstm_gru: spatial+temporal "
                         "with a BiLSTM. attention_gru: projected residual self-attention "
                         "+ per-node temporal GRU + attention-pooled critic. Recurrent "
                         "architectures use per-episode sequence PPO updates")
    ap.add_argument("--lstm-hid", type=int, default=128,
                    help="recurrent hidden size; also the total Q/K/V projection "
                         "dimension for attention/attention_gru "
                         "(128 -> 4 heads x 32)")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument(
        "--resume-state", default="",
        help="training-state checkpoint to resume (model + optimizer + iteration)")
    ap.add_argument(
        "--auto-resume", action="store_true",
        help="resume the newest matching training-state checkpoint, if present")
    ap.add_argument("--no-warmstart", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-diff-reward", action="store_true",
                    help="reward ablation: under random demand, train on plain -delay "
                         "instead of the incumbent-relative paired difference. Demand and "
                         "the held-out eval (mph vs incumbent) are unchanged, so it is a "
                         "controlled comparison against the paired-reward run on the same bank")
    ap.add_argument("--plain-reward-scale", type=float, default=1.0,
                    help="scale c in r=-c*delay; used only with --no-diff-reward")
    ap.add_argument("--random-demand", action="store_true",
                    help=("train on four-parameter LHS demand with paired "
                          "common-random-number incumbent baselines"))
    ap.add_argument("--demand-bank-size", type=int, default=2000,
                    help="number of demand realizations to presample + cache GA baselines for")
    ap.add_argument(
        "--demand-bank-limit", type=int, default=0,
        help=("effective number of training scenarios selected from the cached "
              "master bank (0 = use all). A fixed --demand-subset-seed makes "
              "limits such as 200/500/1000 nested, so no baseline is rerun"))
    ap.add_argument(
        "--demand-subset-seed", type=int, default=42,
        help="seed for reproducible nested subsets of a larger master bank")
    ap.add_argument(
        "--demand-sampling",
        choices=("with_replacement", "without_replacement"),
        default="without_replacement",
        help=("sample training scenarios independently as before, or assign "
              "unique demand-bank entries within each PPO update"))
    ap.add_argument(
        "--train-scenario-file", default=str(scenarios.DEFAULT_TRAIN_FILE),
        help="versioned common LHS design used to build/sample the training bank")
    ap.add_argument(
        "--eval-scenario-file", default=str(scenarios.DEFAULT_EVAL_FILE),
        help="versioned common LHS design; its first --eval-scenarios points are "
             "used for deterministic online monitoring")
    ap.add_argument("--rebuild-bank", action="store_true",
                    help="recompute the demand bank even if the bank file exists")
    ap.add_argument("--bank-path", default="",
                    help="demand-bank file (defaults are isolated by control layout). Set to a "
                         "shared path so related runs reuse ONE bank, or to distinct paths "
                         "so concurrent unrelated jobs don't clobber each other's bank.")
    ap.add_argument(
        "--build-bank-only", action="store_true",
        help="build/validate the backbone-specific baseline cache and exit")
    ap.add_argument("--run-tag", default="",
                    help="suffix isolating this run's log/ckpt/tripinfo (default: --arch "
                         "plus _all10 in the 10-node environment). "
                         "Lets MLP and BiLSTM run concurrently on the SHARED demand bank.")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate the DETERMINISTIC policy every N updates (0 = off)")
    ap.add_argument("--eval-scenarios", type=int, default=3,
                    help="held-out demand scenarios averaged per evaluation")
    ap.add_argument("--log-std-init", type=float, default=-2.5,
                    help="initial exploration log-std (-2.5 -> about 2.5 s offset jitter)")
    ap.add_argument("--ent-coef0", type=float, default=0.002,
                    help="entropy coefficient at iter 0")
    ap.add_argument("--ent-coef-final", type=float, default=0.0001,
                    help="entropy coefficient reached at --ent-decay-iters, held after")
    ap.add_argument("--ent-decay-iters", type=int, default=300,
                    help="iterations over which entropy coef linearly decays to its final value")
    ap.add_argument("--port-base", type=int, default=0,
                    help="base TraCI port; worker w uses port-base+w (avoids the free-port "
                         "race at 27-way concurrency). 0 -> derive from pid (distinct per job)")
    ap.add_argument(
        "--worker-timeout", type=float, default=3600.0,
        help="fail if no worker result arrives for this many seconds")
    args = ap.parse_args()
    if args.backbone is not None and args.base is not None:
        if C.canonical_backbone(args.backbone) != C.canonical_backbone(args.base):
            ap.error("--backbone and deprecated --base select different controllers")
    args.backbone = C.canonical_backbone(args.backbone or args.base)
    # Preserve internal compatibility while all new metadata uses ``backbone``.
    args.base = args.backbone
    # Keep checkpoints, logs, evaluations, and baseline caches from different
    # local controllers physically separate even when run tags are similar.
    OUT = C.rl_output_dir(args.backbone)
    if args.resume_state and args.auto_resume:
        ap.error("use either --resume-state or --auto-resume, not both")
    if args.workers < 1 or args.traj_per_update < 1:
        ap.error("--workers and --traj-per-update must both be positive")
    effective_bank_size = int(
        args.demand_bank_limit or args.demand_bank_size)
    if args.demand_bank_size < 1:
        ap.error("--demand-bank-size must be positive")
    if not 1 <= effective_bank_size <= args.demand_bank_size:
        ap.error("--demand-bank-limit must be 0 or between 1 and "
                 "--demand-bank-size")
    if (args.random_demand
            and args.demand_sampling == "without_replacement"
            and args.traj_per_update > effective_bank_size):
        ap.error("without-replacement sampling requires traj-per-update "
                 "<= effective demand-bank size")
    if args.iters < 1 or args.ckpt_every < 1:
        ap.error("--iters and --ckpt-every must both be positive")
    if args.worker_timeout <= 0:
        ap.error("--worker-timeout must be positive")
    if args.max_grad_norm <= 0:
        ap.error("--max-grad-norm must be positive")
    if args.weight_decay < 0:
        ap.error("--weight-decay must be nonnegative")
    if args.plain_reward_scale <= 0:
        ap.error("--plain-reward-scale must be positive")
    if not args.no_diff_reward and args.plain_reward_scale != 1.0:
        ap.error("--plain-reward-scale is only valid with --no-diff-reward")
    log_std_max = (
        STATE_LOG_STD_MAX if STATE_DEPENDENT_STD else GLOBAL_LOG_STD_MAX)
    if not LOG_STD_MIN <= args.log_std_init <= log_std_max:
        ap.error(
            f"--log-std-init must be in [{LOG_STD_MIN}, {log_std_max}]")
    if args.episode_end <= C.WARMUP_S:
        ap.error("--episode-end must exceed the warmup interval")
    if ((args.episode_end - C.WARMUP_S)
            % C.DECISION_INTERVAL_S) != 0:
        ap.error(
            "--episode-end minus warmup must be divisible by the "
            "decision interval")
    if C.SPLIT_CONTROL and args.backbone != "coord_actuated":
        ap.error(
            "the current setProgramLogic green-split implementation is supported "
            "only with --backbone coord_actuated; it cannot preserve the custom "
            "phase-skipping conditions")
    if args.build_bank_only and not args.random_demand:
        ap.error("--build-bank-only requires --random-demand")
    if args.random_demand and args.no_baseline:
        ap.error(
            "--no-baseline is only for fixed-demand runs; use "
            "--no-diff-reward for the random-demand reward ablation")
    if (args.random_demand and args.distinct_seeds
            and not args.no_diff_reward):
        ap.error(
            "--distinct-seeds breaks the common-random-number demand-bank "
            "baseline; omit it or use --no-diff-reward")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    # Keep all10 artifacts separate from legacy8 by default. Explicit --run-tag
    # remains authoritative.
    default_tag = (
        f"{args.arch}_all10"
        if C.CONTROL_LAYOUT == "all10"
        else args.arch
    )
    tag = args.run_tag or default_tag
    # Use a distinct per-job range below Linux's ephemeral client-port range. The
    # bank currently keys ports by scenario index, so reserve enough room for all
    # bank/eval indices as well as the persistent rollout workers.
    port_span = max(args.workers, args.demand_bank_size, args.eval_scenarios)
    port_base = resolve_port_base(args.port_base, required_span=port_span)
    print(f"TraCI port_base={port_base} (worker w -> port_base+w; "
          f"reserved_span={port_span}; serialized free-port fallback on clash)",
          flush=True)

    # difference-reward baseline (once, in the main process)
    #  - random demand : one PAIRED GA baseline per demand realization (CRN)
    #  - fixed demand  : a single GA baseline at the calibrated demand (seed args.seed)
    baseline = None
    demand_bank = None
    if args.random_demand:
        demand_bank = build_or_load_demand_bank(args, OUT, tag, port_base)
        if not args.build_bank_only:
            demand_bank, _selected_master_indices = select_nested_demand_subset(
                demand_bank, args.demand_bank_limit, args.demand_subset_seed)
    elif not args.no_baseline and not args.no_diff_reward:
        print("computing GA baseline (action=0 -> coordinated plan)...", flush=True)
        benv = TrainingEnv(
            backbone=args.backbone, seed=args.seed,
            episode_end=args.episode_end,
            port=port_base)
        baseline, ref = benv.compute_baseline(); benv.close()
        print(f"baseline: mean delay/step={np.mean(baseline):.3f} veh-h  "
              f"ref VMT={ref.get('vmt_mi', 0):.0f}mi VHT={ref.get('vht_h', 0):.0f}h "
              f"mph={ref.get('vmt_per_vht', 0):.2f} "
              f"wait={ref.get('wait_s_per_veh', 0):.1f}s/veh", flush=True)

    if args.build_bank_only:
        print(
            f"baseline cache ready: backbone={args.backbone}, "
            f"scenarios={len(demand_bank)}", flush=True)
        return

    # fixed held-out eval set (only meaningful under random demand, where it tests
    # generalization to scenarios outside the training bank)
    eval_set = []
    if args.random_demand and args.eval_every > 0 and args.eval_scenarios > 0:
        eval_set = build_eval_set(args, args.eval_scenarios, port_base)

    net = build_net(args.arch, args.lstm_hid)
    warmstart_net(
        net, args.arch, log_std_init=args.log_std_init,
        warmstart_mean=not args.no_warmstart)
    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_iter = 0
    total_eps = 0
    resume_path = args.resume_state
    if args.auto_resume and not resume_path:
        resume_path = _latest_train_state(
            tag, args.backbone, expected_signature=_training_signature(args))
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        payload = _load_torch_file(resume_path)
        expected_signature = _training_signature(args)
        actual_signature = payload.get("signature", {})
        mismatches = {
            key: {"expected": value, "actual": actual_signature.get(key)}
            for key, value in expected_signature.items()
            if actual_signature.get(key) != value
        }
        if mismatches:
            preview = dict(list(mismatches.items())[:8])
            raise RuntimeError(
                f"Training state {resume_path} is incompatible: {preview}")
        net.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        start_iter = int(payload["next_iter"])
        total_eps = int(payload.get(
            "total_eps", start_iter * args.traj_per_update))
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if start_iter > args.iters:
            raise RuntimeError(
                f"Training state starts at iteration {start_iter}, beyond "
                f"--iters={args.iters}.")
        print(
            f"resuming {resume_path}: next_iter={start_iter}, "
            f"episodes={total_eps}", flush=True)
    elif args.auto_resume:
        print("auto-resume: no matching training state found; starting fresh",
              flush=True)
    if start_iter == args.iters:
        final_model = os.path.join(
            OUT, f"ppo_parallel_ckpt_{tag}_{args.backbone}_{args.iters}.pt")
        if not os.path.exists(final_model):
            _atomic_torch_save(net.state_dict(), final_model)
        if not os.path.exists(final_model + ".meta.json"):
            _atomic_json_save({
                "iteration": args.iters,
                "signature": _training_signature(args),
            }, final_model + ".meta.json")
        print(
            f"training already complete at iteration {start_iter}; "
            f"model={final_model}", flush=True)
        return

    # per-worker episode counts summing to traj_per_update
    K, T = args.workers, args.traj_per_update
    n_per = [T // K + (1 if i < T % K else 0) for i in range(K)]
    # Assign held-out scenarios round-robin from the least-loaded workers. This
    # also supports eval_scenarios > workers without leaving the learner waiting
    # forever for scenarios that were never assigned.
    eval_workers = sorted(range(K), key=lambda i: n_per[i])

    log_path = os.path.join(OUT, f"ppo_parallel_log_{tag}.csv")
    # vmt_per_vht = policy mph (stochastic train rollouts); ga_mph = GA mph of the
    # SAME sampled scenarios -> (vmt_per_vht - ga_mph) is the true mph advantage.
    # Evaluation is recorded per held-out scenario. Return remains the probe-based
    # learning signal, whereas mph/wait/timeLoss come directly from SUMO's final
    # all-vehicle output and are the physical monitoring metrics.
    n_eval = len(eval_set)
    eval_cols = []
    for s in range(n_eval):
        eval_cols += [
            f"eval{s}_return", f"eval{s}_mph", f"eval{s}_ga_mph",
            f"eval{s}_time_loss_s_per_veh",
        ]
    # eval_* columns are means over held-out scenarios; eval{s}_* keeps each
    # scenario separate for paired curves and uncertainty summaries.
    log_header = [
        "iter", "episodes", "ep_return_mean", "probe_delay_vehh",
        "offset_deviation_penalty", "mean_abs_offset_adjustment_s",
        "time_loss_s_per_veh", "vmt_mi", "vht_h", "vmt_per_vht",
        "wait_s_per_veh", "ga_mph",
        "n_transitions", "secs", "ent_coef", "eval_mph",
        "eval_ga_mph", "eval_wait_s", "eval_time_loss_s_per_veh",
    ] + eval_cols
    log, w = _open_training_log(log_path, log_header, start_iter)

    ctx = mp.get_context("spawn")
    weight_qs = [ctx.Queue() for _ in range(K)]
    result_q = ctx.Queue()
    eval_result_q = ctx.Queue()
    error_q = ctx.Queue()
    procs = []
    for i in range(K):
        wseed = args.seed + i if args.distinct_seeds else args.seed
        p = ctx.Process(target=worker_loop, daemon=True,
                        args=(i, args.backbone, wseed, args.episode_end, baseline,
                              args.arch, args.lstm_hid, demand_bank, tag, port_base + i,
                              args.no_diff_reward, args.plain_reward_scale,
                              weight_qs[i], result_q, eval_result_q, error_q))
        p.start(); procs.append(p)
    demand_mode = (
        f"RANDOM demand ({len(demand_bank)} realizations; "
        + ("plain reward" if args.no_diff_reward else "paired CRN baselines")
        + ")"
        if demand_bank else "fixed demand")
    if C.RL_METHOD == "diaz":
        reward_mode = "Diaz negative length-normalized queue (no offset prior)"
    elif args.no_diff_reward:
        reward_mode = (
            f"PLAIN -{args.plain_reward_scale:g}*delay (no-diff ablation); "
            f"offset prior={C.OFFSET_DEVIATION_PENALTY:g}*normalized-offset^2"
        )
    else:
        reward_mode = (
            "incumbent-relative paired difference; "
            f"objective scale={C.REWARD_SCALE:g}; "
            f"offset prior={C.OFFSET_DEVIATION_PENALTY:g}*normalized-offset^2 "
            "(scaled together)"
        )
    eval_mode = (f"eval {len(eval_set)} held-out / {args.eval_every} iters" if eval_set
                 else "eval off")
    act_mode = (f"residual +/-{C.ACTION_HALF_RANGE:.0f}s" if C.RESIDUAL_ACTION
                else "ABSOLUTE offset (no-residual ablation)")
    policy_steps = max(0, (args.episode_end - C.WARMUP_S) // C.DECISION_INTERVAL_S)
    print(f"Parallel PPO | env={C.CONTROL_LAYOUT} | arch={args.arch} | "
          f"workers={K} traj/update={T} "
          f"(~{T*policy_steps} transitions) | state={C.STATE_DIM} action={C.ACTION_DIM} "
          f"(active={int(ACTION_MASK.sum())}) "
          f"({act_mode}) | warmstart={not args.no_warmstart} "
          f"std={'state-dependent' if STATE_DEPENDENT_STD else 'global'} "
          f"log_std0={args.log_std_init} max={log_std_max:g} | "
          f"layernorm={USE_SPATIOTEMPORAL_LAYER_NORM} | "
          f"sampling={args.demand_sampling} | "
          f"ent {args.ent_coef0}->{args.ent_coef_final}"
          f"@{args.ent_decay_iters}it | optimizer=AdamW "
          f"weight_decay={args.weight_decay:g} "
          f"grad_clip={args.max_grad_norm:g} | "
          f"{demand_mode} | reward={reward_mode} | {eval_mode}  "
          f"(eval_mph vs eval_ga_mph is the comparison metric)", flush=True)

    try:
        for it in range(start_iter, args.iters):
            t0 = time.time()
            # entropy coef: linear decay from ent_coef0 to ent_coef_final by
            # ent_decay_iters, then held constant
            frac = min(1.0, it / max(1, args.ent_decay_iters))
            ent_coef = args.ent_coef0 + (args.ent_coef_final - args.ent_coef0) * frac
            # this round's eval assignment (one held-out scenario per least-loaded worker)
            do_eval = bool(eval_set) and (it % args.eval_every == 0)
            eval_assign = [[] for _ in range(K)]
            if do_eval:
                for j, scenario in enumerate(eval_set):
                    wi = eval_workers[j % K]
                    eval_assign[wi].append(
                        (j,) + scenario)   # (idx, params, baseline, ga_mph)
            scenario_assign = [None for _ in range(K)]
            if (demand_bank
                    and args.demand_sampling == "without_replacement"):
                scenario_batch = _without_replacement_indices(
                    len(demand_bank), T, args.seed, it)
                cursor = 0
                for worker_index, count in enumerate(n_per):
                    scenario_assign[worker_index] = scenario_batch[
                        cursor:cursor + count]
                    cursor += count
            # Serialize once, then send ordinary bytes to all workers. Sending the
            # tensor-valued state_dict to K torch queues consumes one shared-memory
            # file descriptor per tensor/queue pair and can exhaust HPCC limits at
            # K=40 (especially after adding LayerNorm parameters).
            state_payload = _serialize_broadcast_state_dict(net)
            for i in range(K):
                weight_qs[i].put((
                    state_payload, n_per[i], eval_assign[i], it,
                    scenario_assign[i]))
            results = _collect_worker_results(
                result_q, T, procs, error_q, "training trajectory",
                args.worker_timeout)
            # Queue arrival order depends on SUMO runtime. Sort by worker so MLP
            # minibatches and floating-point reductions are reproducible.
            results.sort(key=lambda item: item[0])
            evals = (
                _collect_worker_results(
                    eval_result_q, len(eval_set), procs, error_q,
                    "evaluation", args.worker_timeout)
                if do_eval else []
            )
            if args.arch in RECURRENT_ARCHS:
                # RECURRENT: keep episodes as ordered sequences (no transition
                # shuffle); GAE per episode; whole-episode BPTT in the update.
                episodes = []
                for _wid, traj, _m in results:
                    cols = list(zip(*traj))            # 6 tuples: s,a,lp,v,r,d
                    s_ = np.array(cols[0], np.float32); a_ = np.array(cols[1], np.float32)
                    lp_ = np.array(cols[2], np.float32); v_ = np.array(cols[3], np.float32)
                    r_ = np.array(cols[4], np.float32); d_ = np.array(cols[5], np.float32)
                    adv_, ret_ = gae(r_, v_, d_, gamma=args.gamma, lam=args.lam)
                    episodes.append({"obs": s_, "act": a_, "old_lp": lp_,
                                     "adv": adv_, "ret": ret_})
                ppo_update_recurrent(net, opt, episodes, epochs=args.epochs,
                                     clip=args.clip, ent_coef=ent_coef,
                                     max_grad_norm=args.max_grad_norm)
                n_trans = sum(len(e["obs"]) for e in episodes)
            else:
                # flatten trajectories (done flags keep GAE from bootstrapping across eps)
                S, A, LP, V, R, D = [], [], [], [], [], []
                for _wid, traj, _m in results:
                    for (s, a, lp, v, r, d) in traj:
                        S.append(s); A.append(a); LP.append(lp); V.append(v); R.append(r); D.append(d)
                S = np.array(S, np.float32); A = np.array(A, np.float32)
                LP = np.array(LP, np.float32); V = np.array(V, np.float32)
                R = np.array(R, np.float32); D = np.array(D, np.float32)
                adv, ret = gae(R, V, D, gamma=args.gamma, lam=args.lam)
                ppo_update(net, opt, S, A, LP, adv, ret,
                           epochs=args.epochs, bs=args.batch_size, clip=args.clip,
                           ent_coef=ent_coef,
                           max_grad_norm=args.max_grad_norm)
                n_trans = len(S)
            total_eps += T

            rets = [m["ret"] for _w, _t, m in results]
            probe_delays = [m["probe_delay_vehh"] for _w, _t, m in results]
            offset_penalties = [m["offset_penalty"] for _w, _t, m in results]
            offset_deltas = [m["offset_delta_s"] for _w, _t, m in results]
            time_losses = [m["time_loss"] for _w, _t, m in results]
            vmt = np.mean([m["vmt"] for _w, _t, m in results])
            vht = np.mean([m["vht"] for _w, _t, m in results])
            ratio = np.mean([m["ratio"] for _w, _t, m in results])
            ga_mph = np.mean([m["ga_mph"] for _w, _t, m in results])   # GA mph of sampled scenarios
            wait = np.mean([m["wait"] for _w, _t, m in results])       # mean waiting s/veh (policy)
            by_s = {e["sidx"]: e for e in evals} if do_eval else {}      # per-scenario eval results
            if do_eval:
                eval_mph_mean = float(np.mean([by_s[s]["ratio"] for s in by_s]))
                eval_ga_mean = float(np.mean([by_s[s]["ga_mph"] for s in by_s]))
                eval_wait_mean = float(np.mean([by_s[s]["wait"] for s in by_s]))
                eval_time_loss_mean = float(np.mean(
                    [by_s[s]["time_loss"] for s in by_s]))
            dt = time.time() - t0
            # print: average eval mph (vs GA) + per-scenario breakdown
            estr = ""
            if do_eval:
                parts = " ".join(
                    f"s{s}:{by_s[s]['ratio']:.2f}({by_s[s]['ratio']-by_s[s]['ga_mph']:+.2f})"
                    for s in range(n_eval) if s in by_s)
                estr = f"  EVAL mph={eval_mph_mean:.2f}(vsGA{eval_mph_mean-eval_ga_mean:+.2f}) [{parts}]"
            print(f"iter {it:3d} (ep {total_eps:4d})  return={np.mean(rets):7.2f}  "
                  f"mph={ratio:5.2f} (GA {ga_mph:5.2f}, {ratio-ga_mph:+.2f})  "
                  f"delay={np.mean(time_losses):5.1f}s/veh  "
                  f"wait={wait:5.1f}s/veh  "
                  f"offset_d={np.mean(offset_deltas):4.1f}s  "
                  f"pen={np.mean(offset_penalties):.4f}  "
                  f"ent={ent_coef:.4f}{estr}  N={n_trans:5d}  ({dt:.0f}s)", flush=True)
            row = [it, total_eps, np.mean(rets), np.mean(probe_delays),
                   np.mean(offset_penalties), np.mean(offset_deltas),
                   round(np.mean(time_losses), 2), round(vmt, 1),
                   round(vht, 2), round(ratio, 3), round(wait, 2), round(ga_mph, 3),
                   n_trans, round(dt, 1), round(ent_coef, 5)]
            row += ([round(eval_mph_mean, 3), round(eval_ga_mean, 3),
                     round(eval_wait_mean, 2), round(eval_time_loss_mean, 2)]
                    if do_eval else ["", "", "", ""])
            for s in range(n_eval):                                      # per-scenario eval columns
                if do_eval and s in by_s:
                    e = by_s[s]
                    row += [
                        round(e["ret"], 4), round(e["ratio"], 3),
                        round(e["ga_mph"], 3), round(e["time_loss"], 2),
                    ]
                else:
                    row += ["", "", "", ""]
            w.writerow(row)
            log.flush()
            if ((it + 1) % args.ckpt_every == 0
                    or (it + 1) == args.iters):
                model_path = os.path.join(
                    OUT,
                    f"ppo_parallel_ckpt_{tag}_{args.backbone}_{it+1}.pt")
                state_path = os.path.join(
                    OUT,
                    f"ppo_parallel_trainstate_{tag}_{args.backbone}_{it+1}.pt")
                _atomic_torch_save(net.state_dict(), model_path)
                _atomic_json_save({
                    "iteration": it + 1,
                    "signature": _training_signature(args),
                }, model_path + ".meta.json")
                _atomic_torch_save({
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "next_iter": it + 1,
                    "total_eps": total_eps,
                    "signature": _training_signature(args),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                }, state_path)
    finally:
        for q in weight_qs:
            q.put("STOP")
        for p in procs:
            p.join(timeout=15)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for q in weight_qs + [result_q, eval_result_q, error_q]:
            # Do not wait for a feeder thread to flush toward a worker that may
            # have just been terminated after an exception.
            q.cancel_join_thread()
            q.close()
        log.close()


if __name__ == "__main__":
    main()
