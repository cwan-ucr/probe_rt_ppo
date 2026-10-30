"""
Centralized PPO for probe-driven 15-min offset control (single policy outputs all
8 offsets). Minimal self-contained PPO in torch (no SB3/gym dependency).

This is the "centralized-first" validation: can one policy learn offsets that beat
the base plan's fixed offsets, using only probe-segment state + segment-delay
reward? Later swap the actor/critic for the CTDE-MARL version.

Run:
  python -m src.rl.train_centralized --base actuated --iters 100
Outputs: outputs/rl/ckpt_*.pt, outputs/rl/train_log.csv
"""
import os, csv, time, argparse
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .corridor_env import CorridorEnv

OUT = os.path.join(C.ROOT, "outputs", "rl")
DEV = "cpu"   # env is the bottleneck; cpu is fine for this MLP
ACTION_EPS = 1e-6
LOG_STD_MIN = -5.0
GLOBAL_LOG_STD_MAX = 1.0
# Keep legacy checkpoints reproducible by default. New phase-skipping runs can
# restrict state-dependent exploration with RL_STATE_LOG_STD_MAX=-1.5.
STATE_LOG_STD_MAX = float(os.environ.get("RL_STATE_LOG_STD_MAX", "0.0"))


def sample_policy_action(action_mean, std, action_mask, squash=False):
    """Sample an action and return its joint log-probability.

    The all10 experiments use a tanh-transformed Normal, so the sampled action,
    the action executed by the environment, and the recurrent ``a_prev`` are the
    same bounded value. Legacy experiments retain their historical unsquashed
    Normal behavior for checkpoint reproducibility.
    """
    if squash:
        mean = torch.atanh(action_mean.clamp(
            -1.0 + ACTION_EPS, 1.0 - ACTION_EPS))
        dist = torch.distributions.Normal(mean, std)
        latent = dist.sample()
        action = torch.tanh(latent) * action_mask
        logp_parts = dist.log_prob(latent) - torch.log(
            1.0 - action.pow(2) + ACTION_EPS)
    else:
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample() * action_mask
        logp_parts = dist.log_prob(action)
    return action, (logp_parts * action_mask).sum(-1)


def evaluate_policy_action(action_mean, std, action, action_mask,
                           squash=False):
    """Log-probability and entropy term for PPO under the rollout distribution."""
    if squash:
        bounded = action.clamp(-1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
        mean = torch.atanh(action_mean.clamp(
            -1.0 + ACTION_EPS, 1.0 - ACTION_EPS))
        latent = torch.atanh(bounded)
        dist = torch.distributions.Normal(mean, std)
        logp_parts = dist.log_prob(latent) - torch.log(
            1.0 - bounded.pow(2) + ACTION_EPS)
    else:
        dist = torch.distributions.Normal(action_mean, std)
        logp_parts = dist.log_prob(action)
    # Base-Normal entropy is a stable exploration regularizer. The PPO ratio uses
    # the exact tanh change-of-variables log-probability above.
    entropy_parts = dist.entropy()
    return ((logp_parts * action_mask).sum(-1),
            (entropy_parts * action_mask).sum(-1))


class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim, hid=256, action_mask=None,
                 squash_actions=False, state_dependent_std=False):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(s_dim, hid), nn.Tanh(),
                                   nn.Linear(hid, hid), nn.Tanh())
        self.mu = nn.Linear(hid, a_dim)
        self.v = nn.Linear(hid, 1)
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(hid, a_dim)
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, -2.0)
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
            self.log_std = nn.Parameter(torch.full((a_dim,), -0.5))
        mask = (torch.ones(a_dim, dtype=torch.float32) if action_mask is None
                else torch.as_tensor(action_mask, dtype=torch.float32))
        if mask.numel() != a_dim:
            raise ValueError(f"action_mask has {mask.numel()} entries; expected {a_dim}")
        self.register_buffer("action_mask", mask.reshape(a_dim), persistent=False)
        self.squash_actions = bool(squash_actions)

    def forward(self, s):
        h = self.trunk(s)
        mu = torch.tanh(self.mu(h)) * self.action_mask
        if self.state_dependent_std:
            log_std = self.log_std_head(h).clamp(
                LOG_STD_MIN, STATE_LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                LOG_STD_MIN, GLOBAL_LOG_STD_MAX)
        std = log_std.exp()
        return mu, std, self.v(h).squeeze(-1)

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


def collect(env, net, n_episodes):
    """Roll out n full episodes. Returns transition tensors + per-episode return,
    mean delay, and true VMT/VHT/mph (from each episode's tripinfo)."""
    S, A, LP, R, V, D = [], [], [], [], [], []
    ep_returns, ep_delays, ep_vmt, ep_vht, ep_ratio = [], [], [], [], []
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        ret, dsum, steps = 0.0, 0.0, 0
        info = {}
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=DEV)
            with torch.no_grad():
                a, logp, v = net.act(st)
            ns, r, done, info = env.step(a.cpu().numpy())
            S.append(s); A.append(a.cpu().numpy()); LP.append(float(logp))
            R.append(r); V.append(float(v)); D.append(float(done))
            s = ns; ret += r; dsum += info["total_delay_vehh"]; steps += 1
        ep_returns.append(ret); ep_delays.append(dsum / max(steps, 1))
        ep_vmt.append(info.get("vmt_mi", 0.0)); ep_vht.append(info.get("vht_h", 0.0))
        ep_ratio.append(info.get("vmt_per_vht", 0.0))
    return (np.array(S, np.float32), np.array(A, np.float32), np.array(LP, np.float32),
            np.array(R, np.float32), np.array(V, np.float32), np.array(D, np.float32),
            ep_returns, ep_delays, ep_vmt, ep_vht, ep_ratio)


def gae(R, V, D, gamma=0.99, lam=0.95):
    adv = np.zeros_like(R); last = 0.0
    for t in reversed(range(len(R))):
        nonterm = 1.0 - D[t]
        nextv = V[t + 1] if (t + 1 < len(R) and D[t] == 0) else 0.0
        delta = R[t] + gamma * nextv * nonterm - V[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + V


def ppo_update(net, opt, S, A, LP, adv, ret, epochs=10, bs=256, clip=0.2,
               ent_coef=0.01, max_grad_norm=0.5):
    S = torch.as_tensor(S, device=DEV); A = torch.as_tensor(A, device=DEV)
    LP = torch.as_tensor(LP, device=DEV)
    adv = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=DEV)
    ret = torch.as_tensor(ret, dtype=torch.float32, device=DEV)
    N = len(S)
    for _ in range(epochs):
        for i in range(0, N, bs):
            idx = slice(i, i + bs)
            newlp, ent, v = net.evaluate(S[idx], A[idx])
            ratio = (newlp - LP[idx]).exp()
            l1 = ratio * adv[idx]
            l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]
            pol = -torch.min(l1, l2).mean()
            vloss = ((v - ret[idx]) ** 2).mean()
            loss = pol + 0.5 * vloss - ent_coef * ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["fixed", "actuated"], default="actuated")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--eps-per-iter", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--episode-end", type=int, default=C.EPISODE_END_S)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--no-warmstart", action="store_true",
                    help="disable zero-init of the actor mean (start at GA offsets)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="disable the difference reward (use plain -delay)")
    args = ap.parse_args()
    if args.max_grad_norm <= 0:
        ap.error("--max-grad-norm must be positive")
    if args.weight_decay < 0:
        ap.error("--weight-decay must be nonnegative")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    env = CorridorEnv(base=args.base, episode_end=args.episode_end)
    net = ActorCritic(env.state_dim, env.action_dim).to(DEV)
    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # same experimental setup as SAC: warm-start the actor mean at the GA plan
    # (zero-init mu, modest log_std), then cache the GA difference-reward baseline.
    if not args.no_warmstart:
        nn.init.zeros_(net.mu.weight); nn.init.zeros_(net.mu.bias)
        with torch.no_grad():
            net.log_std.fill_(-0.5)
    if not args.no_baseline:
        print("computing GA baseline (action=0 -> coordinated plan)...", flush=True)
        base, ref = env.compute_baseline()
        print(f"baseline: mean delay/step={np.mean(base):.3f} veh-h  "
              f"ref VMT={ref.get('vmt_mi', 0):.0f}mi VHT={ref.get('vht_h', 0):.0f}h "
              f"mph={ref.get('vmt_per_vht', 0):.2f}", flush=True)

    log = open(os.path.join(OUT, "train_log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["iter", "ep_return_mean", "mean_delay_vehh", "vmt_mi", "vht_h",
                "vmt_per_vht", "secs"])
    log.flush()
    print(f"Centralized PPO | base={args.base} | state={env.state_dim} "
          f"action={env.action_dim} (delta +/-{C.ACTION_HALF_RANGE:.0f}s around GA) | "
          f"warmstart={not args.no_warmstart} | diff_reward={not args.no_baseline}  "
          f"(ep_return>0 == beating the GA plan)")
    try:
        for it in range(args.iters):
            t0 = time.time()
            S, A, LP, R, V, D, rets, dels, vmts, vhts, ratios = collect(
                env, net, args.eps_per_iter)
            adv, ret = gae(R, V, D)
            ppo_update(
                net, opt, S, A, LP, adv, ret,
                max_grad_norm=args.max_grad_norm)
            dt = time.time() - t0
            vmt, vht, ratio = np.mean(vmts), np.mean(vhts), np.mean(ratios)
            print(f"iter {it:3d}  return={np.mean(rets):8.2f}  "
                  f"mean_delay={np.mean(dels):5.2f} veh-h  "
                  f"VMT={vmt:7.0f}mi VHT={vht:6.0f}h mph={ratio:5.2f}  ({dt:.0f}s)")
            w.writerow([it, np.mean(rets), np.mean(dels), round(vmt, 1), round(vht, 2),
                        round(ratio, 3), round(dt, 1)]); log.flush()
            if (it + 1) % args.ckpt_every == 0:
                torch.save(net.state_dict(), os.path.join(OUT, f"ckpt_{args.base}_{it+1}.pt"))
    finally:
        env.close(); log.close()


if __name__ == "__main__":
    main()
