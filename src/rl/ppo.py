from __future__ import annotations

"""PPO implementation for the orchestration simulator.

The actor has the same architecture as the REINFORCE actor. Training differs
in the learned critic, GAE, clipped multi-epoch updates, horizon and sampling.
The experiment compares these complete procedures; it does not isolate the
causal contribution of the optimizer or any one implementation choice.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

from ..config import RunConfig
from .env import NUM_ACTIONS, OrchestrationEnv, observation_dim
from .policy import MLPPolicy


class ValueNet(nn.Module):
    """State-value critic V(s). Separate trunk from the actor."""

    def __init__(self, state_dim: int, hidden_sizes=(64, 64), activation: str = "tanh") -> None:
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, int(h)), act()]
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PPOConfig:
    total_iterations: int = 200
    episodes_per_iter: int = 16
    num_jobs: int = 100
    max_steps: int = 300
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 512
    lr: float = 3.0e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip_norm: float = 0.5
    init_seed: int = 7
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    eval_every: int = 10


@dataclass
class PPORecord:
    iteration: int
    mean_undisc_return: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    env_steps_cumulative: int


def compute_gae(rewards, values, gamma, lam):
    """GAE(λ). values has len T+1 (bootstrap 0 at terminal). Returns (adv, ret)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    ret = adv + np.array(values[:T])
    return adv, ret


def _collect_episode(env, actor, critic, seed):
    obs, _ = env.reset(seed=seed)
    S, A, LP, R, V = [], [], [], [], []
    while True:
        st = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            dist = actor.distribution(st)
            a = dist.sample()
            lp = dist.log_prob(a)
            v = critic(st)
        nobs, r, term, trunc, _ = env.step(int(a.item()))
        S.append(obs); A.append(int(a.item())); LP.append(float(lp.item()))
        R.append(float(r)); V.append(float(v.item()))
        obs = nobs
        if term or trunc:
            break
    V.append(0.0)  # bootstrap value at terminal
    return S, A, LP, R, V


def train_ppo(cfg: RunConfig, ppo: PPOConfig,
              training_seeds: Optional[List[int]] = None,
              validation_seeds: Optional[List[int]] = None,
              log: Optional[List[PPORecord]] = None,
              log_every: int = 10):
    torch.manual_seed(ppo.init_seed)
    np.random.seed(ppo.init_seed)
    sdim = observation_dim(cfg)
    actor = MLPPolicy(sdim, NUM_ACTIONS, ppo.hidden_sizes, ppo.activation)
    critic = ValueNet(sdim, ppo.hidden_sizes, ppo.activation)
    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=ppo.lr)

    pool = training_seeds if training_seeds is not None else list(cfg.rl.training_seeds)
    rng = np.random.default_rng(ppo.init_seed)
    cum_steps = 0
    best_val = -float("inf")
    best_state = {k: v.clone() for k, v in actor.state_dict().items()}

    for it in range(ppo.total_iterations):
        env = OrchestrationEnv(cfg=cfg, num_jobs=ppo.num_jobs, max_steps=ppo.max_steps,
                               memoryless=False)
        bS, bA, bLP, bAdv, bRet = [], [], [], [], []
        ep_returns = []
        for _ in range(ppo.episodes_per_iter):
            seed = int(pool[int(rng.integers(0, len(pool)))])
            S, A, LP, R, V = _collect_episode(env, actor, critic, seed)
            adv, ret = compute_gae(R, V, ppo.gamma, ppo.gae_lambda)
            bS += S; bA += A; bLP += LP; bAdv += list(adv); bRet += list(ret)
            ep_returns.append(sum(R)); cum_steps += len(R)

        S_t = torch.tensor(np.array(bS), dtype=torch.float32)
        A_t = torch.tensor(bA, dtype=torch.long)
        LP_t = torch.tensor(bLP, dtype=torch.float32)
        Adv_t = torch.tensor(bAdv, dtype=torch.float32)
        Ret_t = torch.tensor(bRet, dtype=torch.float32)
        Adv_t = (Adv_t - Adv_t.mean()) / (Adv_t.std() + 1e-8)

        n = len(bA)
        idx = np.arange(n)
        last_kl = last_clip = last_pl = last_vl = last_ent = 0.0
        for _ in range(ppo.ppo_epochs):
            rng.shuffle(idx)
            for start in range(0, n, ppo.minibatch_size):
                mb = idx[start:start + ppo.minibatch_size]
                mbt = torch.from_numpy(mb)
                dist = actor.distribution(S_t[mbt])
                lp = dist.log_prob(A_t[mbt])
                ratio = torch.exp(lp - LP_t[mbt])
                a = Adv_t[mbt]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                value = critic(S_t[mbt])
                value_loss = ((value - Ret_t[mbt]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + ppo.value_coef * value_loss - ppo.entropy_coef * entropy
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()),
                                         ppo.grad_clip_norm)
                opt.step()
                with torch.no_grad():
                    last_kl = float((LP_t[mbt] - lp).mean().item())
                    last_clip = float((torch.abs(ratio - 1) > ppo.clip_eps).float().mean().item())
                    last_pl, last_vl, last_ent = float(policy_loss.item()), float(value_loss.item()), float(entropy.item())

        rec = PPORecord(it, float(np.mean(ep_returns)), last_pl, last_vl, last_ent,
                        last_kl, last_clip, cum_steps)
        if log is not None:
            log.append(rec)

        # validation (greedy) for best-by-val
        if validation_seeds is not None and (it % ppo.eval_every == 0 or it == ppo.total_iterations - 1):
            vmean = _greedy_eval(cfg, actor, validation_seeds, ppo.num_jobs, ppo.max_steps)
            if vmean > best_val:
                best_val = vmean
                best_state = {k: v.clone() for k, v in actor.state_dict().items()}
            if log_every:
                print(f"[ppo it {it:3d}] ret={rec.mean_undisc_return:+7.2f} "
                      f"pl={last_pl:+.3f} vl={last_vl:.2f} ent={last_ent:.3f} "
                      f"kl={last_kl:+.4f} clip={last_clip:.2f} val={vmean:+.2f} best={best_val:+.2f}")
        elif log_every and it % log_every == 0:
            print(f"[ppo it {it:3d}] ret={rec.mean_undisc_return:+7.2f} "
                  f"pl={last_pl:+.3f} vl={last_vl:.2f} ent={last_ent:.3f}")

    return actor, critic, best_state, best_val


def _greedy_eval(cfg, actor, seeds, num_jobs, max_steps) -> float:
    actor.eval()
    env = OrchestrationEnv(cfg=cfg, num_jobs=num_jobs, max_steps=max_steps, memoryless=False)
    outs = []
    with torch.no_grad():
        for s in seeds:
            obs, _ = env.reset(seed=int(s)); tot = 0.0
            while True:
                st = torch.from_numpy(obs).float().unsqueeze(0)
                a = int(torch.argmax(actor.logits(st), dim=-1).item())
                obs, r, term, trunc, _ = env.step(a); tot += r
                if term or trunc:
                    break
            outs.append(tot)
    actor.train()
    return float(np.mean(outs))


__all__ = ["PPOConfig", "PPORecord", "ValueNet", "compute_gae", "train_ppo"]
