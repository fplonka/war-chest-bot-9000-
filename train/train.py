"""ReBeL training for War Chest.

Two phases inside one wall-clock budget:

1. **Warm start** (`--warm-frac` of the budget). Both players are a stochastic
   one-ply greedy bot on a public-information evaluation; value targets blend
   that evaluation (squashed into (-1, 1)) with the realised game outcome.
   ReBeL never plays a policy directly — every move comes out of CFR using the
   value network at the leaves — so the value network is the natural place to
   inject a starting behaviour. Without it CFR searches on noise and no game
   ever ends inside the horizon. The network at the end of this phase is the
   *initial checkpoint*.

2. **ReBeL** (the rest). Self-play where every decision solves a depth-limited
   CFR subgame over public belief states; targets are the CFR root values,
   projected onto the network's hand-key basis.

Everything except the gradient step runs in Rust across all cores; Python ships
weights down and pulls tensors back once per epoch.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest

FEAT = warchest.FEAT
NHAND = warchest.NHAND


class Mlp(nn.Module):
    """The value network, split around the belief block.

    Same shape and the same parameter count as a plain `FEAT -> h -> h -> dout`
    MLP; the only change is *where* the belief block connects. It is the last
    `SPLIT` inputs and it is the only part of a leaf's encoding that moves
    between CFR iterations, so wiring it into the second hidden layer instead of
    the first leaves the whole public tower — the widest matmul in the network,
    plus a full hidden layer — computable once per leaf per subgame solve rather
    than once per iteration. See `Mlp::trunk` in `engine/src/net.rs`.
    """

    def __init__(self, din, hidden, dout, split=None, layers=2):
        super().__init__()
        self.split = warchest.BELIEF_SPLIT if split is None else split
        dims = [din - self.split] + [hidden] * layers + [dout]
        self.dims = dims
        self.lin = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        # The belief block's connection into the second hidden layer. No bias:
        # it is added to a layer that already has one.
        self.bel = nn.Linear(self.split, hidden, bias=False)
        # LayerNorm on every hidden layer, as the reference does
        # (`use_layer_norm: true`). Two of the raw features are unbounded-ish
        # coin counts, and the bootstrapped targets shift scale over training,
        # so normalising between the affine and the activation is what keeps
        # the hidden distribution stable as the target distribution moves.
        self.norm = nn.ModuleList(nn.LayerNorm(d) for d in dims[1:-1])
        # Start near zero so the first bootstrapped targets are not dominated by
        # random leaf values.
        nn.init.zeros_(self.lin[-1].bias)
        nn.init.normal_(self.lin[-1].weight, std=1e-3)

    def forward(self, x):
        xp, xb = x[..., : self.dims[0]], x[..., self.dims[0]:]
        h = F.relu(self.norm[0](self.lin[0](xp)))
        h = F.relu(self.norm[1](self.lin[1](h) + self.bel(xb)))
        return self.lin[2](h)

    def push(self, slot):
        """Ship weights to the Rust workers (row-major `[in, out]` per layer)."""
        w = np.concatenate([l.weight.detach().cpu().t().contiguous().numpy().ravel()
                            for l in list(self.lin) + [self.bel]])
        b = np.concatenate([l.bias.detach().cpu().numpy().ravel() for l in self.lin])
        # Per hidden layer: LayerNorm weight then bias, in layer order.
        ln = np.concatenate([t.detach().cpu().numpy().ravel()
                             for n in self.norm for t in (n.weight, n.bias)])
        warchest.set_weights(self.dims, np.ascontiguousarray(w, np.float32),
                             np.ascontiguousarray(b, np.float32), slot,
                             np.ascontiguousarray(ln, np.float32), self.split)


class Buffer:
    """FIFO replay over flat sample arrays.

    Features are stored as float16. Bootstrapped targets are averaged over
    whatever history the buffer holds, so its length is a real algorithmic
    knob, not just a memory setting -- the reference implementation runs a 2M
    buffer. Halving the width of the widest array is what makes that affordable
    here.
    """

    def __init__(self, cap, widths, dtypes):
        self.cap = cap
        self.parts = [np.zeros((0, w), dtype=d) for w, d in zip(widths, dtypes)]

    def add(self, arrays):
        self.parts = [np.concatenate([p, a]) for p, a in zip(self.parts, arrays)]
        n = len(self.parts[0])
        if n > self.cap:
            self.parts = [p[n - self.cap:] for p in self.parts]

    def clear(self):
        self.parts = [p[:0] for p in self.parts]

    def __len__(self):
        return len(self.parts[0])

    def sample(self, batch, rng):
        idx = rng.integers(0, len(self), size=batch)
        return [p[idx].astype(np.float32, copy=False) for p in self.parts]


def unpack(d, key, width):
    a = np.asarray(d[key], dtype=np.float32)
    return a.reshape(len(a) // width, width)


def value_loss(net, vx, vy, vm):
    # Huber over the hand keys the belief actually supports.
    per = F.smooth_l1_loss(net(vx), vy, reduction="none", beta=0.5)
    return (per * vm).sum() / vm.sum().clamp(min=1.0)


def train_steps(net, opt, buf, steps, batch, rng, device):
    if len(buf) < batch:
        return float("nan")
    tot = 0.0
    for _ in range(steps):
        parts = [torch.as_tensor(p, device=device) for p in buf.sample(batch, rng)]
        loss = value_loss(net, *parts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        tot += loss.detach().item()
    return tot / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--warm-frac", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm-games", type=int, default=96)
    ap.add_argument("--rebel-games", type=int, default=48)
    ap.add_argument("--train-gen-ratio", type=float, default=4.0)
    # depth 1 puts *zero* opponent decision nodes in the subgame, which reduces
    # CFR to 1-ply value iteration over the network. 2 is the reference's
    # setting for liar's dice and the minimum that is actually ReBeL.
    ap.add_argument("--depth", type=int, default=2)
    # Measured against exact values on micro-endgames: mean |error| is 0.0018 at
    # T=16 and 0.0035 at T=8, both negligible beside a target spread of ~0.3 --
    # so halving the iteration count buys 2x the target rate almost for free,
    # and target rate is the binding constraint at depth 2.
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--explore", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--eval-mix", type=float, default=0.5)
    # Horizon payoff per marker of differential. Each side has 6 markers, so the
    # differential reaches +-5 and this must stay far below a real win (+-1) or
    # stalling out the clock becomes a competing win condition: at 0.15 a
    # five-marker lead banked 0.75 risk-free, which is what collapsed the first
    # run. 0.04 caps the shaped payoff at +-0.20.
    ap.add_argument("--cap-value", type=float, default=0.04)
    # Fraction of the ReBeL phase over which the horizon payoff decays to zero.
    # It reaches zero early so the tail of training -- and the checkpoint we
    # ship -- is fitted to the real game.
    ap.add_argument("--anneal-frac", type=float, default=0.4)
    # Gating is pure overhead against training time, so it runs rarely rather
    # than with many games: at 120 games the standard error is ~0.046, so a
    # peak 2 sigma above trend is probably noise and selecting on it biases the
    # reported score upward. Fewer, larger gates cost the same and select
    # better. `final_vs_init` is the headline number for exactly this reason --
    # it is not the quantity the checkpoint was selected on.
    ap.add_argument("--gate-every", type=float, default=1200.0)
    ap.add_argument("--gate-games", type=int, default=300)
    ap.add_argument("--cap", type=int, default=800_000)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--random-draft", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 8)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    value = Mlp(FEAT, args.hidden, 2 * NHAND).to(dev)
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    value.push(0)
    buf = Buffer(args.cap, [FEAT, 2 * NHAND, 2 * NHAND],
                 [np.float16, np.float32, np.float32])

    total = args.minutes * 60.0
    warm = total * args.warm_frac
    t0 = time.time()
    epoch, phase, log = 0, "greedy", []
    # Bootstrapped value learning is not monotone: keep the checkpoint that
    # actually measured best against the fixed Greedy reference rather than
    # whichever weights happen to be live when the clock runs out.
    best = {"score": -1.0, "t": 0.0, "state": None}
    # Gate rarely: each gate is minutes of eval, so only big runs bother. The
    # first gate is `gate_every` seconds into the ReBeL phase; a run shorter
    # than that has no gates and ships the latest weights (best stays None).
    next_gate = warm + args.gate_every
    gate_curve = []
    # The marker-differential payoff at the horizon distorts the game being
    # solved, so it is annealed away as soon as horizon games become rare, and
    # evaluation always runs on the real game (value 0).
    cap_v = args.cap_value
    warchest.set_cap_value(cap_v)
    probe = None
    print(f"[cfg] FEAT={FEAT} NHAND={NHAND} hidden={args.hidden} depth={args.depth} "
          f"iters={args.iters} budget={total:.0f}s warm={warm:.0f}s device={dev} "
          f"draft={'random' if args.random_draft else 'starter'}", flush=True)

    while True:
        el = time.time() - t0
        if el >= total:
            break
        if phase == "greedy" and el >= warm:
            # Freeze the warm-started network into slot 1: the initial
            # checkpoint the ReBeL phase has to beat.
            value.push(1)
            torch.save({"value": value.state_dict(), "hidden": args.hidden},
                       f"{args.out}/ckpt_init.pt")
            # Drop the warm-phase data. Its job was to initialise the *network*,
            # not to serve as bootstrap targets: it comes from a different
            # policy and its targets are not bootstrapped. Keeping it is
            # actively harmful because the warm phase outproduces ReBeL by
            # ~20x, so it dominates the buffer for the whole run -- at depth 2
            # a 9-minute ReBeL phase contributed 7% of the buffer and the
            # network simply kept fitting greedy play (`runs/diagC`,
            # final_vs_init 0.478, i.e. no improvement at all).
            buf.clear()
            phase = "rebel"
            print(f"[t={el:6.1f}s] --- initial checkpoint saved, switching to ReBeL ---",
                  flush=True)

        tg = time.time()
        kw = dict(random_draft=args.random_draft)
        if phase == "greedy":
            d = warchest.gen_data(args.warm_games, args.seed * 1_000_003 + epoch, "greedy",
                                  temp=args.temp, eval_mix=args.eval_mix, **kw)
        else:
            d = warchest.gen_data(args.rebel_games, args.seed * 1_000_003 + epoch, "rebel",
                                  depth=args.depth, iters=args.iters, explore=args.explore,
                                  **kw)
        gen_s = time.time() - tg
        # Utilities live in [-1, 1]; so does the true value function, so clip
        # the bootstrapped targets to that range.
        vy = np.clip(unpack(d, "vy", 2 * NHAND), -1.0, 1.0)
        vx, vm = unpack(d, "vx", FEAT), unpack(d, "vm", 2 * NHAND)
        buf.add([vx.astype(np.float16), vy, vm])
        # A frozen slice of positions from the warm phase. If the network's
        # spread on these collapses, the value function has gone degenerate --
        # the failure mode a falling training loss hides.
        if probe is None and len(vx) >= 2048:
            probe = torch.as_tensor(vx[:2048], device=dev)
        tgt = vy[vm > 0]
        tgt_mean, tgt_std = float(tgt.mean()), float(tgt.std())

        tt = time.time()
        # Hold a fixed train:generation sample ratio (the reference's
        # `train_gen_ratio: 4`) instead of a fixed step count. The step count
        # then tracks how much fresh data the generator actually produced, which
        # is what keeps the ratio stable across depths -- a fixed count swings
        # the ratio by ~18x between depth 1 and depth 2, and over-trains the
        # thin first epochs after the buffer is cleared.
        steps = max(1, round(args.train_gen_ratio * len(vx) / args.batch))
        lv = train_steps(value, opt, buf, steps, args.batch, rng, dev)
        train_s = time.time() - tt
        value.push(0)
        with torch.no_grad():
            probe_std = float(value(probe).std()) if probe is not None else float("nan")

        # Anneal the horizon payoff to zero on a fixed schedule over the first
        # `anneal_frac` of the ReBeL phase. It must not react to the observed
        # horizon rate: paying out a marker differential at the cap makes
        # stalling a winning line, so a feedback rule that raises the payoff
        # when games stop finishing reinforces exactly the failure it sees.
        # Once it reaches zero the agent is solving the real game, where the
        # clock running out is a draw and only a real win scores.
        if phase == "rebel":
            span = max(args.anneal_frac * (total - warm), 1.0)
            cap_v = args.cap_value * max(0.0, 1.0 - (el - warm) / span)
            warchest.set_cap_value(cap_v)

        # Periodic gate against the fixed reference opponent. Always scored on
        # the real game (horizon payoff 0, so running out the clock is a draw)
        # regardless of what the generator is currently training against --
        # otherwise gate scores drift with the anneal and checkpoint selection
        # would prefer whichever weights exploit the shaped payoff best.
        gate = None
        if phase == "rebel" and time.time() - t0 >= next_gate:
            warchest.set_cap_value(0.0)
            w, l, dr = warchest.eval_match(args.gate_games, 900 + epoch, "rebel", "greedy",
                                           depth=args.depth, iters=args.iters, temp=args.temp,
                                           slot_a=0, random_draft=args.random_draft)
            warchest.set_cap_value(cap_v)
            gate = (w + 0.5 * dr) / max(w + l + dr, 1)
            gate_curve.append({"t": round(time.time() - t0, 1), "score": round(gate, 3)})
            if gate > best["score"]:
                best = {"score": gate, "t": round(time.time() - t0, 1),
                        "state": {k: v.detach().cpu().clone() for k, v in
                                  value.state_dict().items()}}
            next_gate = time.time() - t0 + args.gate_every

        dec = max(d["decisions"], 1)
        rec = {"t": round(time.time() - t0, 1), "epoch": epoch, "phase": phase,
               "games": d["games"], "decisions": dec, "loss": round(lv, 5),
               "cap_frac": round(d["cap_hits"] / max(d["games"], 1), 3),
               "configs": round(d["configs"] / dec, 1), "cap_value": round(cap_v, 4),
               "steps": steps,
               "tgt_mean": round(tgt_mean, 4), "tgt_std": round(tgt_std, 4),
               "probe_std": round(probe_std, 4),
               "gen_s": round(gen_s, 2), "train_s": round(train_s, 2), "buf": len(buf)}
        log.append(rec)
        print(f"[t={rec['t']:6.1f}s] {phase:6s} ep{epoch:3d} games={rec['games']:4d} "
              f"dec={dec:6d} cap={rec['cap_frac']:.2f} cfgs={rec['configs']:5.1f} "
              f"L={lv:.5f} tgt={tgt_mean:+.3f}/{tgt_std:.3f} pstd={probe_std:.3f} "
              f"capv={cap_v:.3f} gen={gen_s:.1f}s train={train_s:.1f}s"
              + (f"  GATE vs greedy {gate:.3f}{'  *best*' if gate >= best['score'] else ''}"
                 if gate is not None else ""), flush=True)
        epoch += 1

    if best["state"] is not None:
        value.load_state_dict(best["state"])
        value.push(0)
        print(f"\nselected checkpoint from t={best['t']}s (gate score {best['score']:.3f})",
              flush=True)
    torch.save({"value": value.state_dict(), "hidden": args.hidden}, f"{args.out}/ckpt_final.pt")
    with open(f"{args.out}/log.json", "w") as f:
        json.dump({"epochs": log, "gate": gate_curve, "best": {k: best[k] for k in ("score", "t")}},
                  f, indent=1)

    # ------------------------------------------------------------- evaluation
    warchest.set_cap_value(0.0)
    print(f"\n=== evaluation on the real game (horizon payoff 0; training ended at "
          f"{cap_v:.3f}) ===", flush=True)
    n = args.eval_games
    kw = dict(depth=args.depth, iters=args.iters, temp=args.temp,
              random_draft=args.random_draft)

    def report(name, res):
        w, l, dr = res
        tot = max(w + l + dr, 1)
        score = (w + 0.5 * dr) / tot
        se = (score * (1 - score) / tot) ** 0.5
        print(f"{name:38s} W{w:4d} L{l:4d} D{dr:4d}   score {score:.3f} +- {2*se:.3f}",
              flush=True)
        return score

    r = {}
    r["final_vs_greedy"] = report("final checkpoint vs Greedy",
                                  warchest.eval_match(n, 303, "rebel", "greedy", slot_a=0, **kw))
    r["final_vs_init"] = report("final checkpoint vs initial checkpoint",
                                warchest.eval_match(n, 101, "rebel", "rebel",
                                                    slot_a=0, slot_b=1, **kw))
    with open(f"{args.out}/eval.json", "w") as f:
        json.dump(r, f, indent=1)

    ok = r["final_vs_init"] > 0.5 and r["final_vs_greedy"] > 0.5
    print(f"\nGOAL: the run produced a checkpoint better than the initial one that also "
          f"beats Greedy -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("      (ReBeL self-play must carry the warm-started network past its own "
          "start; final_vs_init is the headline)", flush=True)


if __name__ == "__main__":
    main()
