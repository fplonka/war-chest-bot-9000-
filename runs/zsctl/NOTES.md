# zsctl

Control for the zero-sum work: golden8 knobs, no projection, `zero_sum_w=0`.
The first run with `probe_zs` — the RMS of `m_0 + m_1` on the fixed probe
batch, measured on the raw network.

Horizon 6.9% over the last quarter, `tgt_std` 0.369, 1,081 balanced solves/s.
The violation is **flat for the whole run**: 0.055 → 0.058 while the value
spread grows 0.183 → 0.258. It does not decay, and the reason is structural —
the target is a backup of the network's own leaf values, so a constant added
to the network comes straight back in its own targets. The network sits on
those targets (belief-weighted mean error +0.0007), the per-config loss has no
gradient along the constant direction, and every constant is an exact fixed
point of the bootstrap.
