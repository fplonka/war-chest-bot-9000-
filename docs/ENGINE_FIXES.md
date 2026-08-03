# Engine fixes from the census replay (verify/replay.py)

Every rule change made to `engine/src` during census verification, one line
each: symptom -> root cause -> fix. Each has a scenario test in
`engine/tests/scenarios.rs` (section "census-replay fixes"). After all fixes:
1112/1112 in-scope games (93,566 actions) replay end-to-end clean.

## Engine bugs

1. **Royal Guard tactic missing with only the Royal Coin in hand.** Symptom:
   "moves the Royal Guard (Royal Coin)" entries unmatched. Root cause:
   tactics were listed only for unit types with a coin in hand, but the RG
   tactic is paid with the Royal COIN, not an RG coin. Fix: list the RG
   tactic under the Royal-Coin hand branch of `list_main_play` (and of the
   Warrior-Priest forced play).
2. **Royal Guard tactic discarded the Royal Coin facedown.** Symptom: zone
   diff (server faceup vs engine facedown) on every RG-tactic entry. Root
   cause: wrong discard pile. Fix: the tactic discard is FACE-UP; only
   facedown *actions* (claim/recruit/pass) hide the Royal Coin.
3. **Footman V2 recruit-deploy held the recruited coin in hand.** Symptom:
   "recruits a Footman" entries unmatched (server: coin in `recruited`
   state, i.e. discard). Root cause: engine moved the recruited coin
   faceup->hand before the instant-deploy decision. Fix: the coin stays in
   the face-up discard; `FootmanInstantDeploy` deploys it from there.
4. **Footman tactic forced a fixed maneuver order.** Symptom: "moves the
   Footman" board diffs when the player maneuvered the second footman (in
   hex order) first. Root cause: `Cont::FootmanManeuver` queued one hex at a
   time. Fix: the node now holds the set of remaining footman hexes and
   offers maneuvers for all of them (player-chosen order); each acts once.
5. **Footman tactic only maneuvered same-version footmen.** Symptom: free
   footman maneuvers of the other version unmatched (board-only diffs).
   Root cause: hexes collected by the played coin's exact type. Fix: a
   Footman coin's tactic maneuvers Footman AND Footman V2 units
   (ambiguity 2 adjudicated against the engine's assumption).
6. **Warrior Priest draw deferred until after the defender's RG soak.**
   Symptom: 25 games mismatched at "Warrior Priest attacks the Royal Guard"
   (drawn coin already visible in the attack entry's snapshot). Root cause:
   the WP draw cont was queued behind the RoyalGuardChoice interrupt. Fix:
   the draw (chance node) resolves BEFORE the defender's choice; the forced
   play still resolves after it (`WarriorPriestDraw` carries `rg_hex` and
   re-installs the `RoyalGuardChoice` on apply).
7. **Berserker chain attack vs Knight used pre-payment height.** Symptom:
   offer/take report: 17 offers, 0 takes for chained attacks on a Knight at
   stack height 2 (post-payment height 1), vs plentiful takes at height 3+.
   Root cause: legality listing checked the stack height before the chain
   cost was discarded, while `apply` discards the cost first. Fix: the
   chain's Knight-immunity check uses height-after-payment (ambiguity 1
   confirmed: cost is paid before the chained maneuver).
8. **Ensign could grant itself a move.** Evidence: 0 of 8,999 "(Ensign)"
   moves in the full 9,108-game log census move the Ensign itself (a
   self-move would carry the "(Ensign)" label suffix). A self-move is also a
   pure transposition of the plain Move. Fix: self-target removed
   (ambiguity 3 adjudicated against the engine's assumption).
9. **Marshal could direct itself.** Evidence: 0 of 3,490 "(Marshal)" attacks
   are by the Marshal itself; also a pure transposition of the plain
   Attack. Fix: self-target removed.

## Ambiguity adjudications (engine/src/rules.rs `// AMBIGUITY:` notes)

1. Berserker chain cost discarded BEFORE the chained maneuver: **confirmed**
   (fix 7).
2. Footman tactic same-version only: **refuted** — both versions (fix 5).
3. Ensign may target itself: **refuted** (fix 8).
4. Footman tactic maneuvers mandatory-if-able, no decline: **consistent
   with all 1,112 replays** (no game required a decline; where a plain
   move/attack transposes with a partial tactic, replays resolve via
   lookahead and never contradict the mandatory reading). Kept.
5. Royal Guard tactic is 1-2 steps, not 0: **confirmed** (takes at both 1
   and 2 steps; no zero-step move ever needed to explain a replay).

## Server behaviors documented for the replay driver (not engine changes)

- Cavalry tactic (move+attack) is ONE log entry; the footman tactic's coin
  play + first maneuver share one entry; later footman maneuvers are their
  own entries.
- An attack deferred by the defender's RG soak choice shows no target-side
  effect at the attack entry; the soak is its own entry ("removes a Royal
  Guard from the supply/board") in both directions of the choice.
- Empty-hand players still get logged "passes" entries with no state change
  (shortage case); the round-boundary refill happens only after them, and
  the server's `round` counter advances only at the refill.
- The terminal snapshot includes the post-win round refill; the engine
  stops at the win (compared via a refill-invariant union projection).
- The server's `recruited` coin state behaves exactly like the face-up
  discard (rejoins the bag on refill); projections merge the two.
- A Warrior Priest trigger with an empty bag AND empty discards recycles
  the just-played coin (server holds it "in play"); the engine's
  discard -> refill -> forced-draw semantics reproduces the identical
  states (verified in game 2458407).
