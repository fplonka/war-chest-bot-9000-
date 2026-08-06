# War Chest engine specification (source of truth)

Everything here is verified against warchestonline.com data (snapshots/logs in
`../data/`), the 2021 AEG rulebook, official card texts extracted from the
site client, and the official FAQ. Where any source disagrees with observed
server behavior, the server wins.

## 1. Scope

2-player ranked configuration: base set, random draft, `royalGuardType: true`.
Implement 19 unit types + the Royal Coin. All are official AEG cards (the V2s
are AEG's revised cards; the site's random draft deals V1 and V2 versions as
distinct units in one pool, so both appear in ranked games).

unitTypeId map (from the site; the engine must key units by these ids):
1 Archer, 2 Berserker, 3 Cavalry, 4 Crossbowman, 5 Ensign, 6 Footman,
7 Knight, 8 Lancer, 9 Light Cavalry, 10 Marshal, 11 Mercenary, 12 Pikeman,
16 Scout, 17 Swordsman, 18 Warrior Priest, 19 Royal Guard (Nobility revision),
52 Berserker V2, 53 Footman V2, 54 Warrior Priest V2.
Royal Coin: unitTypeId 13 (one faction) and 14 (the other); same rules.
Games containing any other unitTypeId are out of scope (~1% of census).

Coin counts (x on card): Archer 4, Berserker 5, Cavalry 4, Crossbowman 5,
Ensign 5, Footman 5, Knight 4, Lancer 4, Light Cavalry 5, Marshal 5,
Mercenary 5, Pikeman 4, Scout 5, Swordsman 5, Warrior Priest 4,
Royal Guard 5, Berserker V2 5, Footman V2 5, Warrior Priest V2 4.
Royal Coin: 1 per player.

## 2. Board (verified from snapshot positions)

Axial hex coordinates, written "x,y" in all site data. Neighbor offsets:
(+1,0), (-1,0), (0,+1), (0,-1), (+1,-1), (-1,+1).
Hex distance = axial distance: (|dx| + |dy| + |dx+dy|) / 2.

Playable 2p board: the 37 hexes with |x-3|<=3 and |y-3|<=3 and |x+y-6|<=3
(radius-3 hexagon centered on (3,3)).

Locations (10): white start (4,0), (6,1); black start (0,5), (2,6);
neutral (2,1), (3,2), (5,3), (1,3), (3,4), (4,5).
(The coords (1,0), (0,2), (5,6), (6,4) are 4-player-only; not on this board.)

Each player has 6 control markers. At setup, white's markers sit on (4,0) and
(6,1); black's on (0,5) and (2,6); 4 markers in hand each. A player wins
immediately when all 6 of their markers are on the board. Capturing an
opponent location returns their marker to them.

## 3. Zones and setup

Per player: bag, hand (max 3), discard_faceup, discard_facedown, supply,
plus units on board (stacks) and eliminated (out of game).
Setup: each player has 4 drafted unit types; 2 coins of each type + 1 Royal
Coin into the bag (9 coins); remaining coins to supply. The engine takes the
draft (8 unitTypeIds + turn order + initiative holder) as input.

## 4. Round flow

1. Both players draw to 3 coins in hand from their own bag. If a bag empties
   during a draw, refill it by shuffling in the whole discard pile (both
   face-up and facedown coins), then continue drawing.
2. Starting with the initiative holder, players alternate playing exactly one
   coin per turn, resolving one action per coin, until both hands are empty
   (6 actions per round total, 3 each). If one player runs out of coins
   (rare shortage case), the other continues alone until their hand empties.
3. Next round. The initiative-marker holder at round end acts first.

## 5. Actions (one coin each)

Place face-up on board:
- **Deploy**: place a unit coin from hand onto an empty location you control.
  Max one unit of that type on board (Footman/Footman V2: two). Cannot deploy
  if you already have that unit on board (subject to the Footman exception).
  Scout may instead deploy onto any empty hex adjacent to a friendly unit.
- **Bolster**: place a unit coin from hand on top of the matching deployed
  unit's stack. No stack height limit.

Discard facedown (any coin, including Royal Coin; identity hidden from
opponent until reshuffle reveals nothing — facedown discards are never
revealed):
- **Claim initiative**: take the marker. Illegal if you hold the marker or
  held it earlier this round (it transfers at most once per round).
- **Recruit**: choose any coin type from your supply, reveal it, put it into
  your discard pile FACE-UP. (The spent coin goes facedown; the recruited
  coin is public.)
- **Pass**: no effect. Passing does not force later passes.

Discard face-up (unit coins only; "maneuvers"):
- **Move**: move the matching unit (its whole stack) one step to an adjacent
  empty hex.
- **Control**: the matching unit stands on a location: if neutral, place your
  marker; if opponent-controlled, return their marker and place yours.
  Illegal on a location you already control. Winning check happens here.
- **Attack**: the matching unit attacks an adjacent enemy unit. Remove one
  coin from the target stack; that coin is ELIMINATED (out of the game, never
  returns). If it was the last coin, the unit leaves the board (its type may
  be redeployed later).
- **Tactic**: execute the unit's card tactic (below).

Royal Coin: facedown actions only, plus it is the discard for the Royal
Guard's tactic.

Multi-step moves (Light Cavalry, Lancer, Royal Guard tactic): every
intermediate hex must be empty too; units never pass through occupied hexes.

## 6. Unit cards (site texts, verbatim semantics)

TACTICS (played via the Tactic action with that unit's coin):
- Archer: attack a unit exactly 2 spaces away; the intervening space may be
  occupied. (Restriction: the Archer cannot use the normal Attack action.)
- Cavalry: move (one step), then attack.
- Crossbowman: attack a unit exactly 2 spaces away in a straight line; the
  intervening space must be empty. (Normal attack also allowed.)
- Ensign: choose a friendly unit within 2 spaces of the Ensign; it makes a
  normal one-step move that must end within 2 spaces of the Ensign.
- Footman / Footman V2: perform one maneuver with each Footman unit on the
  board (the two maneuvers are chosen independently: move/control/attack).
  With one Footman on board, the tactic still grants its one maneuver.
- Lancer: move 1 or 2 spaces and then attack, all in one straight line. The
  tactic is legal only if the final attack can actually be made (e.g. cannot
  end adjacent to an unbolstered-attack-immune Knight with no other target in
  line). (Restriction: no normal Attack action.)
- Light Cavalry: move exactly 2 spaces (each step into an empty hex).
- Marshal: choose a friendly unit within 2 spaces of the Marshal; it performs
  a normal attack, if able. Never grants tactic-attacks (no Archer/Lancer).
- Royal Guard: discard the ROYAL COIN (not a RG coin) to move the Royal Guard
  up to 2 spaces, ending on a location you control.

ATTRIBUTES (free/triggered effects):
- Berserker (V1): after the Berserker MANEUVERS (move/control/attack via any
  action, including granted ones), you may maneuver it again by discarding a
  bolstered coin from its stack (stack must keep >= 1 coin). Repeatable while
  coins remain. The discarded coin goes to your discard pile FACE-UP
  (verified from server data: it recycles, it is not eliminated).
- Berserker V2: same, but only after it ATTACKS or MOVES, and the extra
  maneuver is only an attack or a move (control excluded).
- Knight: can only be attacked by units that are bolstered (stack >= 2).
  Attacks by unbolstered units cannot target it. Tactic-attacks (Archer,
  Crossbowman, Cavalry, Lancer) obey the same rule; Pikeman's reflex is not
  an attack and still hits it.
- Mercenary: after you RECRUIT a Mercenary coin while your Mercenary unit is
  deployed, that unit may immediately make one free maneuver.
- Pikeman: when an ADJACENT unit attacks the Pikeman, remove one coin from
  the attacker (eliminated). Simultaneous with the attack, unconditional,
  not an attack itself (ignores Knight immunity). Range-2 attacks
  (Archer/Crossbowman/Lancer-at-distance... note Lancer always ends adjacent)
  trigger it only when the attacker is adjacent.
- Royal Guard: when the Royal Guard is attacked, you (the defender) may
  remove a Royal Guard coin from your SUPPLY instead of from the unit's
  stack. The removed coin is eliminated (verified from server data).
- Scout: may be deployed onto any empty hex adjacent to any friendly unit
  (in addition to the normal deploy rule).
- Swordsman: after it attacks, it may make a free one-step move.
- Warrior Priest: after it ATTACKS or CONTROLS, draw one coin from your bag
  (refill-shuffle applies if empty) and IMMEDIATELY use it as a normal coin
  play (any action; pass always legal). The forced play must use the drawn
  coin, so the Royal Guard tactic — which is a play of the Royal Coin, not of
  a drawn RG coin — is offered only when the drawn coin IS the Royal Coin
  (adjudication, 2026; the census never separated this case). The hand never
  transiently exceeds 3: the trigger is always preceded, in the same play
  chain, by a coin play that emptied a hand slot.
- Warrior Priest V2: same, at most once per turn. Only V2's own trigger
  counts against the cap: a V1 trigger earlier in the turn does not block V2.
- Footman V2 extra attribute: after you RECRUIT a Footman coin while you
  already have a Footman unit deployed, you may immediately deploy the
  recruited coin (normal deploy legality) instead of putting it in discard.

FAQ rulings that bind the implementation:
- Attribute triggers cascade through granted actions: Ensign/Marshal grants
  and Warrior Priest coin plays trigger Berserker/Swordsman/Warrior Priest
  attributes normally.
- When several attributes trigger from one event, the active player chooses
  resolution order; each action fully resolves before the next starts.
- Berserker chain: one coin discard per extra maneuver; no hand coin needed.
- A unit is never adjacent to itself.

## 7. Engine architecture requirements

- Rust crate at `engine/`, library + pyo3 module named `warchest` (abi3),
  plus a small native benchmark binary.
- Flat state: 37 hexes indexed 0..36; per hex: occupant unit-type id, owner,
  stack height. Per player: multiset counts per zone per unit type (bag,
  hand, faceup discard, facedown discard, supply, eliminated), markers on
  board, initiative flags (holder + whether it moved this round), turn info.
- The game is a sequence of DECISION POINTS. Model every choice — including
  triggered/optional micro-decisions — as an explicit decision node:
  - active player's coin play (coin + action + parameters),
  - defender's Royal Guard supply choice during attack resolution,
  - optional post-triggers (Swordsman move, Berserker chain, Mercenary free
    maneuver, Footman V2 instant deploy, Warrior Priest forced coin use),
  - chance nodes: hand draws at round start and Warrior Priest draws.
  API: `legal_actions(state) -> Vec<Action>` (whoever is to act, including
  chance), `apply(state, action) -> state`. For chance nodes,
  `legal_actions` returns the possible draws with multiplicities so replay
  can force observed draws and search can enumerate exactly.
- Determinism: no hidden RNG inside apply; all randomness enters through
  chance-node actions.
- Zobrist-style or derived hash is nice-to-have, not required.
- Performance: >= 100k apply calls/sec/core on the M1 (measure with the
  bench binary; random playouts from random drafts).
- Python API (pyo3): construct from draft, step, legal actions, tensor-ready
  observation export (leave observation minimal for now: the ReBeL trainer
  defines its own encoding from state accessors).

## 8. Tests the engine must ship with

1. Scenario tests for every FAQ ruling above and every tactic/attribute
   (hand-built positions, assert exact legal-action sets and post-states).
2. Invariants under 10k random playouts: per-type coin conservation
   (bag+hand+discards+supply+board+eliminated == initial total), stack >= 1
   for deployed units, one unit per hex, board hexes valid, game terminates
   (site data: max observed ~200 actions; hard cap 1000 with draw=loss for
   neither—flag if hit), markers 0..6 and win detection correct.
3. The census replay (built by the verify stage, `verify/`) is the
   acceptance test; the engine ships hooks for it: forcing chance outcomes
   and dumping full state for diffing.

## 9. Replay data facts (for verify/)

- `data/logs.jsonl`: `{gameId, log:[{id, label, color, snapshotId}]}`,
  NEWEST-FIRST; sort by `id` ascending for play order.
- `data/snapshots.jsonl.gz`: gzip members, one JSON object per game:
  `{gameId, total, snapshots:[...]}`. Sort snapshots by `snapshotId`.
  There are len(log)+1 snapshots: `log[k].snapshotId == snapshots[k].id`
  and snapshots[k] is the state BEFORE action k; the final snapshot is the
  terminal state.
- Snapshot units: every coin as a record with `state` in {in_bag, in_hand,
  in_supply, deployed, discarded_faceup, discarded_facedown, eliminated},
  `position` ("x,y" or null), stacks via `parentUnitId`/`bolsterUnits`.
  Both players' hidden zones are fully visible (third-party fetch).
- IGNORE `players[].team.controlledBases` and `players[].hasInitiative` in
  snapshots: they are live-joined final values, not historical. The
  top-level `teams[].controlledBases` IS historical per snapshot.
- Label conventions: "P deploys a X", "P bolsters the X", "P moves the X"
  (suffix "(Ensign)" / "(Royal Coin)" for granted/tactic moves), "P's X
  attacks the Y" (suffix "(Marshal)"), "P's X controls a base", "P recruits
  a X", "P claims the initiative", "P passes", "P removes a Royal Guard from
  the supply", Berserker chain shows as repeated maneuver labels plus coin
  removal reflected in snapshots. Facedown discard identities are visible in
  snapshots even though labels hide them.
