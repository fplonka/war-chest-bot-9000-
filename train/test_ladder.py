import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import exp
import ladder
import report


class LadderGraphTest(unittest.TestCase):
    def test_linear_graph(self):
        ref = {"run": None, "endpoint": False, "final": False, "order": -1}
        players = [ref]
        runs = ["c1", "a1", "c2", "a2"]
        for run in runs:
            for order in range(3):
                players.append({"run": run, "order": order,
                                "endpoint": order in (0, 2), "final": order == 2})
        comparisons = {frozenset(("c1", "a1")), frozenset(("c2", "a2"))}
        edges = {}
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                n = ladder.pairing_games(players[i], players[j], comparisons, 100, 1000)
                if n:
                    edges[(i, j)] = n
        self.assertEqual(sum(n == 100 for n in edges.values()), 16)
        self.assertEqual(sum(n == 1000 for n in edges.values()), 2)
        seen = {0}
        while True:
            more = {j for i, j in edges if i in seen} | {i for i, j in edges if j in seen}
            if more <= seen:
                break
            seen |= more
        self.assertEqual(seen, set(range(len(players))))
        games = np.zeros((len(players), len(players)))
        score = np.zeros_like(games)
        for (i, j), count in edges.items():
            games[i, j] = games[j, i] = count
            score[i, j] = score[j, i] = count / 2
        self.assertTrue(np.isfinite(ladder.fit_elo(games, score)).all())

    def test_comparisons_come_from_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            runs = []
            for seed in (1, 2):
                for arm, control in (("base", True), ("candidate", False)):
                    run = pathlib.Path(root, f"{arm}-{seed}")
                    run.mkdir()
                    run.joinpath("log.json").write_text(json.dumps({"cfg": {
                        "experiment": "test", "seed": seed,
                        "arm": arm, "is_control": control,
                    }}))
                    runs.append(str(run))
            self.assertEqual(
                {frozenset(x) for x in exp.comparisons_from_metadata(runs)},
                {frozenset((runs[0], runs[1])), frozenset((runs[2], runs[3]))},
            )

    def test_report_filters_shared_ladder_by_run(self):
        players = [
            {"name": "greedy", "run": None, "t": None},
            {"name": "base.init", "run": "runs/base", "t": 1},
            {"name": "arm.init", "run": "runs/arm", "t": 1},
        ]
        run = {"name": "base", "path": "runs/base",
               "ladder": {"players": players}}
        self.assertEqual([p["name"] for p in report.checkpoints(run)], ["base.init"])


if __name__ == "__main__":
    unittest.main()
