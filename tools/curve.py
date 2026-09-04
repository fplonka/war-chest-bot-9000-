import json
import math
import random
from statistics import fmean
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2)


def reports(branch):
    if branch == "master":
        return [ROOT / "runs/master_s2/comparisons/master_s1.json"]
    return [ROOT / f"runs/{branch}_s{c}/comparisons/master_s{b}.json"
            for c in SEEDS for b in SEEDS]


def points(branch):
    by_minute = {}
    for report in reports(branch):
        for pair in json.loads(report.read_text())["pairs"]:
            won = [0.5 * (1 + (x < 0) - (x > 0))
                   for game in pair["color_pairs"] for x in game]
            by_minute.setdefault(int(round(pair["b_minutes"])), []).extend(won)
    return dict(sorted(by_minute.items()))


def elo(score):
    return 400.0 * math.log10(score / (1.0 - score))


def line(minutes, elos):
    xs = [math.log2(m / 60.0) for m in minutes]
    mx, my = sum(xs) / len(xs), sum(elos) / len(elos)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, elos))
             / sum((x - mx) ** 2 for x in xs))
    return slope, my - slope * mx


def main():
    branch = sys.argv[1]
    won = points(branch)
    minutes = list(won)
    elos = [elo(fmean(v)) for v in won.values()]
    slope, offset = line(minutes, elos)
    rng = random.Random(0)
    slopes = sorted(
        line(minutes, [elo(fmean(rng.choices(v, k=len(v)))) for v in won.values()])[0]
        for _ in range(1000))
    low, high = slopes[25], slopes[975]
    cells = [branch, *(f"{e:+.0f}" for e in elos), f"{offset:+.0f}",
             f"{slope:+.0f} [{low:+.0f}, {high:+.0f}]"]
    print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
