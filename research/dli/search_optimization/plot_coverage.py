"""Plot offline bucket inclusion, never relabel it as native ANN recall."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('routing_csv', type=Path)
parser.add_argument('output_svg', type=Path)
args = parser.parse_args()
groups = defaultdict(list)
with args.routing_csv.open() as stream:
    for row in csv.DictReader(stream):
        groups[row['policy'], int(row['budget'])].append(row)
fig, ax = plt.subplots(figsize=(6, 4))
for policy in ['slot', 'reserved', 'population_density']:
    points = []
    for (name, budget), rows in groups.items():
        if name == policy:
            points.append(tuple(sum(float(r[key]) for r in rows) / len(rows)
                                for key in ['candidates', 'bucket10']))
    points.sort()
    ax.plot(*zip(*points), marker='o', label=policy)
ax.set(xlabel='Mean selected posting offsets (whole buckets)',
       ylabel='Gold top-10 bucket coverage',
       title='Frozen geometric 10M: offline coverage, 100 queries')
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(args.output_svg)
