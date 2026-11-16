"""Summarize opt-in native events without conflating job wall sum and elapsed.
Usage: python analyze.py /path/to/continuation-24
"""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows, curves = [], []
for run in ['pilot-geometric', 'full-size-tiered', 'full-geometric']:
    directory = root / run
    if not (directory / 'summary.json').exists():
        continue
    summary = json.loads((directory / 'summary.json').read_text())
    events = [json.loads(line) for line in (directory / 'native-events.jsonl').read_text().splitlines()]
    for system, key in [('static', 'static_lmi'), ('dli', 'temporal_dli')]:
        attempts = [e for e in events if (e['optimizer'] == 'dli_research') == (system == 'dli')]
        published = [e for e in attempts if e['outcome'] == 'published']
        rewritten = sum(e['destination_points'] for e in published)
        wall = sum(e['job_wall_micros'] for e in attempts) / 1e6
        span = (max(e['started_unix_micros'] + e['job_wall_micros'] for e in attempts)
                - min(e['started_unix_micros'] for e in attempts)) / 1e6 if attempts else 0
        m = summary[key]
        row = dict(run=run, system=system, jobs=len(published), unsuccessful_attempts=len(attempts)-len(published),
                   destination_vectors=rewritten, rewrite_amplification=rewritten/summary['vectors'],
                   job_wall_sum_s=wall, first_start_to_last_completion_s=span,
                   recall10=m['recall10'], recall30=m['recall30'], p50_ms=m['query_p50_ms'], p95_ms=m['query_p95_ms'],
                   scorer_calls=m['batch_scorer_work_calls_attempted_successful'][0],
                   attempted_vectors=m['batch_scorer_work_calls_attempted_successful'][1],
                   successful_vectors=m['batch_scorer_work_calls_attempted_successful'][2],
                   auxiliary_bytes=m['auxiliary_index_bytes'],
                   post_update_learned=m.get('learned_queries_after_updates'),
                   post_update_fallback=m.get('exact_fallback_queries_after_updates'))
        rows.append(row)
        cumulative, cumulative_wall = 0, 0
        for i, e in enumerate(sorted(published, key=lambda x:x['started_unix_micros']), 1):
            cumulative += e['destination_points']
            cumulative_wall += e['job_wall_micros']/1e6
            curves.append(dict(run=run, system=system, job=i, cumulative_destination_vectors=cumulative,
                               cumulative_job_wall_s=cumulative_wall, output_uuid=e['destination']))
for name, values in [('comparison.csv', rows), ('work_curves.csv', curves)]:
    if values:
        with (root/name).open('w') as f:
            writer=csv.DictWriter(f, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
(root/'comparison.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ['cumulative_destination_vectors', 'cumulative_job_wall_s']:
        fig, ax=plt.subplots()
        for run in ['full-size-tiered','full-geometric']:
            for system in ['static','dli']:
                subset=[r for r in curves if r['run']==run and r['system']==system]
                if subset: ax.plot([r['job'] for r in subset], [r[metric] for r in subset], marker='o',label=run+' '+system)
        ax.set_xlabel('Completed native optimization jobs');ax.set_ylabel(metric);ax.legend();fig.tight_layout()
        fig.savefig(root/(metric+'.svg'));plt.close(fig)
except ImportError:
    print('matplotlib unavailable: CSV curves retained',file=sys.stderr)
