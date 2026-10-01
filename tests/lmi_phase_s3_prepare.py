"""Stream a local fvecs corpus into the opt-in native build benchmark's format."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--metric', choices=['Euclid', 'Cosine', 'Dot', 'Manhattan'], default='Euclid')
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
digest = hashlib.sha256()
count = 0
with args.source.open('rb') as src, (args.output / 'corpus.f32').open('xb') as dst:
    header = src.read(4)
    if len(header) != 4:
        raise ValueError('missing dimension header')
    dimension = struct.unpack('<I', header)[0]
    if not 1 <= dimension <= 65536:
        raise ValueError('unsupported dimension')
    src.seek(0)
    row_bytes = 4 * (dimension + 1)
    while block := src.read(row_bytes * 1024):
        if len(block) % row_bytes:
            raise ValueError('truncated fvecs input')
        digest.update(block)
        for offset in range(0, len(block), row_bytes):
            if struct.unpack_from('<I', block, offset)[0] != dimension:
                raise ValueError('inconsistent fvecs dimensions')
            dst.write(block[offset + 4:offset + row_bytes])
            count += 1
(args.output / 'dataset.json').write_text(json.dumps({
    'source': str(args.source.resolve()), 'source_sha256': digest.hexdigest(),
    'corpus_count': count, 'dimension': dimension, 'metric': args.metric,
    'export_dtype': 'little-endian float32', 'order': 'original fvecs order',
}, indent=2))
