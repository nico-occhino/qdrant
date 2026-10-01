"""Small-fixture inspection of native LMI v1/v2 state for lifecycle tests.

This is benchmark/test tooling, never the production reader. Large-scale tests
should stream or hash files instead of materializing the returned posting lists.
"""
import hashlib
import json
import struct


def read_lmi_state(path):
    state = json.loads(path.read_text())
    if state['version'] == 1:
        return state
    if state['version'] != 2:
        raise ValueError('unsupported LMI state version')
    data = path.with_name('lmi_postings.bin').read_bytes()
    (size,) = struct.unpack_from('<Q', data, 0)
    if size < 1 or size > (len(data) - 16) // 8:
        raise ValueError('invalid boundary count')
    boundaries = struct.unpack_from(f'<{size}Q', data, 8)
    (count,) = struct.unpack_from('<Q', data, 8 + size * 8)
    if len(data) != 16 + size * 8 + count * 4:
        raise ValueError('invalid posting file size')
    if boundaries[0] or boundaries[-1] != count or any(a > b for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('invalid posting boundaries')
    points = struct.unpack_from(f'<{count}I', data, 16 + size * 8)
    state['postings'] = [list(points[a:b]) for a,b in zip(boundaries,boundaries[1:])]
    # Exclude sampled offsets: a changed sample alone does not prove a new model.
    model = path.with_name('lmi_router.bin').read_bytes()
    (sample_count,) = struct.unpack_from('<Q', model, 0)
    router_start = 8 + 4 * sample_count
    if router_start >= len(model):
        raise ValueError('invalid router file size')
    state['router'] = hashlib.sha256(model[router_start:]).hexdigest()
    return state


def state_digest(path):
    """Bind smoke-test identity to metadata AND all auxiliary binary files."""
    state = json.loads(path.read_text())
    files = [path]
    if state['version'] == 2:
        files += [path.with_name('lmi_router.bin'), path.with_name('lmi_postings.bin')]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()
