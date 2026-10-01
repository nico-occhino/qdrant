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
    # Lifecycle test compares model identities, not individual weights.
    state['router'] = hashlib.sha256(path.with_name('lmi_router.bin').read_bytes()).hexdigest()
    return state
