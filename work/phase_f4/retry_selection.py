from pathlib import Path
import shutil

root = Path('/home/nicoo/work')
source = root / 'lmi-phase-f4-data'
attempt = root / 'lmi-phase-f4-attempt1-selection-bug'
assert source.is_dir() and not attempt.exists()
source.rename(attempt)
source.mkdir()
shutil.copy2(attempt / 'protocol.json', source / 'protocol.json')
(source / 'verification').mkdir()
print(f'Preserved first F.4 attempt at {attempt}')
