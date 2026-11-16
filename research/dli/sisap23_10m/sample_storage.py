"""Best-effort non-atomic storage sample; sparse/hard-linked files counted once.
Observed maxima are lower bounds on transient peaks, not exact peaks.
"""
import csv,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);output=Path(sys.argv[2]);seen=set();allocated=apparent=files=0
for parent,dirs,names in os.walk(root):
 for name in names:
  try:s=(Path(parent)/name).stat()
  except FileNotFoundError:continue
  inode=(s.st_dev,s.st_ino)
  if inode in seen:continue
  seen.add(inode);allocated+=s.st_blocks*512;apparent+=s.st_size;files+=1
new=not output.exists()
with output.open('a') as f:
 w=csv.writer(f)
 if new:w.writerow(['unix_s','allocated_unique_bytes','apparent_unique_bytes','files'])
 w.writerow([time.time(),allocated,apparent,files])
print('storage sample allocated GiB',round(allocated/2**30,2))
