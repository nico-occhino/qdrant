# Exploratory planner simulation behind MULTILEVEL_DIAGNOSIS.md (2026-11-22). Pure arithmetic, no data.
# Bentley-Saxe (E1) plus "sink consolidation": a level i above the deepest non-empty level D
# may hold at most rho * live(D); a compaction that would exceed it merges levels 0..i into D
# (reusing D's router, no training). Reports final levels, rewrites, a bucket-cost proxy.
import sys
b,a,N=5000,10,10_120_191
def run(rho):
    lv={}; written=0; sinks=0; fresh=0
    n=0
    while n+b<=N:
        n+=b; inc,t=b,1
        while True:
            cap=b*a**t; occ=lv.get(t,0)
            if inc<=cap-occ: break
            inc+=occ; t+=1
        D=max(lv) if lv else None
        if rho is not None and D is not None and t<D and inc+lv.get(t,0)>rho*lv[D] and b+sum(v for k,v in lv.items() if k<D)+lv[D]<=b*a**D:
            # consolidate everything above D (levels < D, including t) into D
            moved=b+sum(v for k,v in lv.items() if k<D)
            for k in [k for k in lv if k<D]: del lv[k]
            lv[D]+=moved; written+=lv[D]; sinks+=1
            continue
        out=inc+lv.get(t,0)
        if lv.get(t,0)==0: fresh+=1
        for k in [k for k in lv if k<t]: del lv[k]
        lv[t]=out; written+=out
    tot=sum(lv.values())
    # proxy: expected bucket size paid per neighbour = sum_l share_l * live_l/a^l (Plain counts whole)
    cost=sum(v/tot*(v/a**k) for k,v in lv.items())+ (N-n)/tot*(N-n)
    return dict(sorted(lv.items())),written/N,sinks,fresh,cost
base=run(None)
print("rho    final levels                                  write_amp sinks fresh  bucket-cost proxy (vs E1)")
for rho in [None,1.0,0.5,0.25,0.1,0.05]:
    lv,wa,s,f,c=run(rho); print(f"{str(rho):5s} {str(lv):48s} {wa:6.1f} {s:5d} {f:5d}  {c:7.0f} ({c/base[4]:.2f}x)")
print("matched static: 10,120,191/11,100 =", round(N/11100))
