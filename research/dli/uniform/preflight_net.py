"""P0.4 multi-node reachability (SCALE_SHARED_SPEC.md §P0). Not data.

    preflight_net.py server --ports 6333,6335 --seconds 600
    preflight_net.py client --host H --ports 6333,6335 --mib 1024 --out FILE

The server accepts one connection per port and discards what it receives. The client connects to each port, sends
--mib MiB, and records throughput together with the /proc/net/dev byte deltas of every interface.
"""
import argparse
import json
import socket
import threading
import time
from pathlib import Path


def netdev():
    out = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, rest = line.split(":", 1)
        f = rest.split()
        out[name.strip()] = {"rx": int(f[0]), "tx": int(f[8])}
    return out


def serve(port, seconds):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    s.settimeout(seconds)
    try:
        c, _ = s.accept()
        while c.recv(1 << 20):
            pass
        c.close()
    except socket.timeout:
        pass


def main():
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["server", "client"])
    a.add_argument("--ports", default="6333,6335")
    a.add_argument("--host")
    a.add_argument("--mib", type=int, default=1024)
    a.add_argument("--seconds", type=int, default=600)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    if args.mode == "server":
        ts = [threading.Thread(target=serve, args=(p, args.seconds)) for p in ports]
        [t.start() for t in ts]
        [t.join() for t in ts]
        return
    rec = {"client": socket.gethostname(), "server": args.host, "ports": {}}
    chunk = b"\0" * (1 << 20)
    for p in ports:
        r = {"connected": False}
        for _ in range(60):  # the server may still be starting
            try:
                c = socket.create_connection((args.host, p), timeout=5)
                r["connected"] = True
                break
            except OSError as e:
                r["last_error"] = repr(e)
                time.sleep(2)
        if r["connected"]:
            n0, t = netdev(), time.time()
            for _ in range(args.mib):
                c.sendall(chunk)
            c.close()
            dt, n1 = time.time() - t, netdev()
            r.update(seconds=dt, mib_per_s=args.mib / dt,
                     tx_delta={k: n1[k]["tx"] - n0[k]["tx"] for k in n1 if k in n0 and n1[k]["tx"] > n0[k]["tx"]})
        rec["ports"][p] = r
    sent = args.mib << 20
    rec["checks"] = {f"p0_4_port_{p}": r["connected"] and any(v >= sent for v in r["tx_delta"].values())
                     for p, r in rec["ports"].items()}
    rec["pass"] = all(rec["checks"].values())
    text = json.dumps(rec, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
