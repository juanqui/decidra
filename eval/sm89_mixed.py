#!/usr/bin/env python3
"""Mixed 1q/8q/64q SM89 scheduler test."""
import argparse, json, statistics, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = [
    ("small", ROOT/"data/semel_short256_64q.json", 1),
    ("medium", ROOT/"data/d3_ticket_2k_64q.json", 8),
    ("large", ROOT/"data/semel_long8k_64q.json", 64),
]

def payload(path, n):
    r = json.loads(Path(path).read_text())
    qs = dict(list(r["questions"].items())[:n])
    return json.dumps({**r, "questions": qs}, separators=(",", ":")).encode()

def post(url, body, timeout=300):
    t = time.perf_counter()
    req = urllib.request.Request(url.rstrip("/")+"/v1/systemone", data=body,
                                 headers={"content-type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.load(resp)
    return (time.perf_counter()-t)*1000.0, len(obj["answers"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--concurrency", default="8,16,32")
    ap.add_argument("--out", default="/tmp/sm89_mixed.json")
    args = ap.parse_args()
    bodies = [(name, path, n, payload(path, n)) for name, path, n in CLASSES]
    for _, _, _, body in bodies:
        post(args.url, body)
    report = {"levels": []}
    for conc in [int(x) for x in args.concurrency.split(",")]:
        jobs = []
        for i in range(conc):
            jobs.append(bodies[i % len(bodies)])
        barrier = threading.Barrier(len(jobs)+1)
        rows=[]; lock=threading.Lock()
        def worker(item):
            name, _, n, body = item
            barrier.wait()
            try:
                ms, answered = post(args.url, body)
                with lock: rows.append((name, n, ms, answered, None))
            except Exception as e:
                with lock: rows.append((name, n, None, 0, str(e)[:120]))
        threads=[threading.Thread(target=worker,args=(j,)) for j in jobs]
        for t in threads: t.start()
        barrier.wait(); started=time.perf_counter()
        for t in threads: t.join()
        wall=time.perf_counter()-started
        per={}
        for name,_,_,_,_ in [(n,p,q,b) for n,p,q,b in []]: pass
        for name in ("small","medium","large"):
            vals=[r[2] for r in rows if r[0]==name and r[2] is not None]
            if vals: per[name]={"p50_ms":statistics.median(vals),"mean_ms":statistics.mean(vals),"n":len(vals)}
        level={"concurrency":conc,"wall_s":wall,"errors":sum(r[4] is not None for r in rows),
               "classes":per,"throughput_qps":sum(r[1] for r in rows if r[2] is not None)/wall}
        report["levels"].append(level)
        print(json.dumps(level), flush=True)
    Path(args.out).write_text(json.dumps(report,indent=1)+"\n")

if __name__=="__main__": main()
