#!/usr/bin/env python3
"""Closed-loop request benchmark for base/adapter at concurrency 1 and 4."""
import argparse, concurrent.futures, json, time, statistics
from pathlib import Path

def benchmark(input_path,base_url,output_dir,warmups=10,requests=100,concurrencies=(1,4)):
    from openai import OpenAI
    rows=[json.loads(x) for x in Path(input_path).read_text().splitlines() if x.strip()]; out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); client=OpenAI(base_url=base_url,api_key="EMPTY"); samples=[]
    def call(row,model):
        prompt="Choose exactly one function. Return only JSON with name and arguments.\nCandidate functions:\n"+json.dumps(row["tools"],ensure_ascii=False)+"\nRequest: "+row["question"]; start=time.perf_counter()
        try:
            r=client.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=256)
            return {"latency_seconds":time.perf_counter()-start,"output_tokens":getattr(r.usage,"completion_tokens",0) or 0,"error":None}
        except Exception as e: return {"latency_seconds":time.perf_counter()-start,"output_tokens":0,"error":str(e)}
    for model in ("base","tool-specialist"):
      for concurrency in concurrencies:
        row=rows[0] if rows else None
        if row is None: raise ValueError("benchmark input is empty")
        for _ in range(warmups): call(row,model)
        workload=[rows[i%len(rows)] for i in range(requests)]; wall=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool: results=list(pool.map(lambda item:call(item,model),workload))
        elapsed=time.perf_counter()-wall; lat=[r["latency_seconds"] for r in results]; toks=sum(r["output_tokens"] for r in results)
        samples.extend({"model":model,"concurrency":concurrency,**r} for r in results)
        summary_key=f"{model}_concurrency_{concurrency}"; print(summary_key)
        if "summary" not in locals(): summary={}
        summary[summary_key]={"requests":requests,"errors":sum(bool(r["error"]) for r in results),"p50_latency_seconds":statistics.median(lat),"p95_latency_seconds":sorted(lat)[int(.95*(len(lat)-1))],"requests_per_second":requests/elapsed,"output_tokens_per_second":toks/elapsed,"output_tokens":toks,"wall_seconds":elapsed,"startup_seconds":None,"peak_gpu_process_memory_mb":None}
    (out/"samples.jsonl").write_text("".join(json.dumps(s)+"\n" for s in samples)); (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input",required=True); ap.add_argument("--base-url",default="http://localhost:8000/v1"); ap.add_argument("--output-dir",default="results/benchmark"); ap.add_argument("--warmups",type=int,default=10); ap.add_argument("--requests",type=int,default=100); a=ap.parse_args(); print(json.dumps(benchmark(a.input,a.base_url,a.output_dir,a.warmups,a.requests),indent=2))
