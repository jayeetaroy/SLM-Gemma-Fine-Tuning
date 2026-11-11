#!/usr/bin/env python3
"""Evaluate base and adapter aliases and retain raw outputs, including failures."""
import argparse, json, time
from pathlib import Path
from common import canonical_call

def score_one(text, expected, tools):
    result={"json_valid":False,"function_correct":False,"arguments_exact":False,"full_call_correct":False,"invalid_tool":False,"parsed":None,"failure":"invalid_json"}
    try:
        obj=json.loads(text)
        if not isinstance(obj,dict) or set(obj)!={"name","arguments"}: raise ValueError("not one contract object")
        pred=canonical_call(obj); result["json_valid"]=True; result["parsed"]=pred
        names={t["name"] for t in tools}; result["invalid_tool"]=pred["name"] not in names
        result["function_correct"]=pred["name"]==expected["name"]
        result["arguments_exact"]=pred["arguments"]==expected["arguments"]
        result["full_call_correct"]=result["function_correct"] and result["arguments_exact"]
        result["failure"]="" if result["full_call_correct"] else ("wrong_function" if not result["function_correct"] else "wrong_arguments")
    except Exception as e: result["failure"]="invalid_json: "+str(e)[:100]
    return result

def evaluate(input_path,base_url,output_dir,models=("base","tool-specialist"),cap=256):
    from openai import OpenAI
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); client=OpenAI(base_url=base_url,api_key="EMPTY")
    rows=[json.loads(x) for x in Path(input_path).read_text().splitlines() if x.strip()]; summary={}
    for model in models:
        predictions=[]
        for i,row in enumerate(rows):
            prompt="Choose exactly one function. Return only JSON with name and arguments.\nCandidate functions:\n"+json.dumps(row["tools"],ensure_ascii=False)+"\nRequest: "+row["question"]
            started=time.perf_counter(); error=None; text=""; usage={}
            try:
                response=client.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=cap)
                text=response.choices[0].message.content or ""; usage={"completion_tokens":getattr(response.usage,"completion_tokens",None)}
                if response.choices[0].finish_reason=="length": error="truncated"
            except Exception as exc: error=str(exc)
            scored=score_one(text,row["target"],row["tools"]); scored["full_call_correct"]=scored["full_call_correct"] and error is None
            predictions.append({"index":i,"model":model,"output":text,"error":error,"latency_seconds":time.perf_counter()-started,**usage,**scored})
        (out/f"{model.replace('/','_')}_predictions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in predictions))
        n=len(rows); summary[model]={k:{"count":sum(bool(p[k]) for p in predictions),"rate":sum(bool(p[k]) for p in predictions)/n if n else 0} for k in ("json_valid","function_correct","arguments_exact","full_call_correct","invalid_tool")}; summary[model]["failures"]={}
        for p in predictions:
            if p["failure"]: summary[model]["failures"][p["failure"]]=summary[model]["failures"].get(p["failure"],0)+1
    (out/"metrics.json").write_text(json.dumps({"n":len(rows),"models":summary},indent=2)+"\n"); return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input",required=True); ap.add_argument("--base-url",default="http://localhost:8000/v1"); ap.add_argument("--output-dir",default="results/quality"); ap.add_argument("--max-tokens",type=int,default=256); a=ap.parse_args(); print(json.dumps(evaluate(a.input,a.base_url,a.output_dir,cap=a.max_tokens),indent=2))
