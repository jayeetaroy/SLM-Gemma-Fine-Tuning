#!/usr/bin/env python3
"""Fetch xLAM, normalize eligible single-call rows, group duplicates, and split."""
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

from common import canonical_call, get_field, normalize_tools


def convert(row):
    question = get_field(row, "query", "question", "instruction", "prompt")
    funcs = get_field(row, "tools", "functions", "function", default=[])
    answer = get_field(row, "answers", "answer", "response", "tool_calls")
    if question is None or answer is None:
        raise ValueError("missing question or answer")
    question = str(question).strip()
    tools = normalize_tools(funcs)
    call = canonical_call(answer)
    if not question or not call["name"] or not isinstance(call["arguments"], dict):
        raise ValueError("invalid question/call")
    if not any(t["name"] == call["name"] for t in tools):
        raise ValueError("expected function not in candidate set")
    return {"question": question, "tools": tools, "target": call}


def group_key(item):
    return hashlib.sha256((item["question"].casefold().strip() + "\n" + json.dumps(item["target"], sort_keys=True)).encode()).hexdigest()


def prepare_rows(rows, output_dir, seed=42, sizes=(1000, 100, 200), max_length=1024, tokenizer=None):
    valid, exclusions = [], Counter()
    for row in rows:
        try:
            item = convert(row)
            if tokenizer:
                prompt = tokenizer.apply_chat_template([{"role":"user","content":__import__("common").prompt_text(item["question"], item["tools"])}], tokenize=False, add_generation_prompt=True)
                answer = json.dumps(item["target"], ensure_ascii=False)
                if len(tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]) > max_length:
                    exclusions["over_max_length"] += 1; continue
            item["candidate_count"] = len(item["tools"])
            valid.append(item)
        except Exception as exc:
            exclusions[type(exc).__name__ + ": " + str(exc)[:100]] += 1
    groups = defaultdict(list)
    for item in valid: groups[group_key(item)].append(item)
    unique = [entries[0] for entries in groups.values()]
    rng = random.Random(seed); rng.shuffle(unique)
    ntrain, nval, ntest = sizes
    splits = {"train": unique[:ntrain], "validation": unique[ntrain:ntrain+nval], "test": unique[ntrain+nval:ntrain+nval+ntest]}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for item in items: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "exclusions.json").write_text(json.dumps(dict(exclusions), indent=2) + "\n")
    manifest = {"seed":seed,"requested_sizes":dict(zip(("train","validation","test"),sizes)),"actual_sizes":{k:len(v) for k,v in splits.items()},"deduplicated_groups":len(groups),"candidate_count_distribution":dict(Counter(x["candidate_count"] for x in unique)),"split_group_ids":{k:[group_key(x) for x in v] for k,v in splits.items()}}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, dict(exclusions)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--dataset",default="Salesforce/xlam-function-calling-60k"); ap.add_argument("--output-dir",default="data/processed"); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--train-size",type=int,default=1000); ap.add_argument("--validation-size",type=int,default=100); ap.add_argument("--test-size",type=int,default=200); ap.add_argument("--max-length",type=int,default=1024); ap.add_argument("--base-model",default="google/gemma-3-1b-it"); ap.add_argument("--skip-token-filter",action="store_true"); args=ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    data=load_dataset(args.dataset,split="train")
    tok=None if args.skip_token_filter else AutoTokenizer.from_pretrained(args.base_model,token=__import__("common").hf_token())
    manifest, excluded=prepare_rows(data,args.output_dir,args.seed,(args.train_size,args.validation_size,args.test_size),args.max_length,tok)
    print(json.dumps({"manifest":manifest,"exclusions":excluded},indent=2))
if __name__ == "__main__": main()
