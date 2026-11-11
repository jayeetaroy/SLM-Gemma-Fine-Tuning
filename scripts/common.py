"""Shared prompt and serialization helpers for the experiment clients."""
import json
import os


def get_field(row, *names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_tools(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("functions", value.get("tools", [value]))
    normalized = []
    for tool in value or []:
        if "function" in tool:
            tool = tool["function"]
        normalized.append({"name": tool.get("name", ""), "description": tool.get("description", ""),
                           "parameters": tool.get("parameters", tool.get("parameters_schema", {}))})
    return normalized


def canonical_call(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected exactly one call")
        value = value[0]
    if "function" in value:
        value = value["function"]
    args = value.get("arguments", value.get("parameters", {}))
    if isinstance(args, str):
        args = json.loads(args)
    return {"name": value.get("name", ""), "arguments": args}


def prompt_text(question, tools):
    return ("Choose exactly one function for the request. Return only a JSON object with keys "
            '"name" and "arguments".\nCandidate functions:\n' + json.dumps(tools, ensure_ascii=False) +
            "\nRequest: " + question)


def chat_prompt(tokenizer, question, tools):
    messages = [{"role": "user", "content": prompt_text(question, tools)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def hf_token():
    return os.environ.get("HF_TOKEN") or None
