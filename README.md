# Gemma Tool-Calling Lab

A reproducible experiment kit for comparing `google/gemma-3-1b-it` with one QLoRA adapter on single-call structured tool selection. This repository contains workflow code and result templates, **not measured results**. Run the experiment before making claims about accuracy, latency, throughput, or memory.

## Prerequisites

- Hugging Face account/token with access to Gemma 3 and `Salesforce/xlam-function-calling-60k`; accept both access terms before downloading.
- Google Colab GPU (T4 16 GB is the starting target; memory must be checked empirically) and Google Drive for saved adapter/results.
- Python 3.10+ for local preparation/evaluation. Docker with NVIDIA Container Toolkit and a compatible NVIDIA GPU for serving.
- About 20–30 GB working storage. Keep credentials in environment/Colab secrets, never in source control.

## Colab training

1. Open `notebooks/train_colab.ipynb` in Colab, select a GPU runtime, and provide `HF_TOKEN` through Colab secrets.
2. Run cells in order. The notebook installs the training stack, downloads and prepares data, inspects a decoded example and assistant-only loss mask, runs a 20-step smoke test, then trains the configured adapter.
3. Confirm the smoke adapter loads and produces nonempty output in vLLM before launching the full run. The notebook saves the final adapter, tokenizer, settings, package versions, and training log under the configured Drive output directory.
4. Start a clean serving runtime using the serving instructions below; training and vLLM dependencies are isolated.

## Local data preparation and evaluation

Create an environment with `requirements-eval.txt`, then set `HF_TOKEN` if the dataset requires authenticated access. Download, filter, and split the source dataset:

```bash
python scripts/prepare_data.py --output-dir data/processed
```

The command writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, `split_manifest.json`, and `exclusions.json`. Review the candidate-tool-count distribution and lock the test manifest before training or model selection. By default data is fetched at runtime and generated data is git-ignored.

After starting the server, run the same locked examples for both aliases:

```bash
python scripts/evaluate.py --input data/processed/test.jsonl --base-url http://localhost:8000/v1 --output-dir results/quality
python scripts/benchmark.py --input data/processed/test.jsonl --base-url http://localhost:8000/v1 --output-dir results/benchmark
```

Evaluation writes raw predictions and `metrics.json`; benchmark writes per-request latency samples and `summary.json`. Both preserve failures. Do not compare runs with different prompts, splits, generation settings, or inference precision.

## Serving with Docker

Build and run from the project root. Mount the model cache and the directory containing the trained adapter. The entrypoint registers `base` and `tool-specialist` as request model names on one vLLM instance.

```bash
docker build -f docker/Dockerfile.vllm -t gemma-tool-lab-vllm .
HF_TOKEN=... BASE_MODEL=google/gemma-3-1b-it ADAPTER_PATH=/models/adapter \
  docker run --rm --gpus all -p 8000:8000 -e HF_TOKEN -e BASE_MODEL -e ADAPTER_PATH \
  -v "$PWD/models:/models" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" gemma-tool-lab-vllm
```

For local CPU-side evaluation tools, build `docker/Dockerfile.tools`; see `docker/README.md` for mounted paths and examples. The training workflow remains in Colab.

## Experimental contract

- Data: `Salesforce/xlam-function-calling-60k`; one expected call, supported JSON argument types, grouped duplicate question/answer records, seed 42, target 1,000/100/200 train/validation/test. Oversize examples are excluded (not silently truncated) when prompt plus answer exceeds 1,024 tokenizer tokens. Actual counts and candidate-tool distribution are recorded.
- Prompt includes user request and candidate function definitions; output is one JSON object, e.g. `{"name":"get_weather","arguments":{"city":"Seattle","unit":"celsius"}}`. Same tokenizer chat template is used in training and inference.
- Training defaults are in `configs/train.json`: 4-bit NF4 QLoRA, rank 16/alpha 32/dropout .05, attention projections, context 1,024, microbatch 1/accumulation 8, one epoch, learning rate 1e-4, seed 42. These are initial choices, not optimized settings.
- Primary quality metric is exact full-call accuracy across all test examples. Also report JSON validity, function-name accuracy, argument exact match, invalid-tool rate, and categorized failures. Greedy decode, 256-token cap; invalid JSON and truncations count as failures. Include a 10-prompt qualitative ordinary-instruction sanity check.
- Benchmark base and adapter at concurrency 1 and 4, with 10 warmups and 100 measured requests per configuration. Record latency, output tokens, errors/timeouts, throughput, sampled peak GPU process memory, startup time, and environment/revision metadata. This closed-loop workload does not establish production capacity.
- No native tool-call parser, real tool execution, no-call/parallel-call behavior, API-disjoint generalization, or distributed scheduling is implemented.

## Results and limitations

`results/` contains only a schema/template until a run is performed. Record GPU, package versions, model revision, data exclusions, split manifest, and serving settings with every report. The test set is for final evaluation after choices are made using validation only. If results are negative, report them as measured. Never infer execution correctness from generated JSON alone.

If memory is insufficient, refilter consistently with shorter context or use a larger GPU; document the changed protocol. If training exceeds the available window, reduce the training subset to 500 while retaining validation and test sets, and record that deviation.

## Troubleshooting

- **401/ gated model:** verify the account accepted Gemma terms and the token is available as `HF_TOKEN`.
- **Dataset access/download:** verify dataset terms, token, network, and free disk; rerun preparation with the same seed and preserve the manifest.
- **CUDA out of memory:** reduce context and refilter, or choose a larger GPU; do not silently truncate labels.
- **vLLM adapter rejected:** confirm the serving image's pinned vLLM supports Gemma 3 and LoRA, and that the adapter was trained against the same base revision.
- **Training/serving dependency conflict:** use the separate Colab training runtime and serving container as documented.

## Repository map

`scripts/` contains preparation, evaluation, and benchmark clients; `notebooks/train_colab.ipynb` contains the training workflow; `configs/` holds experiment settings; `docker/` contains serving and tools images; `results/` is reserved for generated outputs.
