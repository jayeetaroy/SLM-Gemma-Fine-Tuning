# Docker usage

`Dockerfile.vllm` starts one OpenAI-compatible vLLM server. Set `HF_TOKEN`, `BASE_MODEL`, and `ADAPTER_PATH`; optionally set `BASE_REVISION`. It registers `base` and `tool-specialist` aliases. GPU access and a host-compatible NVIDIA Container Toolkit are required.

`Dockerfile.tools` contains preparation/evaluation/benchmark Python dependencies. Mount the project data and result directories when using it, for example `docker run --rm -v "$PWD:/app" gemma-tool-lab-tools scripts/evaluate.py --help`. Dataset preparation needs network access and `HF_TOKEN` where required.
