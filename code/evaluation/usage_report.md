# Usage Report

## Final Full-Dataset Run
- Source: latest generated `code/ai_agent/ai_agent_usage.json` and/or run trace.
- Diagnostics cases: 5
- Output rows: 518
- Routing: candidates were validated against deterministic status and packet evidence in trace diagnostics.

## Model Provider And Names
- Provider: ollama
- Model: qwen3.5:9b-q4_K_M

## Model Calls
- calls: 5
- successful_calls: 5
- failed_calls: 0

## Tokens
- input_tokens_estimate: 3053
- output_tokens_estimate: 948
- total_tokens_estimate: 4001
- average_tokens_per_request: 7.723938223938224

## Cost
- estimated cost: 0.0 USD under local Ollama placeholder pricing.
- Set `PricingProvider` values from the actual hosted model invoice before final submission.

## Notes
- trace available: True
- usage artifact available: True
- `ai_agent/ai_agent_run_trace.jsonl` and `ai_agent/ai_agent_usage.json` were ignored to avoid pushing credentials or large diagnostic artifacts.
- For an official package run, retain those files locally and align them with the final run summary here.
