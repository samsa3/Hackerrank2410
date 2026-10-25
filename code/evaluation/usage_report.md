# Usage Report

## Final Full-Dataset Run
- Source: latest generated `code/ai_agent/ai_agent_usage.json` and/or run trace.
- Diagnostics cases: 518
- Output rows: 518
- Routing: candidates were validated against deterministic status and packet evidence in trace diagnostics.

## Model Provider And Names
- Provider: ollama
- Model: qwen3.5:9b-q4_K_M

## Model Calls
- calls: 508
- successful_calls: 507
- failed_calls: 1

## Tokens
- input_tokens_estimate: 312665
- output_tokens_estimate: 97619
- total_tokens_estimate: 410284
- average_tokens_per_request: 792.0540540540541

## Cost
- estimated cost: 0.0 USD under local Ollama placeholder pricing.
- Set `PricingProvider` values from the actual hosted model invoice before final submission.

## Notes
- trace available: True
- usage artifact available: True
- `ai_agent/ai_agent_run_trace.jsonl` and `ai_agent/ai_agent_usage.json` were ignored to avoid pushing credentials or large diagnostic artifacts.
- For an official package run, retain those files locally and align them with the final run summary here.
