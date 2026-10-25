# Usage Report

## Final Full-Dataset Run
- Source: latest generated `code/ai_agent/ai_agent_usage.json` and/or run trace.
- Diagnostics cases: 112
- Output rows: 518
- Routing: candidates were validated against deterministic status and packet evidence in trace diagnostics.

## Model Provider And Names
- Provider: from MODEL_PROVIDER
- Model: from OLLAMA_MODEL

## Model Calls
- calls: TODO (no usage artifact currently present)
- successful_calls: TODO (no usage artifact currently present)
- failed_calls: TODO (no usage artifact currently present)

## Tokens
- input_tokens_estimate: TODO
- output_tokens_estimate: TODO
- total_tokens_estimate: TODO
- average_tokens_per_request: 0.0

## Cost
- estimated cost: 0.0 USD under local Ollama placeholder pricing.
- Set `PricingProvider` values from the actual hosted model invoice before final submission.

## Notes
- trace available: True
- usage artifact available: False
- `ai_agent/ai_agent_run_trace.jsonl` and `ai_agent/ai_agent_usage.json` were ignored to avoid pushing credentials or large diagnostic artifacts.
- For an official package run, retain those files locally and align them with the final run summary here.
