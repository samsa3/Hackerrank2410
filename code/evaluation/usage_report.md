# Usage Report

## Final Full-Dataset Run
- TODO: populate after successful model run

## Model Provider And Names
- Provider: from `MODEL_PROVIDER`, default ollama
- Model: from `OLLAMA_MODEL` or configured equivalent
- Base URL: from `OLLAMA_BASE_URL`

## Model Calls
- calls: TODO
- successful_calls: TODO
- failed_calls: TODO

## Tokens
- input_tokens_estimate: TODO
- output_tokens_estimate: TODO
- total_tokens_estimate: TODO
- average_tokens_per_request: TODO

## Cost
- estimated input cost: TODO
- estimated output cost: TODO
- estimated total cost: TODO
- per-request cost estimate: TODO

## Notes
- JSON-L only estimates are currently recorded when the provider reports token counts.
- Provider pricing must be set in `ai_agent/model_clients.py` before final submission.
- `ai_agent/ai_agent_usage.json` stores the runtime trace summary.
