# Evaluation Report

- Output rows: 518
- Status distribution from `output.csv`: {'INCOMPLETE': 260, 'COMPLETE': 248, 'NOT_APPLICABLE': 10}
- Trace cases: 518
- Trace status distribution: {'INCOMPLETE': 298, 'COMPLETE': 210, 'NOT_APPLICABLE': 10}
- Model candidate status distribution: {'INCOMPLETE': 296, 'COMPLETE': 212, 'NOT_APPLICABLE': 10}
- Deterministic baseline status distribution: {'INCOMPLETE': 260, 'COMPLETE': 248, 'NOT_APPLICABLE': 10}
- Final status source counts: {'ai+baseline': 478, 'ai': 40}
- Trace validation-error cases: 0
- Reported trace step count: 1981
- Average trace latency: 4957.25 ms

## Sequence And Validation
- All emitted rows were parsed as a valid list of step objects after generation.
- Phase/order follows the challenge operation matrix: `FINDING`, then `REMEDIATION`, then `VERIFICATION`.
- Within each phase, operation order is deterministic, followed by `system_id` and `resource_id`.
- Steps are numbered consecutively starting at 1.

## Custom Evaluation Cases
1. `custom_access_request` — information/rectification request, not deletion; expected `NOT_APPLICABLE`, empty steps.
2. `custom_multilingual_deletion` — internationalized Spanish-language deletion request; expected INCOMPLETE evidence handling.
3. `custom_restore_and_flow` — retained backup/export/sync path; expected INCOMPLETE with at least one flow or restore risk.
4. Sample request set: 32 requests in `dataset/sample_deletion_requests.csv`.

## Failures Found And Improvements Made
- Fixed substring identity matching with exact/token-boundary historical identity resolution.
- Added deterministic fallback whenever no model and endpoint is unavailable.
- Fixed unsafe `OllamaModelClient` `max_output_tokens` access and capped provider request payload.
- Enforced semantic consistency for candidate COMPLEX verdicts that contradict their own explanation.
- Removed a duplicate root package and generated runtime artifacts from tracked source.
- Tracked explicit model/baseline status disagreement and override provenance in trace artifacts.

## Notes
- `output.csv` was generated against the production deletion_requests.csv set.
- The sample request set has separate case identifiers, so it cannot be scored directly by positional comparison alone.
- Hidden grading accuracy cannot be computed from repository files; the canonical answers are not available.
- Use the trace JSONL and summary counts as the run evidence.
