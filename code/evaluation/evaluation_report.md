# Evaluation Report

## Status Accuracy
- COMPLETE / INCOMPLETE / NOT_APPLICABLE counts:
  - TODO: populate after final run

## Step-Level And Field-Level Accuracy
- Method: deterministic comparison against `sample_output.csv`
- Reported mismatch statuses and changed step counts.
- TODO: populate after full evaluation run

## Sequence Accuracy
- Phase order: FINDING, REMEDIATION, VERIFICATION
- Within phase: operation order then system_id and resource_id
- Re-check all step numbers are consecutive from 1

## Custom Evaluation Cases
1. Information-only request (`custom_access_request`): expected `NOT_APPLICABLE`, empty steps
2. Multilingual deletion request (`custom_multilingual_deletion`): expected `INCOMPLETE`; exact resources pending evidence fixture
3. Restore-and-data-flow risk (`custom_restore_and_flow`): expected `INCOMPLETE`; exact resources pending evidence fixture

## Failures Found And Improvements Made
- False user-resource merges caused by substring identity matching were fixed with exact/token-boundary matching.
- Non-deletion classification was fixed for multilingual deletion and access requests.
- Pipeline now falls back safely when no model is reachable.
- TODO: add case-specific failure taxonomy from final AI run.
