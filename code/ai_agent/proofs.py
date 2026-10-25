"""Runtime compliance and attestation helpers for the AI audit agent.

These helpers deliberately add evidence and safety checks around the existing
validated pipeline rather than allowing the model to bypass deterministic rules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class RuntimeComplianceMonitor:
    """Small local predicate gate for dispatch-time and model-time checks."""

    expected_provider: Optional[str] = None

    def validate_dispatch(
        self,
        case_id: str,
        authorized: bool,
        output_status: Optional[str],
        baseline_status: Optional[str],
    ) -> List[str]:
        errors: List[str] = []

        if not isinstance(case_id, str) or not case_id.strip():
            errors.append("runtime monitor: case id must be a non-empty string")

        if authorized is False and output_status not in {None, "NOT_APPLICABLE"}:
            errors.append("runtime monitor: an unauthorized case may only map to NOT_APPLICABLE")
        if authorized is True and output_status == "NOT_APPLICABLE":
            errors.append("runtime monitor: an authorized case cannot be NOT_APPLICABLE")

        if baseline_status in {"COMPLETE", "INCOMPLETE"} and output_status == "NOT_APPLICABLE":
            errors.append("runtime monitor: deterministic baseline conflicts with NOT_APPLICABLE")

        return errors

    def validate_provider(self, actual_provider: Optional[str]) -> List[str]:
        expected = (self.expected_provider or "").lower()
        if not expected:
            return []
        if (actual_provider or "").lower() != expected:
            return [f"runtime monitor: provider mismatch {actual_provider!r} != {expected!r}"]
        return []

    def intercept(
        self,
        *,
        case_id: str,
        authorized: bool,
        output_status: Optional[str],
        baseline_status: Optional[str],
        actual_provider: Optional[str],
    ) -> None:
        errors = self.validate_dispatch(
            case_id=case_id,
            authorized=authorized,
            output_status=output_status,
            baseline_status=baseline_status,
        )
        errors.extend(self.validate_provider(actual_provider))
        if errors:
            raise ValueError("; ".join(errors))


@dataclass
class AttestationPacket:
    """Machine-checkable summary of one complete run."""

    digest_version: str
    run_id: Optional[str]
    model_provider: Optional[str]
    model_name: Optional[str]
    requested_cases: int
    produced_cases: int
    status_counts: Dict[str, int]
    evidence_ids: int
    trace_digest: Optional[str]
    output_digest: Optional[str]
    usage_digest: Optional[str]
    validation_errors: int
    steps_written: int
    attested_at: str = ""


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_steps(rows: List[Dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        steps = row.get("steps_json", [])
        if isinstance(steps, list):
            total += len(steps)
        elif isinstance(steps, str):
            try:
                parsed = json.loads(steps)
                if isinstance(parsed, list):
                    total += len(parsed)
            except (json.JSONDecodeError, ValueError):
                continue
    return total


def build_attestation(
    *,
    run_id: Optional[str],
    model_provider: Optional[str],
    model_name: Optional[str],
    requested_cases: int,
    produced_rows: List[Dict[str, Any]],
    evidence_ids: Set[str],
    validation_error_count: int,
    trace_path: Optional[Path],
    output_path: Optional[Path],
    usage_path: Optional[Path],
) -> AttestationPacket:
    from datetime import datetime, timezone

    counts: Dict[str, int] = {}
    for row in produced_rows:
        status = str(row.get("deletion_status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1

    return AttestationPacket(
        digest_version="ghost-data-attestation-v1",
        run_id=run_id,
        model_provider=model_provider,
        model_name=model_name,
        requested_cases=requested_cases,
        produced_cases=len(produced_rows),
        status_counts=counts,
        evidence_ids=len(evidence_ids or set()),
        trace_digest=sha256_file(trace_path),
        output_digest=sha256_file(output_path),
        usage_digest=sha256_file(usage_path),
        validation_errors=validation_error_count,
        steps_written=_count_steps(produced_rows),
        attested_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def write_attestation(attestation: AttestationPacket, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(attestation), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class PromotionManifest:
    source: str
    destination: str
    valid_rows: int
    digest: Optional[str]


def promote_output_atomically(
    *,
    source: Path,
    destination: Path,
    valid_rows: int,
) -> PromotionManifest:
    """Promote an output only after a deterministic row-count and digest check."""
    if not source.exists():
        raise FileNotFoundError(f"promotion source does not exist: {source}")

    file_rows = 0
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        file_rows = sum(1 for _ in reader)

    if file_rows != valid_rows:
        raise ValueError(f"promotion row mismatch: reported {valid_rows}, file has {file_rows}")

    digest = sha256_file(source)
    temporary = destination.with_suffix(destination.suffix + ".promotion_tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        # pathlib copy2-style copy preserving metadata.
        import shutil as _shutil

        _shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return PromotionManifest(
        source=str(source),
        destination=str(destination),
        valid_rows=valid_rows,
        digest=digest,
    )
