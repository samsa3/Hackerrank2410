"""Offline deterministic retrieval indexes for the AI audit agent.

Step 4 establishes the evidence-backed retrieval layer. These indexes are
rebuildable from participant-facing files and are meant to be the authority for
what a model is allowed to cite. A vector/semantic index, if added later, is
only a candidate-discovery tool and never replaces these deterministic indexes.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))


def parse_timestamp(value: str) -> datetime:
    """Parse ISO-8601 timestamps, with Z-normalized form handling."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


@dataclass
class IdentityRecord:
    user_id: str
    name: str
    email: str
    phone: str
    email_hash: str
    updated_at: str
    identifiers: List[str]


@dataclass
class IndexedRecord:
    file_id: str
    captured_at: str
    observed_at: str
    system_id: str
    resource_id: str
    data_category: Optional[str]
    fields: Dict[str, Any]

    @property
    def captured_dt(self) -> datetime:
        return parse_timestamp(self.captured_at)

    @property
    def observed_dt(self) -> datetime:
        try:
            return parse_timestamp(self.observed_at)
        except ValueError:
            return self.captured_dt


@dataclass
class RetrievalResult:
    """One bounded retrieval response with explicit coverage metadata."""

    resource_id: str
    as_of: Optional[IndexedRecord]
    history: List[IndexedRecord]
    coverage_note: str


class DeterministicEvidenceIndex:
    """A small, rebuildable evidence index.

    TODO: Add optional SQLite persistence once the packet assembler is stable.
    """

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = Path(project_root)
        self.identity_by_user: Dict[str, IdentityRecord] = {}
        self.user_by_identifier: Dict[str, str] = {}
        self.records: List[IndexedRecord] = []
        self.record_by_id: Dict[str, List[IndexedRecord]] = defaultdict(list)
        self.resource_to_users: Dict[str, Set[str]] = defaultdict(set)
        self.evidence_ids: Set[str] = set()
        self.evidence_captured_at: Dict[str, datetime] = {}

    def load(self) -> None:
        #  is not a Python package, so load it explicitly by path.
        import importlib.util

        deterministic_path = self.project_root / "code" / "main.py"
        spec = importlib.util.spec_from_file_location("deterministic_main", deterministic_path)
        deterministic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deterministic)

        # Keep the loaded module reachable for later deterministic calls.
        self._deterministic = deterministic
        raw_records = deterministic.extract_records()
        identities, raw_identities = self._deterministic.collect_user_identifiers()

        self.evidence_captured_at = {
            file_id: timestamp
            for file_id, timestamp in [
                (entry["file_id"], entry["_captured"]) for entry in self._deterministic.MANIFEST
            ]
        }

        # Convert deterministic records into a compact indexed representation.
        for record in raw_records:
            file_id = record["file_id"]
            captured = record["captured_at"]
            observed = record.get("observed_at", captured)

            if isinstance(captured, datetime):
                captured_string = captured.isoformat().replace("+00:00", "Z")
            else:
                captured_string = str(captured)

            if isinstance(observed, datetime):
                observed_string = observed.isoformat().replace("+00:00", "Z")
            else:
                observed_string = str(observed)

            indexed = IndexedRecord(
                file_id=file_id,
                captured_at=captured_string,
                observed_at=observed_string,
                system_id=record["system_id"],
                resource_id=record["resource_id"],
                data_category=record.get("data_category"),
                fields=record.get("fields") or {},
            )
            self.records.append(indexed)
            self.record_by_id[indexed.resource_id].append(indexed)

        # Load identity information.
        for user_id, identifiers in identities.items():
            row = raw_identities.get(user_id, {})
            self.identity_by_user[user_id] = IdentityRecord(
                user_id=user_id,
                name=row.get("name", ""),
                email=row.get("email", ""),
                phone=row.get("phone", ""),
                email_hash=row.get("email_hash", ""),
                updated_at=row.get("updated_at", ""),
                identifiers=sorted(identifiers),
            )
            for identifier in identifiers:
                if isinstance(identifier, str):
                    self.user_by_identifier[identifier] = user_id
                    self.user_by_identifier[identifier.lower()] = user_id

        # Load resource-to-user mappings.
        mapping = self._deterministic.build_resource_to_users(raw_records, identities)
        for resource_id, users in mapping.items():
            self.resource_to_users[resource_id].update(users)

        self.evidence_ids = {entry["file_id"] for entry in self._deterministic.MANIFEST}

    def records_for_user(self, user_id: str) -> List[IndexedRecord]:
        resource_ids = [
            rid for rid, users in self.resource_to_users.items()
            if user_id in users
        ]
        return [
            record
            for rid in sorted(resource_ids)
            for record in self.record_by_id.get(rid, [])
        ]

    def latest_at_or_before(self, resource_id: str, audit_at: str) -> Optional[IndexedRecord]:
        cutoff = parse_timestamp(audit_at)
        candidates = [
            record for record in self.record_by_id.get(resource_id, [])
            if record.captured_dt <= cutoff
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda rec: (rec.captured_dt, rec.file_id))
        return candidates[-1]

    def resource_history(self, resource_id: str) -> List[IndexedRecord]:
        return sorted(self.record_by_id.get(resource_id, []), key=lambda r: (r.captured_dt, r.file_id))

    def evidence_cutoff(self, audit_at: str) -> List[IndexedRecord]:
        cutoff = parse_timestamp(audit_at)
        return [record for record in self.records if record.captured_dt <= cutoff]


# Optional Pydantic-compatible sketch, commented out because Pydantic is not
# currently installed in the active interpreter:
#
# from pydantic import BaseModel, Field
#
# class IndexedRecordModel(BaseModel):
#     file_id: str
#     captured_at: str
#     resource_id: str
#     system_id: str
#     data_category: Optional[str] = None
#     fields: Dict[str, Any] = Field(default_factory=dict)
#
# This dataclass-based representation can be replaced by the commented model
# later without changing packet or validation interfaces.
