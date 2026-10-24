import csv
import json
import os
import re
import sqlite3
import subprocess
import datetime
from collections import defaultdict, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
EVIDENCE = DATASET / "evidence"


def parse_dt(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(s)


def load_manifest():
    rows = list(csv.DictReader(open(DATASET / "evidence_files.csv", newline="", encoding="utf-8")))
    for r in rows:
        r["_captured"] = parse_dt(r["captured_at"])
    return rows


MANIFEST = load_manifest()
MANIFEST_BY_ID = {r["file_id"]: r for r in MANIFEST}


def run_cmd(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def file_text(meta):
    rel = meta["file_path"]
    path = DATASET / rel
    lc = rel.lower()
    if lc.endswith(".pdf"):
        return run_cmd(["pdftotext", "-layout", str(path), "-"])
    if lc.endswith(".png"):
        return run_cmd(["tesseract", str(path), "stdout"])
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


KNOWN_SERVICE_KEYS = {"account_service", "analytics_platform", "backup_service", "export_service",
                      "integration_service", "search_service", "security_service", "support_platform",
                      "identity_service", "legal_service", "cache_service", "marketing_platform",
                      "email_delivery_service", "data_catalog", "account_failover_replica",
                      "profile_service"}


def infer_system_from_resource(rid):
    rid = rid or ""
    if rid.startswith("account_"):
        return "account_service"
    if rid.startswith("search_"):
        return "search_service"
    if rid.startswith("analytics_"):
        return "analytics_platform"
    if rid.startswith("support_"):
        return "support_platform"
    if rid.startswith("export_"):
        return "export_service"
    if rid.startswith("data_flow_") or rid.startswith("flow_"):
        return "integration_service"
    if rid.startswith("backup_"):
        return "backup_service"
    if rid.startswith("tombstone_"):
        return "identity_service"
    if rid.startswith("audit_record_"):
        return "security_service"
    if rid.startswith("legal_hold_"):
        return "legal_service"
    if rid.startswith("cache_"):
        return "cache_service"
    if rid.startswith("marketing_"):
        return "marketing_platform"
    if rid.startswith("hold_"):
        return "legal_service"
    return None


def map_service_name(name):
    if not name:
        return None
    n = str(name).lower()
    mapping = {
        "support_replica": "support_platform",
        "account_service": "account_service",
        "search_service": "search_service",
        "analytics_platform": "analytics_platform",
        "integration_service": "integration_service",
        "backup_service": "backup_service",
        "security_service": "security_service",
        "legal_service": "legal_service",
        "identity_service": "identity_service",
        "export_service": "export_service",
        "support_platform": "support_platform",
        "cache_service": "cache_service",
        "marketing_platform": "marketing_platform",
        "email_delivery_service": "email_delivery_service",
    }
    return mapping.get(n, name)


def get_record_system(rec, ctx):
    for key in ("system", "data_store", "source_system", "target_system", "service"):
        if isinstance(rec.get(key), str) and rec.get(key):
            s = map_service_name(rec.get(key))
            if s:
                return s
    if isinstance(ctx.get("service"), str) and ctx.get("service"):
        return map_service_name(ctx.get("service"))
    rid = rec.get("resource_id")
    s = infer_system_from_resource(rid)
    return s or ctx.get("service") or "unknown_service"


def get_record_category(rec, ctx):
    # explicit category/type
    for key in ("data_category", "category", "record_type", "record_category", "collection"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            up = v.upper()
            if up in ("USER_PROFILE", "SECONDARY_IDENTIFIER", "SEARCH_INDEX_ENTRY", "CACHE_ENTRY",
                      "SUPPORT_MESSAGE", "SUPPORT_ATTACHMENT", "ANALYTICS_PROFILE", "PRODUCT_TELEMETRY",
                      "USER_EXPORT", "MARKETING_PROFILE", "EMAIL_DELIVERY_LOG", "SECURITY_AUDIT_LOG",
                      "LEGAL_HOLD_RECORD", "BACKUP_ARCHIVE", "ACCOUNT_TOMBSTONE"):
                return up
            # support kind extraction
            if "ATTACHMENT" in up:
                return "SUPPORT_ATTACHMENT"
            if "SUPPORT" in up:
                return "SUPPORT_MESSAGE"
            if "ANALYTICS" in up:
                return "ANALYTICS_PROFILE"
            if "EXPORT" in up:
                return "USER_EXPORT"
            if "BACKUP" in up:
                return "BACKUP_ARCHIVE"
            if "AUDIT" in up:
                return "SECURITY_AUDIT_LOG"
            if "LEGAL" in up or "HOLD" in up:
                return "LEGAL_HOLD_RECORD"
            if "TOMBSTONE" in up:
                return "ACCOUNT_TOMBSTONE"
            if "SEARCH" in up:
                return "SEARCH_INDEX_ENTRY"
            if "PROFILE" in up:
                return "USER_PROFILE"
    rid = rec.get("resource_id") or ""
    if rid.startswith("account_"):
        return "USER_PROFILE"
    if rid.startswith("search_"):
        return "SEARCH_INDEX_ENTRY"
    if rid.startswith("analytics_"):
        return "ANALYTICS_PROFILE"
    if rid.startswith("support_"):
        return "SUPPORT_MESSAGE"
    if rid.startswith("export_"):
        return "USER_EXPORT"
    if rid.startswith("data_flow_") or rid.startswith("flow_"):
        return None
    if rid.startswith("backup_"):
        return "BACKUP_ARCHIVE"
    if rid.startswith("tombstone_"):
        return "ACCOUNT_TOMBSTONE"
    if rid.startswith("audit_record_"):
        return "SECURITY_AUDIT_LOG"
    if rid.startswith("legal_hold_"):
        return "LEGAL_HOLD_RECORD"
    if rid.startswith("hold_"):
        return "LEGAL_HOLD_RECORD"
    if rid.startswith("cache_"):
        return "CACHE_ENTRY"
    if rid.startswith("marketing_"):
        return "MARKETING_PROFILE"
    return None


def get_record_timestamp(rec, file_captured):
    for key in ("observed_at", "recorded_at", "captured_at", "updated_at", "timestamp", "created_at", "event_time"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            try:
                return parse_dt(v)
            except ValueError:
                pass
    return file_captured


def flatten_values(obj, acc=None):
    if acc is None:
        acc = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.append((k, v))
            flatten_values(v, acc)
    elif isinstance(obj, list):
        for x in obj:
            flatten_values(x, acc)
    else:
        acc.append(("_value", obj))
    return acc


def walk_json(obj, ctx, out, file_id, file_captured):
    if isinstance(obj, dict):
        if "resource_id" in obj and isinstance(obj.get("resource_id"), str):
            rec = dict(obj)
            rec["_file_id"] = file_id
            sys_id = get_record_system(rec, ctx)
            cat = get_record_category(rec, ctx)
            ts = get_record_timestamp(rec, file_captured)
            newrec = {
                "file_id": file_id,
                "system_id": sys_id,
                "resource_id": rec["resource_id"],
                "data_category": cat,
                "captured_at": file_captured,
                "observed_at": ts,
                "fields": rec,
            }
            out.append(newrec)
        new_ctx = dict(ctx)
        for k, v in obj.items():
            if isinstance(k, str) and (k in KNOWN_SERVICE_KEYS or k in ("data_store", "system", "service", "source_system", "target_system")):
                if isinstance(v, str):
                    new_ctx["service"] = v
            if isinstance(k, str) and k in ("collection", "table", "record_type", "record_category"):
                if isinstance(v, str):
                    new_ctx["collection"] = v
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                walk_json(v, new_ctx, out, file_id, file_captured)
    elif isinstance(obj, list):
        for x in obj:
            walk_json(x, ctx, out, file_id, file_captured)


def extract_sql(path, file_id, file_captured):
    out = []
    text = open(path, encoding="utf-8").read()
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(text)
    except Exception:
        return out
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    except Exception:
        return out
    for tbl in tables:
        try:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{tbl}")')]
            rows = con.execute(f'SELECT * FROM "{tbl}"')
        except Exception:
            continue
        for row in rows:
            d = dict(zip(cols, row))
            rid = d.get("resource_id")
            if not isinstance(rid, str):
                continue
            rec = dict(d)
            rec["_table"] = tbl
            sys_id = infer_system_from_resource(rid)
            if sys_id is None:
                # table name hints
                t = tbl.lower()
                if "search" in t:
                    sys_id = "search_service"
                elif "analytics" in t:
                    sys_id = "analytics_platform"
                elif "support" in t:
                    sys_id = "support_platform"
                elif "export" in t:
                    sys_id = "export_service"
                elif "backup" in t:
                    sys_id = "backup_service"
                elif "security" in t or "audit" in t:
                    sys_id = "security_service"
                elif "profile" in t:
                    sys_id = "account_service"
                else:
                    sys_id = "unknown_service"
            cat = get_record_category(rec, {"collection": tbl})
            if cat is None:
                t = tbl.lower()
                if "search" in t:
                    cat = "SEARCH_INDEX_ENTRY"
                elif "analytics" in t:
                    cat = "ANALYTICS_PROFILE"
                elif "support" in t:
                    cat = "SUPPORT_MESSAGE"
                elif "export" in t:
                    cat = "USER_EXPORT"
                elif "backup" in t:
                    cat = "BACKUP_ARCHIVE"
                elif "security" in t or "audit" in t:
                    cat = "SECURITY_AUDIT_LOG"
                elif "profile" in t:
                    cat = "USER_PROFILE"
            ts = get_record_timestamp(rec, file_captured)
            out.append({
                "file_id": file_id,
                "system_id": sys_id,
                "resource_id": rid,
                "data_category": cat,
                "captured_at": file_captured,
                "observed_at": ts,
                "fields": rec,
            })
    try:
        con.close()
    except Exception:
        pass
    return out


KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*("([^"]*)"|\'([^\']*)\'|(\S+))')


def parse_kv_line(line):
    fields = {}
    for m in KV_RE.finditer(line):
        k = m.group(1)
        v = m.group(3) if m.group(3) is not None else (m.group(4) if m.group(4) is not None else m.group(5))
        if v is None:
            v = ""
        v = v.strip().rstrip(",")
        fields[k] = v
    return fields


HANDOVER_LINE_RE = re.compile(r'^\s*-\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$')


def extract_text_records(meta, file_id, file_captured, text):
    out = []
    lines = text.splitlines()
    # handover style
    for line in lines:
        m = HANDOVER_LINE_RE.match(line)
        if m:
            dt, system, rid, category, status, ident = [x.strip() for x in m.groups()]
            try:
                ts = parse_dt(dt)
            except Exception:
                ts = file_captured
            out.append({
                "file_id": file_id,
                "system_id": map_service_name(system) if map_service_name(system) else infer_system_from_resource(rid) or system,
                "resource_id": rid,
                "data_category": category.upper().replace(" ", "_") if category.upper() in ("USER_PROFILE", "SECONDARY_IDENTIFIER", "SEARCH_INDEX_ENTRY", "CACHE_ENTRY", "SUPPORT_MESSAGE", "SUPPORT_ATTACHMENT", "ANALYTICS_PROFILE", "PRODUCT_TELEMETRY", "USER_EXPORT", "MARKETING_PROFILE", "EMAIL_DELIVERY_LOG", "SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD", "BACKUP_ARCHIVE", "ACCOUNT_TOMBSTONE") else get_record_category({"resource_id": rid, "record_type": category}, {}),
                "captured_at": file_captured,
                "observed_at": ts,
                "fields": {"status": status, "identifier": ident, "system": system, "category": category},
            })
            # normalize data_category for handover abbreviations
            cat = out[-1]["data_category"]
            if cat is None:
                cat = get_record_category({"resource_id": rid, "record_type": category}, {})
            out[-1]["data_category"] = cat
    # generic kv lines with resource_id
    seen_rids_this_file = set()
    for line in lines:
        if "resource_id" not in line:
            continue
        fields = parse_kv_line(line)
        rid = fields.get("resource_id")
        if not isinstance(rid, str) or not rid:
            # maybe "resource_id:foo" captured? try regex
            m = re.search(r'resource_id["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]+)', line)
            if m:
                rid = m.group(1)
            else:
                continue
        # avoid duplicating handover already added? handover lines do not have resource_id=, so ok
        # system extraction
        sys_id = None
        if fields.get("system"):
            sys_id = map_service_name(fields.get("system")) or infer_system_from_resource(rid)
        elif fields.get("service"):
            sys_id = map_service_name(fields.get("service")) or infer_system_from_resource(rid)
        if sys_id is None:
            sys_id = infer_system_from_resource(rid)
        if sys_id is None:
            sys_id = "unknown_service"
        # category from record_type if present else infer
        cat = get_record_category({"resource_id": rid, "record_type": fields.get("record_type") or fields.get("type")}, {})
        ts = file_captured
        # try parse event time at line start
        mtime = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', line.strip())
        if mtime:
            try:
                ts = parse_dt(mtime.group(1))
            except Exception:
                ts = file_captured
        # store all fields
        fcopy = dict(fields)
        fcopy["_line"] = line.strip()
        out.append({
            "file_id": file_id,
            "system_id": sys_id,
            "resource_id": rid,
            "data_category": cat,
            "captured_at": file_captured,
            "observed_at": ts,
            "fields": fcopy,
        })
    # Generic text/PDF/image: extract resource IDs embedded; create one record per unique
    # resource id found in text if not already added.
    existing = {(r["resource_id"]) for r in out if r["file_id"] == file_id}
    for rid in set(re.findall(r'\b(account_|search_|analytics_|support_|export_|data_flow_|backup_|tombstone_|audit_record_|legal_hold_|cache_|marketing_)[A-Za-z0-9]+\b', text)):
        # rid is full match? regex includes prefix? Our pattern capture group excludes maybe; use finditer
        pass
    for m in re.finditer(r'\b((?:account_|search_|analytics_|support_|export_|data_flow_|backup_|tombstone_|audit_record_|legal_hold_|cache_|marketing_)[0-9]+)\b', text):
        rid = m.group(1)
        # trim trailing underscores
        rid = rid.rstrip("_")
        if rid in existing:
            continue
        # only if user identifier appears in text? We'll still create; filtering user later.
        sys_id = infer_system_from_resource(rid)
        if sys_id is None:
            continue
        cat = get_record_category({"resource_id": rid}, {})
        # collect a compact text snippet
        out.append({
            "file_id": file_id,
            "system_id": sys_id,
            "resource_id": rid,
            "data_category": cat,
            "captured_at": file_captured,
            "observed_at": file_captured,
            "fields": {"_text": text[:5000]},
        })
        existing.add(rid)
    return out


def extract_records():
    allrecs = []
    for meta in MANIFEST:
        fid = meta["file_id"]
        fc = meta["_captured"]
        path = DATASET / meta["file_path"]
        lc = meta["file_path"].lower()
        if lc.endswith(".json"):
            text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
            except Exception:
                data = None
            if data is not None:
                walk_json(data, {}, allrecs, fid, fc)
            # also text records? no
        elif lc.endswith(".sql"):
            sql_recs = extract_sql(DATASET / meta["file_path"], fid, fc)
            allrecs.extend(sql_recs)
        else:
            text = file_text(meta)
            # Handle SQL text? no
            recs = extract_text_records(meta, fid, fc, text)
            allrecs.extend(recs)
    return allrecs


def norm_ident(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v == "" or v.lower() in ("null", "none", "nan"):
            return None
        return v
    return str(v)


def collect_user_identifiers():
    # map user_id -> set of normalized identifier strings (lowercased where sensible)
    identities = {}
    for row in csv.DictReader(open(DATASET / "user_identities.csv", newline="", encoding="utf-8")):
        identities[row["user_id"]] = {
            "name": row["name"],
            "email": row["email"],
            "phone": row["phone"],
            "email_hash": row["email_hash"],
        }
    audit = defaultdict(lambda: defaultdict(set))
    for row in csv.DictReader(open(DATASET / "user_audit_log.csv", newline="", encoding="utf-8")):
        uid = row["user_id"]
        field = row["field_name"]
        oldv = row["old_value"]
        newv = row["new_value"]
        audit[uid][field].add(oldv)
        audit[uid][field].add(newv)
    user_identifiers = {}
    for uid, base in identities.items():
        idents = set()
        # base
        for v in (base["name"], base["email"], base["phone"], base["email_hash"], uid):
            nv = norm_ident(v)
            if nv:
                idents.add(nv)
        # audit history
        for field, vals in audit.get(uid, {}).items():
            for v in vals:
                nv = norm_ident(v)
                if nv:
                    idents.add(nv)
        user_identifiers[uid] = idents
    return user_identifiers, identities


def resolve_user(email):
    email_l = email.strip().lower()
    # direct current
    for row in csv.DictReader(open(DATASET / "user_identities.csv", newline="", encoding="utf-8")):
        if row["email"].strip().lower() == email_l:
            return row["user_id"]
    # audit log
    for row in csv.DictReader(open(DATASET / "user_audit_log.csv", newline="", encoding="utf-8")):
        if row["field_name"] == "email" and (row["old_value"].strip().lower() == email_l or row["new_value"].strip().lower() == email_l):
            return row["user_id"]
    return None


def record_matches_user(rec, uid, idents):
    fields = rec.get("fields") or {}
    # structured fields
    matched = False
    # exact customer/subject reference
    for k, v in fields.items():
        if not isinstance(v, str):
            continue
        kl = k.lower()
        vv = v.strip()
        if vv.lower() in ("null", "none", ""):
            continue
        if uid and vv == uid:
            if any(tok in kl for tok in ("reference", "user", "subject", "customer", "owner")) or kl in ("user_id",):
                matched = True
        # exact identifier match
        if vv in idents or vv.lower() in {i.lower() for i in idents}:
            # avoid generic huge text? fields values generally small
            if len(vv) < 500:
                matched = True
    # raw text field
    text = None
    if isinstance(fields.get("_text"), str):
        text = fields.get("_text")
    elif isinstance(fields.get("_line"), str):
        text = fields.get("_line")
    if text:
        for ident in idents:
            il = ident.lower()
            if len(il) < 4:
                continue
            if __import__('re').search(r'(?<![A-Za-z0-9_])' + __import__('re').escape(il) + r'(?![A-Za-z0-9_])', text.lower()):
                matched = True
                break
    # identifier in entire record system/path? none
    return matched


def build_resource_to_users(records, user_identifiers):
    ident_to_user = {}
    for uid, idents in user_identifiers.items():
        for ident in idents:
            if isinstance(ident, str):
                ident_to_user[ident] = uid
                ident_to_user[ident.lower()] = uid
    user_ids = set(user_identifiers.keys())
    r2u = defaultdict(set)
    for rec in records:
        fields = rec.get("fields") or {}
        users = set()
        for k, v in fields.items():
            if not isinstance(v, str):
                continue
            vv = v.strip()
            if vv == "" or vv.lower() in ("null", "none"):
                continue
            if vv in ident_to_user:
                users.add(ident_to_user[vv])
            elif vv.lower() in ident_to_user:
                users.add(ident_to_user[vv.lower()])
            kl = k.lower()
            if vv in user_ids and any(tok in kl for tok in ("reference", "user", "subject", "customer", "owner", "lookup")):
                users.add(vv)
        for u in users:
            r2u[rec["resource_id"]].add(u)
    return r2u

def has_direct_identifier(obs):
    if obs is None:
        return False
    fields = obs.get("fields") or {}
    # check direct identity fields
    direct_keys = []
    for k, v in fields.items():
        kl = k.lower()
        if any(tok in kl for tok in ("full_name", "name", "email", "phone", "email_hash", "hash")):
            if not any(tok in kl for tok in ("hash_algorithm",)):
                direct_keys.append((k, v))
    for k, v in direct_keys:
        if isinstance(v, str):
            if v.strip() and v.strip().lower() not in ("null", "none", ""):
                return True
        elif v is not None:
            return True
    # status signals
    st = " ".join(str(v) for v in fields.values()).lower()
    if any(tok in st for tok in ("identifiable", "exposed", "present", "found")) and not any(tok in st for tok in ("redacted", "anonymized", "not_found", "absent", "blocked")):
        # only counts for some records; direct field null could still be exposed due to status
        if any(tok in st for tok in ("identifiable", "exposed")):
            return True
    return False


def restore_allowed(obs):
    if obs is None:
        return False
    fields = obs.get("fields") or {}
    # explicit
    for k, v in fields.items():
        if "restore_allowed" in k.lower():
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("1", "true", "yes"):
                return True
            if s in ("0", "false", "no"):
                return False
    st = " ".join(str(v) for v in fields.values()).lower()
    if any(tok in st for tok in ("active", "allowed", "found", "present", "enabled")) and not any(tok in st for tok in ("blocked", "disabled", "stopped", "false")):
        # could be a restore entry
        if "restore" in st or "backup" in st or obs.get("data_category") == "BACKUP_ARCHIVE":
            return True
    return False


def flow_active(obs):
    if obs is None:
        return False
    fields = obs.get("fields") or {}
    for k, v in fields.items():
        kl = k.lower()
        if kl == "enabled":
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("1", "true", "yes"):
                return True
            if s in ("0", "false", "no"):
                return False
    st = " ".join(str(v) for v in fields.values()).lower()
    if any(tok in st for tok in ("active", "enabled", "processing", "started")) and not any(tok in st for tok in ("disabled", "stopped", "inactive", "blocked")):
        if obs.get("data_category") is None or "DATA_FLOW" in str(obs.get("data_category", "")):
            return True
    # if status-like field says ACTIVE
    if "active" in st:
        return True
    if "disabled" in st or "stopped" in st or "inactive" in st:
        return False
    return False


def is_present(obs):
    if obs is None:
        return None  # unknown
    fields = obs.get("fields") or {}
    st = " ".join(str(v) for v in fields.values()).lower()
    if any(tok in st for tok in ("not_found", "absent", "deleted", "removed", "404")):
        return False
    if any(tok in st for tok in ("found", "present", "active", "exists", "success")):
        return True
    # if direct identifiers present
    if has_direct_identifier(obs):
        return True
    # if it's a record observation at all
    return True


def classify(request_text):
    t = " ".join(request_text.lower().split())
    non_phrases = [
        "not a request to erase",
        "don't delete", "don’t delete", "do not delete", "do not close the account",
        "send me a copy", "i still use it", "leave my profile", "leave my purchase",
        "leave my account active", "portable format", "download my account information",
        "keep the account", "i am not asking for deletion", "not asking for deletion",
        "marketing lists only", "promotional emails", "out of date", "replace it with the new one",
        "copy of last year", "resetting the login", "reset the login",
        "misspelled", "correct it without removing", "change the email address",
        "remove me from marketing", "stop the promotional", "information request",
    ]
    for p in non_phrases:
        if p in t:
            return False
    return True

def main():
    user_identifiers, base_identities = collect_user_identifiers()
    all_records = extract_records()
    r2u = build_resource_to_users(all_records, user_identifiers)

    # Read requests
    requests = list(csv.DictReader(open(DATASET / "deletion_requests.csv", newline="", encoding="utf-8")))

    # Precompute global latest evidence file per audit? fallback
    def latest_file_before(audit_dt):
        cands = [m for m in MANIFEST if m["_captured"] <= audit_dt]
        if not cands:
            return MANIFEST[0]["file_id"]
        cands.sort(key=lambda m: (m["_captured"], m["file_id"]))
        return cands[-1]["file_id"]

    out_rows = []
    for req in requests:
        case_id = req["case_id"]
        email = req["email"]
        audit_at = parse_dt(req["audit_at"])
        authorized = classify(req["request_text"])
        if not authorized:
            out_rows.append((case_id, "NOT_APPLICABLE", []))
            continue
        uid = resolve_user(email)
        # gather mapped resources for user
        user_resources = [rid for rid, us in r2u.items() if uid is not None and uid in us]
        # sort
        user_resources = sorted(set(user_resources))

        # For each resource, get latest as-of obs
        steps = []  # each dict has phase/order fields
        has_finding = False
        for rid in user_resources:
            # observations for this resource
            obs_list = [r for r in all_records if r["resource_id"] == rid]
            asof_candidates = [r for r in obs_list if r["captured_at"] <= audit_at]
            asof = None
            if asof_candidates:
                asof_candidates.sort(key=lambda r: (r["captured_at"], r["file_id"]))
                asof = asof_candidates[-1]
            # resolve system/category from asof or any obs
            if asof is not None:
                system_id = asof["system_id"]
                cat = asof["data_category"]
            else:
                # use latest global obs for this resource (future) for id/metadata
                obs_list_sorted = sorted(obs_list, key=lambda r: (r["captured_at"], r["file_id"]))
                if obs_list_sorted:
                    latest_global = obs_list_sorted[-1]
                    system_id = latest_global["system_id"]
                    cat = latest_global["data_category"]
                else:
                    system_id = infer_system_from_resource(rid)
                    cat = get_record_category({"resource_id": rid}, {})
            if system_id is None:
                system_id = infer_system_from_resource(rid) or "unknown_service"
            if cat is None:
                cat = get_record_category({"resource_id": rid}, {})

            # data flow special
            is_flow = (system_id == "integration_service" or (rid.startswith("data_flow_") or rid.startswith("flow_")))
            if is_flow:
                cat = None  # not a deletion_rules category; treat as active flow

            # Evaluate compliance and finding
            noncompliant = False
            finding_op = None
            finding_state = None
            remediation_op = None
            remediation_state = None
            verification_op = None
            verification_state = None
            finding_evidence = ""

            if is_flow:
                active = flow_active(asof) if asof is not None else False
                # if no asof, maybe check global future? classify noncompliant if globally active
                if asof is None:
                    # fall back: if any global obs active
                    active = any(flow_active(r) for r in obs_list)
                verification_state = "DISABLED"
                verification_op = "VERIFY_DATA_FLOW_DISABLED"
                if active:
                    noncompliant = True
                    finding_op = "REPORT_ACTIVE_DATA_FLOW"
                    finding_state = "ACTIVE"
                    remediation_op = "DISABLE_DATA_FLOW"
                    remediation_state = "DISABLED"
                else:
                    # compliant verification only
                    pass
            elif cat is None:
                # unknown category; skip
                continue
            elif cat in ("SECURITY_AUDIT_LOG", "LEGAL_HOLD_RECORD"):
                # retained records: no finding, verify present
                noncompliant = False
                verification_op = "VERIFY_RECORD_PRESENT"
                verification_state = "PRESENT"
            elif cat == "BACKUP_ARCHIVE":
                risk = restore_allowed(asof)
                if asof is None:
                    risk = any(restore_allowed(r) for r in obs_list)
                verification_op = "VERIFY_RESTORE_BLOCKED"
                verification_state = "BLOCKED"
                if risk:
                    noncompliant = True
                    finding_op = "REPORT_RESTORE_RISK"
                    finding_state = "ACTIVE"
                    remediation_op = "BLOCK_RESTORE"
                    remediation_state = "BLOCKED"
            elif cat == "ACCOUNT_TOMBSTONE":
                exposed = has_direct_identifier(asof) if asof is not None else False
                if asof is None:
                    exposed = any(has_direct_identifier(r) for r in obs_list)
                verification_op = "VERIFY_RECORD_REDACTED"
                verification_state = "REDACTED"
                if exposed:
                    noncompliant = True
                    finding_op = "REPORT_EXPOSED_PII"
                    finding_state = "PRESENT"
                    remediation_op = "REDACT_RECORD"
                    remediation_state = "REDACTED"
            elif cat == "SUPPORT_MESSAGE":
                exposed = has_direct_identifier(asof) if asof is not None else False
                if asof is None:
                    exposed = any(has_direct_identifier(r) for r in obs_list)
                if has_direct_identifier(asof) is None:
                    pass
                verification_op = "VERIFY_RECORD_REDACTED"
                verification_state = "REDACTED"
                if exposed:
                    noncompliant = True
                    finding_op = "REPORT_EXPOSED_PII"
                    finding_state = "PRESENT"
                    remediation_op = "REDACT_RECORD"
                    remediation_state = "REDACTED"
                else:
                    # compliant if record exists; if no asof, still verify redacted
                    pass
            elif cat in ("ANALYTICS_PROFILE", "PRODUCT_TELEMETRY"):
                exposed = has_direct_identifier(asof) if asof is not None else False
                if asof is None:
                    exposed = any(has_direct_identifier(r) for r in obs_list)
                verification_op = "VERIFY_RECORD_ANONYMIZED"
                verification_state = "ANONYMIZED"
                if exposed:
                    noncompliant = True
                    finding_op = "REPORT_EXPOSED_PII"
                    finding_state = "PRESENT"
                    remediation_op = "ANONYMIZE_RECORD"
                    remediation_state = "ANONYMIZED"
            elif cat == "EMAIL_DELIVERY_LOG":
                exposed = has_direct_identifier(asof) if asof is not None else False
                if asof is None:
                    exposed = any(has_direct_identifier(r) for r in obs_list)
                verification_op = "VERIFY_RECORD_REDACTED"
                verification_state = "REDACTED"
                if exposed:
                    noncompliant = True
                    finding_op = "REPORT_EXPOSED_PII"
                    finding_state = "PRESENT"
                    remediation_op = "REDACT_RECORD"
                    remediation_state = "REDACTED"
            elif cat in ("SUPPORT_ATTACHMENT",):
                present = is_present(asof)
                if present is None:
                    present = any(is_present(r) for r in obs_list) is True
                verification_op = "VERIFY_RECORD_ABSENT"
                verification_state = "ABSENT"
                if present:
                    noncompliant = True
                    finding_op = "REPORT_REMAINING_RECORD"
                    finding_state = "PRESENT"
                    remediation_op = "DELETE_RECORD"
                    remediation_state = "ABSENT"
            else:
                # presence-forbidden categories
                present = is_present(asof)
                if present is None:
                    present = any(is_present(r) for r in obs_list) is True
                verification_op = "VERIFY_RECORD_ABSENT"
                verification_state = "ABSENT"
                if present:
                    noncompliant = True
                    finding_op = "REPORT_REMAINING_RECORD"
                    finding_state = "PRESENT"
                    remediation_op = "DELETE_RECORD"
                    remediation_state = "ABSENT"
                else:
                    # special: tombstone? already handled; absent account -> verify absent
                    pass

            if noncompliant:
                has_finding = True
                # finding evidence citation: asof file if available
                if asof is not None:
                    finding_evidence = asof["file_id"]
                else:
                    # try latest obs for resource
                    if obs_list:
                        obs_list_sorted = sorted(obs_list, key=lambda r: (r["captured_at"], r["file_id"]))
                        finding_evidence = obs_list_sorted[-1]["file_id"]
                    else:
                        finding_evidence = latest_file_before(audit_at)
                # add finding step
                steps.append({
                    "phase": "FINDING",
                    "system_id": system_id,
                    "resource_id": rid,
                    "operation": finding_op,
                    "state": finding_state,
                    "evidence_id": finding_evidence,
                    "_oporder": None,
                })
                # remediation step
                steps.append({
                    "phase": "REMEDIATION",
                    "system_id": system_id,
                    "resource_id": rid,
                    "operation": remediation_op,
                    "state": remediation_state,
                    "evidence_id": "",
                    "_oporder": None,
                })

            # verification step for this resource
            # For INCOMPLETE, all verification empty evidence; for COMPLETE cite evidence
            # cite: if complete, use asof evidence if available else latest file before audit (global)
            if asof is not None:
                ver_ev = asof["file_id"]
            else:
                # if there is some obs (future) use its? but audit constraint says before; use latest before if any resource-specific before
                b4 = [r for r in obs_list if r["captured_at"] <= audit_at]
                if b4:
                    b4.sort(key=lambda r: (r["captured_at"], r["file_id"]))
                    ver_ev = b4[-1]["file_id"]
                else:
                    ver_ev = latest_file_before(audit_at)
            # default verification for flows etc set; for retained redacted etc set
            if verification_op is None:
                # unknown; skip
                continue
            # In incomplete cases, verification evidence empty. Status decided later? We'll handle evidence after has_finding known.
            steps.append({
                "phase": "VERIFICATION",
                "system_id": system_id,
                "resource_id": rid,
                "operation": verification_op,
                "state": verification_state,
                "evidence_id": ver_ev,
                "_oporder": None,
            })

        status = "INCOMPLETE" if has_finding else "COMPLETE"

        # Adjust verification evidence emptiness for incomplete
        op_order_map = {
            "FINDING": {"REPORT_REMAINING_RECORD": 1, "REPORT_EXPOSED_PII": 2, "REPORT_ACTIVE_DATA_FLOW": 3, "REPORT_RESTORE_RISK": 4},
            "REMEDIATION": {"DISABLE_DATA_FLOW": 1, "BLOCK_RESTORE": 2, "DELETE_RECORD": 3, "REDACT_RECORD": 4, "ANONYMIZE_RECORD": 5},
            "VERIFICATION": {"VERIFY_RECORD_ABSENT": 1, "VERIFY_RECORD_PRESENT": 2, "VERIFY_RECORD_REDACTED": 3, "VERIFY_RECORD_ANONYMIZED": 4, "VERIFY_DATA_FLOW_DISABLED": 5, "VERIFY_RESTORE_BLOCKED": 6},
        }
        phase_rank = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}

        for st in steps:
            if status == "INCOMPLETE" and st["phase"] == "VERIFICATION":
                st["evidence_id"] = ""
            if status == "INCOMPLETE" and st["phase"] == "REMEDIATION":
                st["evidence_id"] = ""
            # findings already evidence
            st["_sort"] = (phase_rank[st["phase"]], op_order_map[st["phase"]].get(st["operation"], 99), st["system_id"], st["resource_id"])

        steps.sort(key=lambda s: s["_sort"])
        # remove helper keys
        final_steps = []
        for i, s in enumerate(steps, start=1):
            final_steps.append({
                "step_no": i,
                "phase": s["phase"],
                "system_id": s["system_id"],
                "resource_id": s["resource_id"],
                "operation": s["operation"],
                "state": s["state"],
                "evidence_id": s["evidence_id"],
            })

        out_rows.append((case_id, status, final_steps))

    # Write output.csv at ROOT
    with open(ROOT / "output.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "deletion_status", "steps_json"])
        for case_id, status, steps in out_rows:
            w.writerow([case_id, status, json.dumps(steps, separators=(",", ":"))])

    print("done", len(out_rows))


if __name__ == "__main__":
    main()
