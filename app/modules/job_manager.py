"""Chunk-based job management for multi-user mode.

A job assigns one or more chunks to a user; the user labels every conversation
in those chunks.  Progress is tracked per (conv_id, chunk_id) work unit.

A chunk is either a norm (when norms partition traces differently) or a
sliding-window slice of 30 traces with 5-trace overlap (when all norms cover
all traces).

Simple mode uses chunk-scoped label files directly (no job abstraction).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from app.modules.storage import (
    append_jsonl,
    ensure_dir,
    now_iso,
    read_jsonl,
    write_jsonl,
)


# ── Multi-user job helpers ─────────────────────────────────────────────────────

def create_job(
    username: str,
    chunk_ids: list[str],
    chunk_conv_map: dict[str, list[dict]],
    jobs_dir: Path,
    conv_ids_filter: "dict[str, list[str]] | None" = None,
) -> str:
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    ensure_dir(jobs_dir)

    manifest_entry: dict = {
        "job_id": job_id,
        "username": username,
        "chunk_ids": chunk_ids,
        "created_at": now_iso(),
        "status": "pending",
    }
    if conv_ids_filter:
        manifest_entry["conv_ids_filter"] = conv_ids_filter
    append_jsonl(jobs_dir / "manifest.jsonl", manifest_entry)

    units = []
    for chunk_id in chunk_ids:
        allowed = set(conv_ids_filter[chunk_id]) if conv_ids_filter and chunk_id in conv_ids_filter else None
        for conv in chunk_conv_map.get(chunk_id, []):
            conv_id = conv.get("convID", "")
            if allowed is not None and conv_id not in allowed:
                continue
            units.append({
                "conv_id": conv_id,
                "chunk_id": chunk_id,
                "unit_status": "pending",
                "labeled_by": None,
                "labeled_at": None,
                "turns": [],
            })
    write_jsonl(jobs_dir / f"{job_id}_labels.jsonl", units)
    return job_id


def get_all_jobs(jobs_dir: Path) -> list[dict]:
    return read_jsonl(jobs_dir / "manifest.jsonl")


def get_user_jobs(username: str, jobs_dir: Path) -> list[dict]:
    return [j for j in get_all_jobs(jobs_dir) if j.get("username") == username]


def get_job_units(job_id: str, jobs_dir: Path) -> list[dict]:
    return read_jsonl(jobs_dir / f"{job_id}_labels.jsonl")


def save_unit_labels(
    job_id: str,
    conv_id: str,
    chunk_id: str,
    turns: list[dict],
    username: str,
    jobs_dir: Path,
) -> None:
    labels_path = jobs_dir / f"{job_id}_labels.jsonl"
    units = get_job_units(job_id, jobs_dir)
    for unit in units:
        if unit["conv_id"] == conv_id and unit["chunk_id"] == chunk_id:
            unit["unit_status"] = "completed"
            unit["labeled_by"] = username
            unit["labeled_at"] = now_iso()
            unit["turns"] = turns
            break
    write_jsonl(labels_path, units)


def is_chunk_complete_job(job_id: str, chunk_id: str, jobs_dir: Path) -> bool:
    units = [
        u for u in get_job_units(job_id, jobs_dir)
        if u["chunk_id"] == chunk_id
    ]
    return bool(units) and all(u["unit_status"] == "completed" for u in units)


def get_completed_conv_ids_job(job_id: str, chunk_id: str, jobs_dir: Path) -> set[str]:
    return {
        u["conv_id"]
        for u in get_job_units(job_id, jobs_dir)
        if u["chunk_id"] == chunk_id and u["unit_status"] == "completed"
    }


def delete_job(job_id: str, jobs_dir: Path) -> None:
    manifest = read_jsonl(jobs_dir / "manifest.jsonl")
    write_jsonl(
        jobs_dir / "manifest.jsonl",
        [j for j in manifest if j["job_id"] != job_id],
    )
    labels_path = jobs_dir / f"{job_id}_labels.jsonl"
    if labels_path.exists():
        labels_path.unlink()
    bundles_path = jobs_dir / _BUNDLES_FILE
    bundles = read_jsonl(bundles_path)
    changed = any(b.get("job_id") == job_id for b in bundles)
    if changed:
        for b in bundles:
            if b.get("job_id") == job_id:
                b["claimed_by"] = None
                b["claimed_at"] = None
                b["job_id"] = None
        write_jsonl(bundles_path, bundles)


def update_job_status(job_id: str, jobs_dir: Path) -> None:
    manifest = read_jsonl(jobs_dir / "manifest.jsonl")
    units = get_job_units(job_id, jobs_dir)
    total = len(units)
    done = sum(1 for u in units if u["unit_status"] == "completed")
    status = "completed" if done == total else ("pending" if done == 0 else "in_progress")
    for j in manifest:
        if j["job_id"] == job_id:
            j["status"] = status
            break
    write_jsonl(jobs_dir / "manifest.jsonl", manifest)


# ── Simple mode helpers ────────────────────────────────────────────────────────

def _chunk_labels_path(labels_dir: Path, chunk_id: str) -> Path:
    return labels_dir / f"{chunk_id}.jsonl"


def get_simple_labels(labels_dir: Path, chunk_id: str) -> list[dict]:
    return read_jsonl(_chunk_labels_path(labels_dir, chunk_id))


def get_completed_conv_ids_simple(labels_dir: Path, chunk_id: str) -> set[str]:
    return {
        rec["conv_id"]
        for rec in get_simple_labels(labels_dir, chunk_id)
        if rec.get("unit_status") == "completed"
    }


def save_simple_label(
    labels_dir: Path,
    chunk_id: str,
    conv_id: str,
    turns: list[dict],
    labeled_by: str = "default",
) -> None:
    labels_path = _chunk_labels_path(labels_dir, chunk_id)
    existing = read_jsonl(labels_path)

    updated = False
    for rec in existing:
        if rec["conv_id"] == conv_id:
            rec["unit_status"] = "completed"
            rec["labeled_by"] = labeled_by
            rec["labeled_at"] = now_iso()
            rec["turns"] = turns
            updated = True
            break

    if not updated:
        existing.append({
            "conv_id": conv_id,
            "chunk_id": chunk_id,
            "unit_status": "completed",
            "labeled_by": labeled_by,
            "labeled_at": now_iso(),
            "turns": turns,
        })
    write_jsonl(labels_path, existing)


def is_chunk_complete_simple(labels_dir: Path, chunk_id: str, conv_count: int) -> bool:
    done = len(get_completed_conv_ids_simple(labels_dir, chunk_id))
    return done >= conv_count


def cleanup_empty_and_completed_jobs(jobs_dir: Path) -> list[str]:
    deleted: list[str] = []
    for job in get_all_jobs(jobs_dir):
        if not job.get("chunk_ids"):
            delete_job(job["job_id"], jobs_dir)
            deleted.append(job["job_id"])
    return deleted


def get_job_conv_ids_filter_for_chunk(username: str, chunk_id: str, jobs_dir: Path) -> "list[str] | None":
    """Return the conv_ids filter list for a user's job containing chunk_id, or None if no filter."""
    for job in get_user_jobs(username, jobs_dir):
        if chunk_id in job.get("chunk_ids", []):
            conv_filter = job.get("conv_ids_filter", {})
            return conv_filter.get(chunk_id)
    return None


# ── Bundle helpers ─────────────────────────────────────────────────────────────

_BUNDLES_FILE = "bundles.jsonl"


def create_bundle(
    name: str,
    chunk_ids: list[str],
    chunk_conv_map: dict[str, list[dict]],
    jobs_dir: Path,
    conv_ids_filter: "dict[str, list[str]] | None" = None,
    eligible_labelers: "list[str] | None" = None,
    original_labeler: "str | None" = None,
) -> str:
    bundle_id = f"bundle_{uuid.uuid4().hex[:8]}"
    if conv_ids_filter:
        n_convs = sum(len(v) for v in conv_ids_filter.values())
    else:
        n_convs = sum(len(chunk_conv_map.get(c, [])) for c in chunk_ids)
    entry: dict = {
        "bundle_id": bundle_id,
        "name": name,
        "chunk_ids": chunk_ids,
        "n_convs": n_convs,
        "created_at": now_iso(),
        "claimed_by": None,
        "claimed_at": None,
        "job_id": None,
    }
    if conv_ids_filter:
        entry["conv_ids_filter"] = conv_ids_filter
    if eligible_labelers is not None:
        entry["eligible_labelers"] = eligible_labelers
    if original_labeler is not None:
        entry["original_labeler"] = original_labeler
    append_jsonl(jobs_dir / _BUNDLES_FILE, entry)
    return bundle_id


def get_all_bundles(jobs_dir: Path) -> list[dict]:
    return [b for b in read_jsonl(jobs_dir / _BUNDLES_FILE) if b.get("chunk_ids")]


def get_unclaimed_bundles(jobs_dir: Path) -> list[dict]:
    return [b for b in get_all_bundles(jobs_dir) if not b.get("claimed_by")]


def claim_bundle(
    bundle_id: str,
    username: str,
    chunk_conv_map: dict[str, list[dict]],
    jobs_dir: Path,
) -> str:
    bundles = get_all_bundles(jobs_dir)
    job_id = None
    for bundle in bundles:
        if bundle["bundle_id"] == bundle_id:
            if bundle.get("claimed_by"):
                raise ValueError(f"Bundle already claimed by {bundle['claimed_by']}")
            conv_ids_filter = bundle.get("conv_ids_filter") or None
            job_id = create_job(username, bundle["chunk_ids"], chunk_conv_map, jobs_dir,
                                conv_ids_filter=conv_ids_filter)
            bundle["claimed_by"] = username
            bundle["claimed_at"] = now_iso()
            bundle["job_id"] = job_id
            break
    else:
        raise ValueError(f"Bundle {bundle_id} not found")
    write_jsonl(jobs_dir / _BUNDLES_FILE, bundles)
    return job_id


def delete_bundle(bundle_id: str, jobs_dir: Path) -> None:
    bundles = get_all_bundles(jobs_dir)
    write_jsonl(jobs_dir / _BUNDLES_FILE, [b for b in bundles if b["bundle_id"] != bundle_id])
