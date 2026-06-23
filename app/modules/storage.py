from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

_GH_API = "https://api.github.com"
_APP_ROOT = Path(__file__).resolve().parents[2]
_cache: dict[str, tuple[str | None, str | None]] = {}


@st.cache_resource
def _gh_config() -> dict | None:
    try:
        cfg = st.secrets.get("github", {})
        token = cfg.get("token", "")
        repo = cfg.get("repo", "")
        # Accept both "owner/repo" and full URLs like "https://github.com/owner/repo.git"
        if "github.com" in repo:
            repo = repo.rstrip("/").removesuffix(".git").split("github.com/")[-1]
        if token and repo:
            return {
                "token": token,
                "repo": repo,
                "branch": cfg.get("branch", "main"),
                "prefix": cfg.get("prefix", "labels").strip("/"),
            }
    except Exception:
        pass
    return None


def _use_cloud() -> bool:
    return _gh_config() is not None


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_gh_config()['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_path(path: str | Path) -> str:
    cfg = _gh_config()
    try:
        rel = Path(path).resolve().relative_to(_APP_ROOT)
    except ValueError:
        rel = Path(str(path).lstrip("/"))
    p = str(rel)
    return f"{cfg['prefix']}/{p}" if cfg["prefix"] else p


def _gh_get(path: str | Path) -> tuple[str | None, str | None]:
    key = str(path)
    if key in _cache:
        return _cache[key]
    cfg = _gh_config()
    url = f"{_GH_API}/repos/{cfg['repo']}/contents/{_gh_path(path)}"
    resp = requests.get(url, headers=_headers(), params={"ref": cfg["branch"]})
    if resp.status_code == 404:
        _cache[key] = (None, None)
        return None, None
    resp.raise_for_status()
    data = resp.json()
    result = base64.b64decode(data["content"]).decode("utf-8"), data["sha"]
    _cache[key] = result
    return result


def _gh_put(path: str | Path, content: str, sha: str | None = None) -> None:
    cfg = _gh_config()
    url = f"{_GH_API}/repos/{cfg['repo']}/contents/{_gh_path(path)}"
    body: dict = {
        "message": f"update {path}",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": cfg["branch"],
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(url, headers=_headers(), json=body)
    resp.raise_for_status()
    new_sha = resp.json().get("content", {}).get("sha")
    _cache[str(path)] = (content, new_sha)


# ── Public API ─────────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> None:
    if not _use_cloud():
        Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> dict | list:
    if _use_cloud():
        content, _ = _gh_get(path)
        return json.loads(content) if content else {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict | list) -> None:
    if _use_cloud():
        _, sha = _gh_get(path)
        _gh_put(path, json.dumps(data, indent=2, ensure_ascii=False), sha)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict]:
    if _use_cloud():
        content, _ = _gh_get(path)
        if not content:
            return []
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict) -> None:
    if _use_cloud():
        existing, sha = _gh_get(path)
        _gh_put(path, (existing or "") + json.dumps(record, ensure_ascii=False) + "\n", sha)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    if _use_cloud():
        _, sha = _gh_get(path)
        _gh_put(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), sha)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_cache() -> None:
    """Evict all in-process GitHub content cache entries.

    Call this after an external git push so the app re-fetches fresh content
    from the GitHub API instead of serving stale cached state.
    """
    _cache.clear()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
