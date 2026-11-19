"""Register the Manga Live Translator native messaging host for Edge and Chrome.

Writes <repo>/cache/com.mangalivetranslator.host.json and points
HKCU\\Software\\{Microsoft\\Edge,Google\\Chrome}\\NativeMessagingHosts\\<name>
at it. HKCU only, no admin rights needed.

Usage: python native_host/register_host.py [--extension-id ID ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

HOST_NAME = "com.mangalivetranslator.host"
FALLBACK_EXTENSION_ID = "gkljnbldlglhdijoanaeenhpfkhcahch"
ID_RE = re.compile(r"^[a-p]{32}$")
REGISTRY_KEYS = (
    r"Software\Microsoft\Edge\NativeMessagingHosts",
    r"Software\Google\Chrome\NativeMessagingHosts",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def default_user_data_roots() -> list:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return []
    base = Path(local)
    return [
        base / "Microsoft" / "Edge" / "User Data",
        base / "Google" / "Chrome" / "User Data",
    ]


def _ids_from_prefs(prefs_file: Path, extension_dir: str) -> list:
    try:
        data = json.loads(prefs_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    settings = (data.get("extensions") or {}).get("settings") or {}
    found = []
    if not isinstance(settings, dict):
        return found
    for ext_id, entry in settings.items():
        if not isinstance(entry, dict) or not ID_RE.match(str(ext_id)):
            continue
        raw = entry.get("path")
        if not isinstance(raw, str) or not raw:
            continue
        candidate = raw if os.path.isabs(raw) else str(prefs_file.parent / raw)
        if _norm(candidate) == extension_dir:
            found.append(str(ext_id))
    return found


def discover_extension_ids(repo_root: Path = REPO_ROOT, user_data_roots: Optional[Iterable[Path]] = None) -> list:
    """Unpacked extension IDs whose path resolves to <repo>/extension, sorted, deduplicated."""
    extension_dir = _norm(str(Path(repo_root) / "extension"))
    roots = list(default_user_data_roots() if user_data_roots is None else user_data_roots)
    ids = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            profiles = [p for p in root.iterdir() if p.is_dir()]
        except OSError:
            continue
        for profile in profiles:
            for name in ("Secure Preferences", "Preferences"):
                prefs = profile / name
                if prefs.is_file():
                    ids.update(_ids_from_prefs(prefs, extension_dir))
    return sorted(ids)


def build_host_manifest(repo_root: Path = REPO_ROOT, extension_ids: Optional[Iterable[str]] = None) -> dict:
    ids = [i for i in (extension_ids or []) if ID_RE.match(i)]
    if not ids:
        ids = [FALLBACK_EXTENSION_ID]
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return {
        "name": HOST_NAME,
        "description": "Manga Live Translator local server launcher",
        "path": str(Path(repo_root).resolve() / "native_host" / "host.cmd"),
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{i}/" for i in seen],
    }


def manifest_path(repo_root: Path = REPO_ROOT) -> Path:
    return Path(repo_root).resolve() / "cache" / f"{HOST_NAME}.json"


def write_manifest(repo_root: Path, manifest: dict) -> Path:
    path = manifest_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def register_native_host(
    repo_root: Path = REPO_ROOT,
    extension_ids: Optional[Iterable[str]] = None,
    user_data_roots: Optional[Iterable[Path]] = None,
    winreg_module=None,
) -> dict:
    """Discover IDs, write the manifest, register HKCU keys. Returns a summary dict."""
    repo_root = Path(repo_root).resolve()
    ids = list(extension_ids or [])
    for found in discover_extension_ids(repo_root, user_data_roots):
        if found not in ids:
            ids.append(found)
    manifest = build_host_manifest(repo_root, ids)
    path = write_manifest(repo_root, manifest)

    registered = []
    reg = winreg_module
    if reg is None and os.name == "nt":
        import winreg as reg  # type: ignore
    if reg is not None:
        for base in REGISTRY_KEYS:
            key = reg.CreateKeyEx(reg.HKEY_CURRENT_USER, base + "\\" + HOST_NAME, 0, reg.KEY_SET_VALUE)
            try:
                reg.SetValueEx(key, "", 0, reg.REG_SZ, str(path))
            finally:
                reg.CloseKey(key)
            registered.append("HKCU\\" + base + "\\" + HOST_NAME)
    return {
        "manifest_path": str(path),
        "extension_ids": [o[len("chrome-extension://"):-1] for o in manifest["allowed_origins"]],
        "registry_keys": registered,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-id", action="append", default=[], help="extra extension ID to allow")
    args = parser.parse_args(argv)
    bad = [i for i in args.extension_id if not ID_RE.match(i)]
    if bad:
        print("Invalid extension ID(s): " + ", ".join(bad), file=sys.stderr)
        return 2
    result = register_native_host(REPO_ROOT, args.extension_id)
    print("Native messaging host manifest: " + result["manifest_path"])
    print("Allowed extension IDs: " + ", ".join(result["extension_ids"]))
    for key in result["registry_keys"]:
        print("Registered: " + key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
