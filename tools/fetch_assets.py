#!/usr/bin/env python3
"""Download the game's free third-party assets into ./assets.

    python tools/fetch_assets.py              2k textures, ~110 MB
    python tools/fetch_assets.py --low        1k textures, ~35 MB
    python tools/fetch_assets.py --check      report what is missing, download nothing

Every file is CC0 (Kenney, Poly Haven) or SIL OFL (Google Fonts) - see
rally/assets_manifest.py. Files already on disk are skipped, so re-running is
cheap and resumes an interrupted download. A CREDITS.md listing every file and
its licence is written alongside.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rally.assets_manifest import Asset, manifest  # noqa: E402

ASSETS = ROOT / "assets"
USER_AGENT = "OchrePassRally-asset-fetch/2.0 (+https://github.com)"


def _ssl_context() -> ssl.SSLContext | None:
    """A verifying TLS context that works on a stock python.org macOS install.

    That installer ships without a CA bundle until its "Install Certificates"
    script is run, so the default context rejects every HTTPS server. certifi
    carries its own bundle; if it is missing we return None and fall back to
    the system curl, which uses the OS trust store. Verification is never
    switched off.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


SSL_CONTEXT = _ssl_context()
CURL = shutil.which("curl")

# First bytes each file type must start with. A CDN error page saved under a
# .jpg name would otherwise only surface later as a baffling texture failure.
MAGIC = {
    ".glb": b"glTF",
    ".png": b"\x89PNG",
    ".jpg": b"\xff\xd8",
    ".ogg": b"OggS",
    ".ttf": (b"\x00\x01\x00\x00", b"OTTO", b"true"),
    ".hdr": (b"#?RADIANCE", b"#?RGBE"),
}


def _valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    magic = MAGIC.get(path.suffix.lower())
    if magic is None:
        return True
    head = path.read_bytes()[:16]
    options = magic if isinstance(magic, tuple) else (magic,)
    return any(head.startswith(m) for m in options)


def _download(asset: Asset, attempts: int = 3) -> tuple[Asset, str]:
    dest = ASSETS / asset.dest
    if _valid(dest):
        return asset, "cached"

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    last_error = ""
    for attempt in range(attempts):
        try:
            if SSL_CONTEXT is None and CURL:
                subprocess.run([CURL, "-fsSL", "--max-time", "120", "-A", USER_AGENT,
                                "-o", str(partial), asset.url], check=True)
            else:
                request = urllib.request.Request(asset.url,
                                                 headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60,
                                            context=SSL_CONTEXT) as response, \
                        open(partial, "wb") as handle:
                    while True:
                        block = response.read(1 << 16)
                        if not block:
                            break
                        handle.write(block)
            partial.replace(dest)
            if not _valid(dest):
                dest.unlink(missing_ok=True)
                raise ValueError("downloaded file is not the expected type")
            return asset, "downloaded"
        except (urllib.error.URLError, OSError, ValueError,
                subprocess.CalledProcessError) as error:
            last_error = str(error)
            partial.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    return asset, f"FAILED: {last_error}"


def write_credits(assets: list[Asset]) -> None:
    groups: dict[tuple[str, str], list[Asset]] = {}
    for asset in assets:
        if asset.dest.startswith("licences/"):
            continue
        groups.setdefault((asset.author, asset.licence), []).append(asset)

    lines = ["# Credits", "",
             "Every third-party asset in Ochre Pass Rally is free. Kenney and Poly "
             "Haven assets are CC0 (public domain): no attribution is required, but "
             "they are credited here anyway. Fonts are under the SIL Open Font "
             "License; its text is in `licences/`.", ""]
    for (author, licence), items in sorted(groups.items()):
        lines.append(f"## {author} - {licence}")
        lines.append("")
        sources = sorted({a.source for a in items})
        for source in sources:
            files = sorted(a.dest for a in items if a.source == source)
            lines.append(f"- {source} - {len(files)} file(s)")
        lines.append("")
    (ASSETS / "CREDITS.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--low", action="store_true", help="1k textures (smaller download)")
    parser.add_argument("--check", action="store_true", help="only report missing files")
    args = parser.parse_args()

    assets = manifest("low" if args.low else "high")
    missing = [a for a in assets if not _valid(ASSETS / a.dest)]

    if args.check:
        print(f"{len(assets) - len(missing)}/{len(assets)} assets present")
        for asset in missing[:20]:
            print("  missing", asset.dest)
        return 0 if not missing else 1

    print(f"Fetching {len(missing)} of {len(assets)} assets into {ASSETS} ...")
    failures = []
    done = 0
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        for asset, status in pool.map(_download, assets):
            if status.startswith("FAILED"):
                failures.append((asset, status))
            elif status == "downloaded":
                done += 1
                print(f"  [{done:>3}] {asset.dest}")

    write_credits(assets)
    size = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file()) / 1e6
    print(f"\nassets/ now holds {size:.0f} MB. CREDITS.md written.")
    if failures:
        print(f"\n{len(failures)} file(s) failed - re-run to retry:")
        for asset, status in failures:
            print(f"  {asset.dest}: {status}")
        return 1
    print("All assets present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
