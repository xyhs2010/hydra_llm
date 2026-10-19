"""Fetch the Paul Graham essay corpus used as RULER's haystack.

RULER's own downloader lives in the cloned NVIDIA/RULER repo; this is a small
equivalent so the corpus can be fetched without executing third-party code.

The URL list mixes two hosts:
  - www.paulgraham.com  -> reachable directly
  - github.com/.../raw  -> redirects to raw.githubusercontent.com, which is only
                           reachable through the local proxy

Every URL gets an entry in fetch_manifest.json, so the success count is always
auditable. Output: <out_dir>/PaulGrahamEssays.json ({"text": ...}), matching
RULER's format.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import html2text
from bs4 import BeautifulSoup

PROXY = "http://127.0.0.1:7890"


def _opener_for(url: str, use_proxy: bool):
    proxied = use_proxy or "github.com" in url or "githubusercontent.com" in url
    handler = urllib.request.ProxyHandler(
        {"http": PROXY, "https": PROXY} if proxied else {}
    )
    return urllib.request.build_opener(handler)


def fetch(url: str, *, timeout: float = 10.0, use_proxy: bool = False) -> str:
    with _opener_for(url, use_proxy).open(url, timeout=timeout) as resp:
        raw = resp.read()
    # Older paulgraham.com pages are cp1252; try UTF-8 first, then fall back.
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def to_text(page: str, h: html2text.HTML2Text) -> str | None:
    tag = BeautifulSoup(page, "html.parser").find("font")
    if tag is None:
        return None
    return h.handle(str(tag))


def fetch_one(url: str, h: html2text.HTML2Text) -> dict:
    name = url.rsplit("/", 1)[-1]
    errors = []
    # Direct first; a few pages intermittently refuse direct connections, so
    # fall back to the (paid) proxy only for those.
    for use_proxy in (False, True):
        try:
            page = fetch(url, use_proxy=use_proxy)
            text = to_text(page, h) if ".html" in url else page
            if text is None:
                raise ValueError("no <font> tag")
            return {"url": url, "file": name, "ok": True, "chars": len(text), "proxy": use_proxy, "text": text}
        except Exception as exc:  # noqa: BLE001 - record and try next route
            errors.append(f"{'proxy' if use_proxy else 'direct'}: {type(exc).__name__}: {exc}")
    return {"url": url, "file": name, "ok": False, "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--urls",
        type=Path,
        default=Path("data/raw/ruler_repo/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"),
    )
    ap.add_argument("--out_dir", type=Path, default=Path("data/raw/ruler_corpus"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_tables = True
    h.escape_all = True
    h.reference_links = False
    h.mark_code = False

    urls = [u.strip() for u in args.urls.read_text().splitlines() if u.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, u, h): u for u in urls}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"FAIL [{i}/{len(urls)}] {r['file']}: {r['errors']}")
            elif i % 25 == 0:
                print(f"  {i}/{len(urls)}")

    results.sort(key=lambda r: r["url"])
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    text = "\n\n".join(r["text"] for r in ok)

    out = args.out_dir / "PaulGrahamEssays.json"
    out.write_text(json.dumps({"text": text}, ensure_ascii=False))

    manifest = {
        "total_urls": len(urls),
        "ok": len(ok),
        "failed": len(failed),
        "chars": len(text),
        "via_proxy": sum(1 for r in ok if r["proxy"]),
        "failed_files": [r["file"] for r in failed],
        "files": [
            {"file": r["file"], "url": r["url"], "ok": r["ok"], "chars": r.get("chars", 0), "proxy": r.get("proxy", False)}
            for r in results
        ],
    }
    (args.out_dir / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"wrote {out}  ok={len(ok)}/{len(urls)} failed={len(failed)} "
        f"via_proxy={manifest['via_proxy']} chars={len(text):,}"
    )


if __name__ == "__main__":
    main()
