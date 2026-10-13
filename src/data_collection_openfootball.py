"""OpenFootball EPL Data Ingestion and Verification Pipeline.

Retrieves official Public-Domain (CC0-1.0) match records from:
https://github.com/openfootball/england

Covers 5 English Premier League seasons:
- 2019/20, 2020/21, 2021/22, 2022/23, 2023/24

Stores untouched raw source files under data/raw/openfootball/ and records
cryptographic checksums (SHA-256) and ingestion provenance.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_ROOT / "data" / "raw" / "openfootball"
AUDIT_DIR = WORKSPACE_ROOT / "data" / "audit"

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

BASE_URL = "https://raw.githubusercontent.com/openfootball/england/master"
REPO_URL = "https://github.com/openfootball/england"
LICENSE_URL = "https://raw.githubusercontent.com/openfootball/england/master/LICENSE.md"


def download_and_hash_file(url: str, dest_path: Path) -> dict:
    """Download a raw file from GitHub and compute its SHA-256 digest."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EPL-Research-OpenFootball-Migrator/1.0"}
    )
    with urllib.request.urlopen(req) as response:
        content = response.read()

    with open(dest_path, "wb") as f:
        f.write(content)

    sha256_hash = hashlib.sha256(content).hexdigest()
    return {
        "url": url,
        "path": str(dest_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "size_bytes": len(content),
        "sha256": sha256_hash,
    }


def ingest_openfootball_raw_data() -> dict:
    """Download all 5 seasons of OpenFootball Premier League text files."""
    print("=" * 70)
    print("STARTING OPENFOOTBALL RAW DATA INGESTION")
    print(f"Source Repository: {REPO_URL}")
    print("Licence: Creative Commons Zero 1.0 Universal (CC0-1.0)")
    print("=" * 70)

    provenance_records = {
        "source_repository": REPO_URL,
        "licence": "CC0-1.0 Public Domain Dedication",
        "licence_url": LICENSE_URL,
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    # 1. Download License
    license_path = RAW_DIR / "LICENSE.md"
    lic_meta = download_and_hash_file(LICENSE_URL, license_path)
    provenance_records["files"]["LICENSE.md"] = lic_meta
    print(f"Downloaded LICENSE.md -> SHA256: {lic_meta['sha256'][:12]}...")

    # 2. Download all 5 season files
    for season in SEASONS:
        file_url = f"{BASE_URL}/{season}/1-premierleague.txt"
        dest_file = RAW_DIR / season / "1-premierleague.txt"
        meta = download_and_hash_file(file_url, dest_file)
        meta["season"] = season
        provenance_records["files"][f"{season}/1-premierleague.txt"] = meta
        print(f"Downloaded {season}/1-premierleague.txt ({meta['size_bytes']} bytes) -> SHA256: {meta['sha256'][:12]}...")

    # 3. Save provenance audit log
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    prov_path = AUDIT_DIR / "openfootball_ingestion_manifest.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance_records, f, indent=2)

    print(f"\nSaved ingestion manifest to: {prov_path}")
    print("OPENFOOTBALL RAW INGESTION COMPLETE")
    print("=" * 70)
    return provenance_records


if __name__ == "__main__":
    ingest_openfootball_raw_data()
