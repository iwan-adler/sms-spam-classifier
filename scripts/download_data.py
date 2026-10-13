"""Download the UCI SMS Spam Collection and verify its checksum.

Usage: python scripts/download_data.py

Source: UCI Machine Learning Repository, "SMS Spam Collection" (Almeida & Hidalgo, 2011),
licensed CC BY 4.0. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
TARGET = Path(__file__).resolve().parent.parent / "data" / "raw" / "sms+spam+collection.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if TARGET.is_file() and sha256(TARGET.read_bytes()) == SHA256:
        print(f"Already downloaded and verified: {TARGET}")
        return 0
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = sha256(payload)
    if digest != SHA256:
        # A different file would silently change every result, so refuse it.
        print(f"Checksum mismatch: expected {SHA256}, got {digest}. Not saving.", file=sys.stderr)
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(payload)
    print(f"Saved {len(payload):,} bytes to {TARGET} (SHA-256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
