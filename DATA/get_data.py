# get_data.py: Read DATA/data_manifest.csv to find one Google Drive file (raw.zip) and downloads it, checks it is intact, and unpacks it into DATA/raw/

import csv, hashlib, sys, shutil, zipfile
from pathlib import Path

RAW_DIR = Path("DATA/raw")
ZIP_PATH = Path("DATA/raw.zip")
MANIFEST = Path("DATA/data_manifest.csv")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_with_gdown(file_id: str, dest: Path):
    try:
        import gdown
    except Exception:
        sys.exit("Please `pip install gdown` first (or `pip install -r requirements.txt`).")
    url = f"https://drive.google.com/uc?id={file_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading -> {dest}")
    gdown.download(url, str(dest), quiet=False)

def safe_extract_zip(zip_path: Path, target_dir: Path):
    tmp_root = target_dir.parent / (target_dir.name + "_tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {tmp_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_root)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_root.rename(target_dir)
    print(f"Ready: {target_dir}")

def main():
    if not MANIFEST.exists():
        sys.exit("Missing DATA/data_manifest.csv.")
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    if not rows:
        sys.exit("data_manifest.csv is empty.")
    row = rows[0]

    file_id = row["id"].strip()
    want_sha = row["sha256"].strip().lower()
    want_bytes = int(row["bytes"])
    dest_zip = Path(row["dest"])

    if dest_zip.exists() and dest_zip.stat().st_size != want_bytes:
        print("Local zip size mismatch; re-downloading.")
        dest_zip.unlink()
    if not dest_zip.exists():
        download_with_gdown(file_id, dest_zip)

    got_sha = sha256(dest_zip)
    if got_sha != want_sha:
        dest_zip.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {dest_zip}\n expected {want_sha}\n got      {got_sha}")
    if dest_zip.stat().st_size != want_bytes:
        dest_zip.unlink(missing_ok=True)
        raise SystemExit(f"Size mismatch for {dest_zip}.")

    print(f"Verified zip: {dest_zip}")
    safe_extract_zip(dest_zip, RAW_DIR)
    print("All done.")

if __name__ == "__main__":
    main()