
from pathlib import Path
from urllib.request import urlretrieve
import hashlib

URL = (
    "https://pub-f6328c385dad4f2e886cc52369b01984.r2.dev/"
    "dvc/files/md5/70/fa01ee10da1bd9b17fe18234356bfa"
)
DEST = Path("data/raw/dataset.csv")
EXPECTED_MD5 = "70fa01ee10da1bd9b17fe18234356bfa"

DEST.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(URL, DEST)

actual_md5 = hashlib.md5(DEST.read_bytes()).hexdigest()

if actual_md5 != EXPECTED_MD5:
    DEST.unlink(missing_ok=True)
    raise RuntimeError("Dataset MD5 mismatch; downloaded file was removed.")

print(f"Dataset ready: {DEST}")
