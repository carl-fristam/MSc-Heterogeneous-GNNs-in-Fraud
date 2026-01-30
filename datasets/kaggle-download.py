import shutil
from pathlib import Path

import kagglehub

DATASET_ID = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

if __name__ == "__main__":
    print("Downloading dataset from Kaggle...")
    cache_path = Path(kagglehub.dataset_download(DATASET_ID))

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for file in cache_path.iterdir():
        dest = DATASETS_DIR / file.name
        shutil.copy2(file, dest)
        print(f"Copied: {file.name}")

    print(f"Dataset ready at: {DATASETS_DIR}")