# Sleep Stage Classification — Sleep-EDF Expanded

Automatic 5-class sleep-stage classification (Wake, N1, N2, N3, REM) from
polysomnography (EEG, EOG, EMG), using the [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
dataset (22-patient ST telemetry subset).

## Pipeline

1. **Load** raw EDF signals + hypnogram annotations per patient (`src/data_loading.py`)
2. **Preprocess**: bandpass filtering (`src/preprocessing.py`)
3. **Extract features**: ~1000 time/frequency-domain features per 30s epoch
   via tsflex + tsfresh + antropy + yasa band power, at 30s/60s/90s windows,
   plus adjacent-epoch context features (`src/feature_extraction.py`,
   `src/feature_engineering.py`)
4. **Select features**: correlation filter + [PowerShap](https://github.com/predict-idlab/powershap)
   statistical selection, fit fresh inside every CV fold — no leakage from
   held-out patients (`src/feature_selection.py`)
5. **Model**: CatBoost with early stopping, balanced class weights, and a
   macro-F1-aligned eval metric, tuned via a grouped hyperparameter search;
   compared against a linear SGD baseline (`src/modeling.py`)
6. **Evaluate**: nested group-k-fold cross-validation (patients never split
   across train/test within a fold) — `src/evaluation.py`

## Setup

```bash
pip install -r requirements.txt
```

### Dataset

The pipeline expects the raw [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
dataset to live at `data/sleep-edf-database-expanded-1.0.0/`, matching the
structure the official zip extracts to:

```
data/
  sleep-edf-database-expanded-1.0.0/
    sleep-cassette/
      SC4001E0-PSG.edf
      SC4001EC-Hypnogram.edf
      ...
    sleep-telemetry/          <- this is the 22-patient subset config.py uses by default
      ST7011J0-PSG.edf
      ST7011JP-Hypnogram.edf
      ...
    SC-subjects.xls
    ST-subjects.xls
    RECORDS
    SHA256SUMS.txt
```

`data/` is git-ignored — the dataset is ~8GB and never gets committed.

**Fresh download:**

```bash
mkdir -p data
wget https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip -P data
unzip data/sleep-edf-database-expanded-1.0.0.zip -d data
rm data/sleep-edf-database-expanded-1.0.0.zip
```

**Already have it downloaded elsewhere?** Either move it in, or symlink it to
avoid duplicating an 8GB folder:

```bash
# move
mv /path/to/your/sleep-edf-database-expanded-1.0.0 data/

# or symlink (safer if other projects also use this copy)
ln -s /path/to/your/sleep-edf-database-expanded-1.0.0 data/sleep-edf-database-expanded-1.0.0
```

If your existing copy uses a different folder name or lives at a different
path entirely, either rename/symlink it to match the layout above, or edit
`DATA_DIR` in `config.py` to point at it directly.

## Run

```bash
python scripts/01_build_feature_dataset.py    # raw EDF -> features/*.parquet (slow, run once)
python scripts/02_hyperparameter_search.py    # -> results/best_hyperparams.json
python scripts/03_train_evaluate.py           # -> results/metrics.json, confusion_matrix.png
```

## Results

_Populated after running the scripts above — see `results/metrics.json`._

| Model | Test Macro F1 | Test Balanced Accuracy |
|---|---|---|
| Linear (SGD) baseline | — | — |
| CatBoost (tuned) | — | — |

## Repo structure

```
config.py                  # paths & constants
src/
  data_loading.py           # EDF/annotation loading
  preprocessing.py          # bandpass filtering
  feature_extraction.py     # tsflex/tsfresh/antropy/yasa feature collection
  feature_engineering.py    # band ratios, shift-context features, label cleanup
  feature_selection.py      # correlation filter + PowerShap
  modeling.py                # CatBoost (tuned) + linear baseline
  evaluation.py               # nested group-CV runner
  visualization.py            # confusion matrix / learning curve plots
scripts/
  01_build_feature_dataset.py
  02_hyperparameter_search.py
  03_train_evaluate.py
data/                       # raw dataset (git-ignored, ~8GB — see Dataset section above)
features/                   # cached feature parquet (git-ignored, regenerable)
results/                    # metrics.json, best_hyperparams.json (small, committed)
figures/                    # confusion_matrix.png etc. (small, committed — these feed the website showcase)
```

## Notes on methodology

- All cross-validation is **grouped by patient** (`GroupKFold`/
  `GroupShuffleSplit`) — no patient's epochs ever appear in both train and
  test within a fold.
- Feature selection and the CatBoost early-stopping validation split are
  both fit on **training-fold data only**, refreshed for every outer fold,
  to avoid any leakage from held-out test patients into feature choice or
  model selection.
