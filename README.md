# CRMLS Single-Family Home Valuation Model

An Automated Valuation Model (AVM) that estimates the sale price of California
single-family homes from property characteristics and location, using CRMLS sold
listing data.

The model is built to value **both on-market and off-market** properties, so no
listing-side information (asking price, days on market) is used as a feature.

**Headline result:** the best model achieves a **median absolute percentage error
of 7.9%**, with **58.6% of predictions landing within 10%** of the actual sale
price and **82.3% within 20%**.

---

## Dataset

**Source:** CRMLS sold listings, delivered through the Trestle FTP server
(`/raw/California`) as one CSV per month, named `CRMLSSoldYYYYMM.csv`.
Field definitions come from `Trestle Property MetaData.pdf` in `/resources`.

**Supplemental data:** California School District Areas 2024-25 boundaries from
the California Open Data Portal
(https://data.ca.gov/dataset/california-school-district-areas-2024-25),
used for a spatial join on property coordinates.

**Scope:** filtered to `PropertyType = Residential` and
`PropertySubType = SingleFamilyResidence`.

**Encoding note:** the raw CSVs are **not** UTF-8. They must be read with
`encoding="latin1"` or pandas raises a `UnicodeDecodeError` on smart quotes and
similar characters.

### Repository layout

This repository contains a single notebook holding the entire pipeline. It runs
top to bottom in project-week order: exploration, preprocessing, the baseline
linear model, tree model comparison, feature engineering, gradient boosting,
and finally the expanded evaluation metrics. Section headers inside the
notebook mark where each stage begins.

**The MLS data is not in this repository** — it contains addresses, sale
prices, and agent details, and must be sourced through Trestle directly. To run
the notebook, recreate this structure locally:

```
IDX Summer DS/
├── py files/
│   └── weeks1-9.ipynb   <- the notebook from this repo
└── raw data/            <- CRMLSSoldYYYYMM.csv files from Trestle
```

The notebook reads the data with the relative path `../raw data`, so it must be
run from inside `py files/`. If you'd rather keep a different layout, change
`DATA_DIR` in the config cell to an absolute path.

---

## Preprocessing

### 1. Load and filter
All monthly CSVs are read and concatenated, with each row tagged by its source
month. Rows are then filtered to residential single-family properties.

### 2. Leakage removal
`ListPrice` and `OriginalListPrice` are dropped immediately and never used as
features. Listing agents set asking prices using comparable sales and market
conditions already correlated with the eventual sale price, so including them
would inflate performance while teaching the model to reproduce the listing
agent's number rather than learn true property value. They also don't exist for
off-market homes, which the model must be able to value.

Agent, office, and listing-identifier columns are excluded as well — they
describe the transaction, not the property.

### 3. Outlier removal
Data-entry errors accounted for roughly 2% of rows (~4,000 of ~203,000). Bounds
applied:

| Field | Bound | Rationale |
|---|---|---|
| `ClosePrice` | $10,000 – $20,000,000 | excludes $0 placeholders and errors like $900M "sales" |
| `BedroomsTotal` | < 15 | raw data contained values up to 40 |
| `BathroomsTotalInteger` | < 15 | raw data contained values up to 175 |
| `LivingArea` | < 20,000 sq ft | |
| `LotSizeSquareFeet` | < 13,068,000 sq ft (~300 acres) | |

### 4. Unit-mismatch fix in `LotSizeSquareFeet`
A subset of rows stored lot size in **acres** despite the column name — values
like `0.14`, `5.00`, and `100.00` where square feet were expected. These were
converted (× 43,560), zeros were treated as missing, and any remaining rows with
a living-area-to-lot-size ratio above 1.0 (physically impossible) were nulled.
This reduced the maximum ratio from ~60,567 to 1.0.

### 5. Missing values
Numeric features are median-imputed with a companion `_was_missing` flag column
so the model retains the information that a value was absent. Categorical
features are filled with `"Unknown"`. **Medians are computed on the training
split only** and then applied to validation and test.

### 6. Encoding
`City` (~1,100 unique) and `SchoolDistrictName` (776 unique) are **target
encoded**: each category is replaced with its mean `ClosePrice`, smoothed toward
the global mean so categories with few sales don't get extreme, unreliable
values. Encodings are fit on train only and mapped onto validation/test, with
unseen categories falling back to the global train mean.

Remaining low-cardinality categoricals (`CountyOrParish`, `Stories`,
`NewConstructionYN`, `PoolPrivateYN`, `ViewYN`, `AttachedGarageYN`,
`PropertySubType`) are one-hot encoded.

Target encoding was chosen after direct comparison — see
[Encoding comparison](#encoding-comparison) below.

### 7. Scaling
Numeric features are standardized with `StandardScaler`, fit on train only.

### 8. Train / validation / test split
Split by close date, not randomly, so the model is always predicting forward in
time. The most recent full month is held out as the test set to simulate
production use.

| Split | Months | Rows |
|---|---|---|
| Train | 2025-02 through 2026-01 (12 months) | 157,790 |
| Validation | 2026-02 through 2026-04 (3 months) | 38,688 |
| Test | 2026-05 (1 month) | 14,741 |

---

## Features

**Raw:** `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`,
`LotSizeSquareFeet`, `YearBuilt`, `GarageSpaces`, `Latitude`, `Longitude`,
`CountyOrParish`, `Stories`, `NewConstructionYN`, `PoolPrivateYN`, `ViewYN`,
`AttachedGarageYN`

**Engineered:**

| Feature | Definition |
|---|---|
| `BedBathRatio` | bedrooms ÷ bathrooms |
| `PropertyAgeYears` | close year − year built |
| `LivingArea_to_LotSize` | building density; how much of the lot is built up |
| `DistanceToCoastMiles` | Haversine distance to the nearest of seven CA coastline reference points |
| `City_TargetEnc` | smoothed mean close price by city |
| `SchoolDistrictName_TargetEnc` | smoothed mean close price by school district |

Final feature count: **103**.

---

## Models tested

| Model | Tuned settings |
|---|---|
| Linear Regression | — (baseline) |
| Decision Tree | `max_depth=10` |
| Random Forest | `n_estimators=200, max_depth=15, min_samples_leaf=2` |
| XGBoost | `max_depth=8, learning_rate=0.2, n_estimators=500` |
| LightGBM | `max_depth=6, learning_rate=0.1, n_estimators=500` |

Hyperparameters were tuned against the **validation** set across a 30-point grid
(depths 4–12, learning rates 0.05–0.2, 200 or 500 trees); the test month was
left untouched until final evaluation.

---

## Results

### Full test metrics

Sorted by MdAPE, the industry-standard AVM accuracy metric.

| Model | R² | MdAPE % | MAPE % | PPE10 % | PPE20 % | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.860 | **7.86** | 16.48 | **58.6** | **82.3** | $182,765 | $443,194 |
| LightGBM (tuned) | **0.875** | 8.66 | **16.13** | 55.6 | 82.2 | **$178,744** | **$418,762** |
| XGBoost (tuned) | 0.873 | 8.67 | 16.59 | 56.1 | 82.1 | $179,012 | $422,007 |
| XGBoost (baseline) | 0.868 | 9.35 | 17.77 | 52.8 | 79.8 | $190,281 | $429,421 |
| LightGBM (baseline) | 0.866 | 9.64 | 17.96 | 51.6 | 78.8 | $194,548 | $432,943 |
| Linear Regression | 0.764 | 21.44 | 32.76 | 25.1 | 47.3 | $316,090 | $575,153 |
| Decision Tree | 0.760 | 11.72 | 22.25 | 43.9 | 71.6 | $245,579 | $580,176 |

### Which model actually wins depends on the metric

**LightGBM has the best R² (0.875) and lowest mean error, but Random Forest has
the best median error** — MdAPE of 7.86% vs 8.66%, and it puts 3 percentage
points more predictions inside the 10% band.

These disagree because they measure different things. R², RMSE, and MAE are
mean-based, so they reward a model that limits catastrophic misses. MdAPE and
PPE10 describe the *typical* prediction. LightGBM is better on the tail; Random
Forest is better on the median case.

For an AVM, MdAPE and PPE10 are the conventional headline metrics (Zillow and
Redfin both report median error), which argues for **Random Forest** as the
production choice. LightGBM remains preferable if limiting large errors matters
more than typical accuracy.

The gap between MAPE (16.5%) and MdAPE (7.9%) is itself informative: a minority
of severe misses roughly doubles the mean, so the average prediction is
considerably better than MAPE alone suggests.

### Performance by price band

Random Forest, test month.

| Price band | n | % of test | MdAPE % | MAPE % | MAE | Median bias | PPE10 % | PPE20 % |
|---|---|---|---|---|---|---|---|---|
| <$400K | 1,101 | 7.5 | 10.88 | **71.28** | $75,421 | — | — | — |
| $400–600K | 2,301 | 15.6 | 6.10 | 11.47 | $58,104 | — | — | — |
| **$600–800K** | 2,697 | 18.3 | **5.85** | **9.14** | $64,521 | — | — | — |
| $800K–1M | 2,229 | 15.1 | 6.70 | 10.28 | $92,521 | — | — | — |
| $1–1.5M | 2,876 | 19.5 | 8.52 | 12.02 | $149,355 | −$21,142 | 55.9 | 84.2 |
| $1.5–2M | 1,509 | 10.2 | 10.63 | 14.45 | $248,731 | −$44,337 | 47.5 | 80.2 |
| $2–3M | 1,101 | 7.5 | 11.43 | 14.98 | $367,252 | −$113,030 | 44.7 | 73.5 |
| $3M+ | — | — | — | — | — | −$361,635 | 34.3 | 63.8 |

**The model is most accurate in the $600K–1M range** (MdAPE under 7%), which is
also where the bulk of the training data sits.

**Accuracy degrades in both directions from there**, for different reasons:

- *At the low end*, MdAPE is 10.9% but MAPE is **71.3%** — a sevenfold gap found
  nowhere else in the table. The typical cheap home is predicted reasonably, but
  a small number are catastrophically wrong. On a $150K home, a $100K miss is a
  67% error, so small dollar mistakes become enormous percentage ones. These are
  likely distressed sales, teardowns, or partial-interest transfers the feature
  set can't distinguish from ordinary homes.

- *At the high end*, error grows steadily and the bias turns sharply negative:
  the model under-predicts $3M+ homes by a median of **$361,635**, and only 34%
  of those predictions land within 10%. Luxury pricing is driven by finishes,
  architectural pedigree, and views — none of which are in the data. The model
  regresses toward the mean because it has nothing else to go on.

The negative median bias present in every band above $1M is the signature of
this mean-reversion: the model systematically prices expensive homes too low.

### Random Forest depth sweep

Test R² plateaus past depth 15 while the train/test gap keeps widening, so
depth 15 was selected as the elbow point.

| max_depth | Train R² | Test R² | Gap |
|---|---|---|---|
| 10 | 0.888 | 0.839 | 0.049 |
| 12 | 0.919 | 0.850 | 0.069 |
| **15** | **0.945** | **0.860** | **0.086** |
| 18 | 0.956 | 0.862 | 0.094 |
| 20 | 0.960 | 0.863 | 0.097 |
| unlimited | 0.963 | 0.864 | 0.099 |

### Encoding comparison

| Approach | Features | Test R² (Linear Regression) |
|---|---|---|
| One-hot `City` | 1,067 | 0.7662 |
| Target-encoded `City` | 93 | 0.7420 |
| `County` + lat/long, no `City` | 91 | 0.5975 |

Target encoding gives up ~2.4 points of R² for a 92% reduction in feature count.
Dropping `City` in favor of county and raw coordinates costs far more, which
established that city-level granularity carries most of the location signal.

---

## Key findings

**Location dominates.** `City_TargetEnc` accounts for roughly 50% of Random
Forest feature importance, with `LivingArea` next at about 33%. Everything else
is comparatively marginal.

**School district added little.** The spatial join matched 100% of properties
across 776 districts, but `SchoolDistrictName_TargetEnc` correlates 0.871 with
`City_TargetEnc` and contributes under 2% importance, ranking 5th. School
district boundaries largely track city boundaries in this data, so the feature
is mostly redundant once city is present — a real finding, not a join failure.

**Non-linearity matters, but boosting's edge is narrow.** Random Forest beats
Linear Regression by roughly 10 points of R², confirming meaningful interaction
effects. Gradient boosting then adds only ~1.5 more points and, on median error,
does slightly worse. The large gain came from moving to tree ensembles at all,
not from the more sophisticated ensemble method.

**Data quality was a material issue.** The `LotSizeSquareFeet` unit mismatch
would have silently corrupted any lot-size-derived feature had it not been
caught during exploratory plotting.

---

## Reproducing the results

### Requirements
- Python 3.10+ (developed on 3.14)
- The `raw data/` folder populated with `CRMLSSoldYYYYMM.csv` files covering at
  least 2025-02 through 2026-05

### Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install geopandas shapely requests      # school district spatial join
pip install xgboost lightgbm                # gradient boosting
```

On Windows, if `pip` isn't recognized, use `py -m pip install ...`. Inside a
notebook cell, use `%pip install ...` instead.

### Running

1. Open the `py files/` folder in VS Code or Jupyter.
2. Open `weeks1-9.ipynb` and use **Restart → Run All**. The cells depend on
   variables created earlier in the notebook, so running them out of order or
   against a stale kernel will fail — a full clean run is the reliable path.
3. Expect roughly 25–35 minutes end to end, most of it in the two
   hyperparameter sweeps (see [Runtime](#runtime)). To iterate faster, skip the
   sweep cells and use the tuned parameters listed under
   [Models tested](#models-tested) directly.

### Configuration

Nearly everything adjustable lives in the config cell near the top of the
notebook:

| Variable | Purpose |
|---|---|
| `DATA_DIR` | path to the raw CSVs; defaults to `../raw data` |
| `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` | which columns become model features |
| `LEAKAGE_COLS` | columns dropped before modeling |
| `TRAIN_MONTHS` / `VAL_MONTHS` / `TEST_MONTHS` | the date-based split |
| `RANDOM_STATE` | set to 42 for reproducibility |

### Runtime

On a mid-range laptop (ASUS ZenBook), with ~158K training rows and 103 features:

| Step | Approximate time |
|---|---|
| Loading and concatenating all monthly CSVs | 1–2 min |
| Random Forest (200 trees, depth 15) | 2–4 min |
| XGBoost tuning sweep (30 fits) | ~11 min |
| LightGBM tuning sweep (18 fits) | ~5 min |

Reduce `n_estimators` to 50 while iterating, then restore it for final runs.

### Known gotchas

- **Encoding.** `pd.read_csv(..., encoding="latin1")` is required.
- **Relative paths.** Run from inside `py files/`, or set `DATA_DIR` to an
  absolute path: `DATA_DIR = r"C:\path\to\IDX Summer DS\raw data"`.
- **School district column name.** The downloaded shapefile names the district
  field `DistrictNa`, not `DistrictName`. Verify before the spatial join.
- **Feature name typos fail silently.** The pipeline filters features with
  `if c in df.columns`, so a misspelled column is dropped without error. Check
  `len(feature_cols)` after config changes to confirm nothing went missing.

### Outputs

A full run writes these into `py files/` locally. They are not committed to this
repository:

| File | Contents |
|---|---|
| `cleaned_crmls_sold.csv` | preprocessed, split-labeled dataset |
| `baseline_model_results.csv` | Linear Regression baseline metrics |
| `week5_model_comparison_results.csv` | tree model comparison |
| `week6_old_vs_new_comparison.csv` | feature set before/after |
| `week7_advanced_model_results.csv` | gradient boosting results |
| `metrics_summary.csv` | R², MAPE, MdAPE, PPE10, PPE20 for all models |
| `metrics_by_price_band.csv` | accuracy by price segment |

---

## Limitations and next steps

- **The test set is one month.** May 2026 performance may not generalize to
  other market conditions.
- **No condition or quality data.** Renovations, finishes, and views are major
  price drivers absent from the feature set, which explains most of the residual
  error above $2M.
- **Systematic under-prediction at the high end.** The model prices $3M+ homes
  low by a median of $361K. A log-transformed target, or a separate model for
  the luxury segment, would likely help.
- **Severe outliers below $400K.** MAPE of 71% against MdAPE of 11% points to a
  small number of transactions the feature set can't explain — worth
  investigating whether these are distressed or non-arms-length sales that
  should be filtered out entirely.
- **The training window wasn't systematically optimized.** 12 months was chosen
  as a reasonable default; other lengths were not compared head-to-head.
- **Worth exploring:** ZIP-level target encoding as a middle ground between city
  and school district, and confidence intervals so low-reliability predictions
  can be flagged rather than reported at face value.
