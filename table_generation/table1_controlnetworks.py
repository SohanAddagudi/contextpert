"""
Table 1: Control network MSE on sample-held-out split.

Usage:
    python table1_controlnetworks.py population
    python table1_controlnetworks.py cell_specific
    python table1_controlnetworks.py contextualized
    python table1_controlnetworks.py contextualized_full   # + dose, time

Each run saves results to table1_results/<mode>.json.
After all runs, use:
    python table1_controlnetworks.py aggregate
to print the final table.
"""

import sys
import json
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
VALID_MODES = ['population', 'cell_specific', 'contextualized', 'contextualized_full', 'aggregate']

if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
    print(f"Usage: python {sys.argv[0]} <{'|'.join(VALID_MODES)}>")
    sys.exit(1)

MODEL_MODE = sys.argv[1]

RESULTS_DIR = Path(__file__).parent / 'table1_results2'
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Aggregate mode: just read JSONs and print table
# ---------------------------------------------------------------------------
if MODEL_MODE == 'aggregate':
    rows = []
    labels = {
        'population': 'Population',
        'cell_specific': 'Group-specific',
        'contextualized': 'CellVS-Net',
        'contextualized_full': '+ dose, time',
    }
    for mode, label in labels.items():
        path = RESULTS_DIR / f'{mode}.json'
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            rows.append((label, r['test_full'], r['test_nc_high'], r['test_nc_low']))
        else:
            print(f"WARNING: {path} not found, skipping {label}")

    if rows:
        print("\n" + "=" * 60)
        print("TABLE 1: Control Network MSE (sample-held-out)")
        print("-" * 60)
        print(f"{'Model':<20} | {'Full Test':>12} | {'n_c > 3':>12} | {'n_c <= 3':>12}")
        print("-" * 60)
        for label, full, high, low in rows:
            print(f"{label:<20} | {full:>12.4f} | {high:>12.4f} | {low:>12.4f}")
        print("=" * 60)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.33
N_DATA_PCS = 50
BATCH_SIZE = 32

from lightning import seed_everything
seed_everything(RANDOM_STATE, workers=True)

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
PERT_TYPE_DIR = DATA_DIR / 'pert_type_csvs'

controls_to_fit = ['ctl_vehicle', 'ctl_vector', 'ctl_untrt']

USE_FULL_CONTEXT = (MODEL_MODE == 'contextualized_full')
IS_CONTEXTUALIZED = MODEL_MODE in ('contextualized', 'contextualized_full')

print(f"MODEL_MODE: {MODEL_MODE}")
print(f"USE_FULL_CONTEXT: {USE_FULL_CONTEXT}")
print(f"Targeting Controls: {controls_to_fit}")
print("---------------------------------------------------\n")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading control data...")
ctrl_dfs = []
for ctrl in controls_to_fit:
    path = PERT_TYPE_DIR / f'{ctrl}.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Control data file not found at {path}")
    ctrl_dfs.append(pd.read_csv(path, engine='pyarrow'))
df = pd.concat(ctrl_dfs, ignore_index=True)
del ctrl_dfs; gc.collect()

# Quality filter
if 'distil_cc_q75' in df.columns and 'pct_self_rank_q25' in df.columns:
    condition = (
        (df['distil_cc_q75'] < 0.2) |
        (df['pct_self_rank_q25'] > 5)
    )
    df = df[~condition].reset_index(drop=True)

# n_c per cell line
cell_counts = df['cell_id'].value_counts()
df['n_c'] = df['cell_id'].map(cell_counts)

# Extract dose/time metadata before dropping them
if USE_FULL_CONTEXT:
    dose_raw = df['pert_dose'].copy()
    time_raw = df['pert_time'].copy()
    dose_unit_raw = df['pert_dose_unit'].copy() if 'pert_dose_unit' in df.columns else None

# Feature columns (numeric, excluding metadata)
columns_to_drop = ['pert_dose', 'pert_dose_unit', 'pert_time',
                   'distil_cc_q75', 'pct_self_rank_q25', 'n_c']
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                if col not in columns_to_drop and col in df.columns]
X = df[feature_cols].values.astype(np.float32)

cell_ids = df['cell_id'].values
nc_values = df['n_c'].values
unique_cells = np.unique(cell_ids)

print(f"Data shape: {X.shape}, unique cells: {len(unique_cells)}")

# ---------------------------------------------------------------------------
# 2. Build context
# ---------------------------------------------------------------------------
cell_to_int = {cell: i for i, cell in enumerate(unique_cells)}

# One-hot cell context (used by contextualized and cell_specific label mapping)
cell_dummies = pd.get_dummies(unique_cells, prefix='cell', dtype=int)
cell2vec = dict(zip(unique_cells, cell_dummies.values))
cell_context_matrix = np.array([cell2vec[cid] for cid in cell_ids], dtype=np.float32)

# Pert dummies
pert_dummies = pd.get_dummies(df['pert_id'], drop_first=True).values.astype(np.float32)

if USE_FULL_CONTEXT:
    # Handle missing dose/time (sentinel -666 -> NaN -> mean impute)
    # Coerce to numeric — pyarrow may give mixed dtypes when '-666' was stored as str.
    dose = pd.to_numeric(dose_raw, errors='coerce').replace(-666, np.nan)
    time_ = pd.to_numeric(time_raw, errors='coerce').replace(-666, np.nan)
    ig_dose = dose.isna().astype(np.float32).values.reshape(-1, 1)
    ig_time = time_.isna().astype(np.float32).values.reshape(-1, 1)
    dose.fillna(dose.mean(), inplace=True)
    time_.fillna(time_.mean(), inplace=True)

    # Dose unit dummies
    if dose_unit_raw is not None:
        du_dum = pd.get_dummies(dose_unit_raw, drop_first=True).values.astype(np.float32)
    else:
        du_dum = np.zeros((len(df), 0), dtype=np.float32)

    # Continuous: cell one-hot + dose + time (scaled)
    continuous = np.hstack([
        cell_context_matrix,
        dose.values.reshape(-1, 1),
        time_.values.reshape(-1, 1),
    ])
    sc_cont = StandardScaler()
    continuous_scaled = sc_cont.fit_transform(continuous).astype(np.float32)

    # Categorical: pert dummies + dose unit dummies + ignore flags
    categorical = np.hstack([pert_dummies, du_dum, ig_dose, ig_time])

    C_global = np.hstack([continuous_scaled, categorical]).astype(np.float32)

    del dose, time_, ig_dose, ig_time, du_dum, continuous, continuous_scaled, categorical
    del dose_raw, time_raw, dose_unit_raw
    gc.collect()
else:
    C_global = np.hstack([pert_dummies, cell_context_matrix]).astype(np.float32)

del pert_dummies, cell_context_matrix
gc.collect()

# Free the dataframe
del df; gc.collect()

print(f"Context dim: {C_global.shape[1]}")

# ---------------------------------------------------------------------------
# 3. Train/test split (per cell line, sample-held-out)
# ---------------------------------------------------------------------------
X_train_list, X_test_list = [], []
C_train_list, C_test_list = [], []
nc_test_list = []
labels_train_list, labels_test_list = [], []

print("Splitting data...")
for cell in tqdm(unique_cells, desc="Splitting by cell line"):
    cell_mask = (cell_ids == cell)
    X_cell = X[cell_mask]
    n_samples = len(X_cell)
    current_nc = nc_values[cell_mask][0]
    current_cell_int = cell_to_int[cell]

    if n_samples < 2:
        continue

    indices = np.arange(n_samples)
    if n_samples == 2:
        train_idx, test_idx = [0], [1]
    elif n_samples == 3:
        train_idx, test_idx = [0, 1], [2]
    else:
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train_list.append(X_cell[train_idx])
    X_test_list.append(X_cell[test_idx])
    nc_test_list.append(np.full(len(test_idx), current_nc))
    labels_train_list.append(np.full(len(train_idx), current_cell_int))
    labels_test_list.append(np.full(len(test_idx), current_cell_int))

    C_cell = C_global[cell_mask]
    C_train_list.append(C_cell[train_idx])
    C_test_list.append(C_cell[test_idx])

# Free raw arrays
del X, C_global, cell_ids, nc_values; gc.collect()

X_train = np.vstack(X_train_list).astype(np.float32)
X_test = np.vstack(X_test_list).astype(np.float32)
nc_test = np.concatenate(nc_test_list)
labels_train = np.concatenate(labels_train_list)
labels_test = np.concatenate(labels_test_list)
C_train = np.vstack(C_train_list).astype(np.float32)
C_test = np.vstack(C_test_list).astype(np.float32)

del X_train_list, X_test_list, C_train_list, C_test_list
del nc_test_list, labels_train_list, labels_test_list
gc.collect()

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 4. Normalize + PCA
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
del X_train, X_test; gc.collect()

pca = PCA(n_components=min(N_DATA_PCS, X_train_scaled.shape[1]))
X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
X_test_pca = pca.transform(X_test_scaled).astype(np.float32)
del X_train_scaled, X_test_scaled; gc.collect()

X_mean = X_train_pca.mean(axis=0)
X_std = X_train_pca.std(axis=0)
X_std = np.where(X_std == 0, 1e-6, X_std)
X_train_norm = ((X_train_pca - X_mean) / X_std).astype(np.float32)
X_test_norm = ((X_test_pca - X_mean) / X_std).astype(np.float32)
del X_train_pca, X_test_pca; gc.collect()

# ---------------------------------------------------------------------------
# 5. Train model
# ---------------------------------------------------------------------------

model_obj = None
trainer = None

if MODEL_MODE == 'population':
    from contextualized.baselines.networks import CorrelationNetwork
    print("\n--- Training: Population (CorrelationNetwork) ---")
    model_obj = CorrelationNetwork()
    model_obj.fit(X_train_norm)
    print("Done.")

elif MODEL_MODE == 'cell_specific':
    from contextualized.baselines.networks import GroupedNetworks, CorrelationNetwork
    from joblib import Parallel, delayed
    print("\n--- Training: Cell Specific (GroupedNetworks) ---")
    model_obj = GroupedNetworks(CorrelationNetwork)
    unique_train_labels = np.unique(labels_train)

    def fit_model_for_group(label):
        mask = labels_train == label
        X_group = X_train_norm[mask]
        model = CorrelationNetwork()
        model.fit(X_group)
        return (label, model)

    print(f"Fitting {len(unique_train_labels)} cell-specific models...")
    jobs = [delayed(fit_model_for_group)(label) for label in unique_train_labels]
    results = Parallel(n_jobs=1)(tqdm(jobs, desc="Training cell models"))
    model_obj.models = {label: model for label, model in results}
    print("Done.")

elif IS_CONTEXTUALIZED:
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from contextualized.regression.lightning_modules import ContextualizedCorrelation
    from contextualized.data import CorrelationDataModule

    label = "CellVS-Net + dose/time" if USE_FULL_CONTEXT else "CellVS-Net"
    print(f"\n--- Training: {label} (Contextualized, Lightning) ---")

    contextualized_model = ContextualizedCorrelation(
        context_dim=C_train.shape[1],
        x_dim=X_train_norm.shape[1],
        encoder_type='mlp',
        num_archetypes=50,
    )

    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train_norm)), test_size=0.1, random_state=RANDOM_STATE)

    datamodule = CorrelationDataModule(
        C_train=C_train[tr_idx], X_train=X_train_norm[tr_idx],
        C_val=C_train[val_idx],  X_val=X_train_norm[val_idx],
        C_test=C_test,           X_test=X_test_norm,
        C_predict=C_test,        X_predict=X_test_norm,
        batch_size=BATCH_SIZE,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1, filename='best_model'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss', patience=1, mode='min'
    )

    trainer = Trainer(
        max_epochs=15,
        accelerator='auto', devices='auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
    )

    trainer.fit(contextualized_model, datamodule=datamodule)
    model_obj = contextualized_model

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print(f"EVALUATING ({MODEL_MODE})")
print("=" * 50)


def get_mse_score(X_subset, C_subset, labels_subset):
    if len(X_subset) == 0:
        return float('nan')

    if MODEL_MODE == 'population':
        return float(np.mean(model_obj.measure_mses(X_subset)))

    elif MODEL_MODE == 'cell_specific':
        return float(np.mean(model_obj.measure_mses(X_subset, labels_subset)))

    elif IS_CONTEXTUALIZED:
        datamodule.X_test = X_subset
        datamodule.C_test = C_subset
        results = trainer.test(model_obj, datamodule=datamodule, verbose=False)
        return float(results[0]['test_loss'])


# Full test
score_test_full = get_mse_score(X_test_norm, C_test, labels_test)
print(f"  Full Test:   {score_test_full:.4f}")

# nc > 3
mask_high = nc_test > 3
score_test_high = get_mse_score(
    X_test_norm[mask_high], C_test[mask_high], labels_test[mask_high])
print(f"  n_c > 3:     {score_test_high:.4f}")

# nc <= 3
mask_low = nc_test <= 3
score_test_low = get_mse_score(
    X_test_norm[mask_low], C_test[mask_low], labels_test[mask_low])
print(f"  n_c <= 3:    {score_test_low:.4f}")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
result = {
    'model_mode': MODEL_MODE,
    'test_full': score_test_full,
    'test_nc_high': score_test_high,
    'test_nc_low': score_test_low,
}

out_path = RESULTS_DIR / f'{MODEL_MODE}.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {out_path}")
print(f"  Full Test: {score_test_full:.4f}")
print(f"  n_c > 3:   {score_test_high:.4f}")
print(f"  n_c <= 3:  {score_test_low:.4f}")
