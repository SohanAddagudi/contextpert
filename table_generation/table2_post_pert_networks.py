import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
import gc
import torch
import lightning as pl
from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from lightning import Trainer
from contextualized.callbacks import PredictionWriter
from pathlib import Path
import glob
from contextualized.baselines.networks import GroupedNetworks, CorrelationNetwork
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
PATH_TRT_CP = DATA_DIR / 'trt_cp.csv'
PATH_CTLS = DATA_DIR / 'ctrls.csv'

N_DATA_PCS = 50
N_CTRL_PCS = 20
N_EMBEDDING_PCS = 20
N_PERT_PCS = 50
RANDOM_STATE = 42
TEST_SIZE = 0.33

RESULTS_DIR = Path(__file__).parent / 'results'


def load_and_filter_data():
    df = pd.read_csv(PATH_TRT_CP)
    df = df[df['pert_type'].isin(['trt_cp'])].reset_index(drop=True)
    condition = (
        (df['distil_cc_q75'] < 0.2) |
        (df['distil_cc_q75'] == -666) |
        (df['distil_cc_q75'].isna()) |
        (df['pct_self_rank_q25'] > 5) |
        (df['pct_self_rank_q25'] == -666) |
        (df['pct_self_rank_q25'].isna())
    )
    df = df[~condition].reset_index(drop=True)
    return df


def extract_features(df):
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    columns_to_drop = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
    feature_cols = [col for col in feature_cols if col not in columns_to_drop]
    return df[feature_cols].values


def build_pert_context(df):
    """Build PCA-reduced perturbation context using sparse dummies + TruncatedSVD."""
    le = LabelEncoder()
    pert_encoded = le.fit_transform(df['pert_id'].values)
    n_perts = len(le.classes_)
    n_samples = len(df)
    # Sparse one-hot (drop first by slicing off column 0)
    pert_sparse = sparse.csr_matrix(
        (np.ones(n_samples), (np.arange(n_samples), pert_encoded)),
        shape=(n_samples, n_perts)
    )[:, 1:]  # drop_first
    n_components = min(N_PERT_PCS, pert_sparse.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    pert_pcs = svd.fit_transform(pert_sparse)
    print(f"Pert context: {pert_sparse.shape[1]} dummies -> {n_components} PCs "
          f"(explained variance: {svd.explained_variance_ratio_.sum():.3f})")
    return pert_pcs


def build_full_context_features(df):
    """Build dose/time context features."""
    pert_unit_dummies = pd.get_dummies(df['pert_dose_unit'], drop_first=True).values

    ignore_time = (df['pert_time'].values == -666).astype(np.float32).reshape(-1, 1)
    ignore_dose = (df['pert_dose'].values == -666).astype(np.float32).reshape(-1, 1)

    pert_time = df['pert_time'].values.copy().astype(np.float64)
    pert_dose = df['pert_dose'].values.copy().astype(np.float64)
    for arr in [pert_time, pert_dose]:
        mean_val = arr[arr != -666].mean()
        arr[arr == -666] = mean_val

    return {
        'pert_unit_dummies': pert_unit_dummies,
        'pert_time': pert_time.reshape(-1, 1),
        'pert_dose': pert_dose.reshape(-1, 1),
        'ignore_time': ignore_time,
        'ignore_dose': ignore_dose,
    }


def build_context_matrix(df, cell_context_mode, use_full_context):
    """Build the global context matrix for contextualized modes."""
    cell_ids = df['cell_id'].values
    unique_cells = np.unique(cell_ids)

    # Pert context (PCA-reduced)
    pert_pcs = build_pert_context(df)

    # Cell context
    if cell_context_mode == 'expression':
        print("Loading 'expression' context (PCA of control expression)...")
        ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
        ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells)]
        scaler_ctrls = StandardScaler()
        ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
        n_components = min(N_CTRL_PCS, ctrls_scaled.shape[0])
        pca_ctrls = PCA(n_components=n_components, random_state=RANDOM_STATE)
        ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
        cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))

        missing_cells = set(unique_cells) - set(cell2vec.keys())
        if missing_cells:
            print(f"Warning: Missing context for {len(missing_cells)} cells. Filtering them out.")
            mask = ~df['cell_id'].isin(missing_cells)
            return None, mask  # caller should filter and retry

        cell_context_matrix = np.array([cell2vec[cid] for cid in cell_ids])

    elif cell_context_mode == 'onehot':
        print("Using 'onehot' context (PCA-reduced)...")
        le_cell = LabelEncoder()
        cell_encoded = le_cell.fit_transform(cell_ids)
        n_cells = len(le_cell.classes_)
        # Small enough for dense (71 cells)
        cell_onehot = np.zeros((len(cell_ids), n_cells), dtype=np.float32)
        cell_onehot[np.arange(len(cell_ids)), cell_encoded] = 1.0
        cell_onehot = cell_onehot[:, 1:]  # drop_first
        cell_context_matrix = cell_onehot
        print(f"Cell onehot context: {cell_context_matrix.shape[1]} dims")

    # Assemble context matrix
    continuous_parts = [pert_pcs, cell_context_matrix] if cell_context_mode != 'onehot' else [pert_pcs]
    categorical_parts = [cell_context_matrix] if cell_context_mode == 'onehot' else []

    if use_full_context:
        fc = build_full_context_features(df)
        categorical_parts.extend([fc['pert_unit_dummies'], fc['ignore_time'], fc['ignore_dose']])
        continuous_parts.extend([fc['pert_time'], fc['pert_dose']])

    # Scale continuous, leave categorical as-is
    if continuous_parts:
        C_continuous = np.hstack(continuous_parts)
        scaler = StandardScaler()
        C_continuous_scaled = scaler.fit_transform(C_continuous)
        if categorical_parts:
            C_categorical = np.hstack(categorical_parts)
            C_global = np.hstack([C_continuous_scaled, C_categorical]).astype(np.float32)
        else:
            C_global = C_continuous_scaled.astype(np.float32)
    else:
        C_global = np.hstack(categorical_parts).astype(np.float32)

    print(f"Context matrix shape: {C_global.shape}")
    return C_global, None


def split_by_context_pairs(X, cell_ids, pert_ids, C_global=None):
    """Split data within each (cell, pert) pair for stratified train/test."""
    pairs = list(zip(cell_ids, pert_ids))
    unique_pairs = sorted(set(pairs))

    X_train_list, X_test_list = [], []
    C_train_list, C_test_list = [], []
    cell_train_list, cell_test_list = [], []
    pert_train_list, pert_test_list = [], []

    for cell_id, pert_id in tqdm(unique_pairs, desc="Splitting by (cell, pert) pair"):
        mask = (cell_ids == cell_id) & (pert_ids == pert_id)
        if mask.sum() < 2:
            continue

        X_pair = X[mask]
        cells_pair = cell_ids[mask]
        perts_pair = pert_ids[mask]

        if C_global is not None:
            C_pair = C_global[mask]
            X_tr, X_te, C_tr, C_te, c_tr, c_te, p_tr, p_te = train_test_split(
                X_pair, C_pair, cells_pair, perts_pair,
                test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            C_train_list.append(C_tr)
            C_test_list.append(C_te)
        else:
            X_tr, X_te, c_tr, c_te, p_tr, p_te = train_test_split(
                X_pair, cells_pair, perts_pair,
                test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        X_train_list.append(X_tr)
        X_test_list.append(X_te)
        cell_train_list.append(c_tr)
        cell_test_list.append(c_te)
        pert_train_list.append(p_tr)
        pert_test_list.append(p_te)

    result = {
        'X_train': np.vstack(X_train_list),
        'X_test': np.vstack(X_test_list),
        'cell_ids_train': np.concatenate(cell_train_list),
        'cell_ids_test': np.concatenate(cell_test_list),
        'pert_ids_train': np.concatenate(pert_train_list),
        'pert_ids_test': np.concatenate(pert_test_list),
    }
    if C_global is not None:
        result['C_train'] = np.vstack(C_train_list).astype(np.float32)
        result['C_test'] = np.vstack(C_test_list).astype(np.float32)
    return result


def normalize_features(X_train, X_test):
    """Scale, PCA, and normalize features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=N_DATA_PCS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_mean = X_train_pca.mean(axis=0)
    X_std = X_train_pca.std(axis=0)
    X_std = np.where(X_std == 0, 1e-6, X_std)
    X_train_norm = (X_train_pca - X_mean) / X_std
    X_test_norm = (X_test_pca - X_mean) / X_std
    return X_train_norm, X_test_norm


def compute_per_cell_mses(cell_ids_train, cell_ids_test, mse_train, mse_test):
    """Compute per-cell MSE aggregates."""
    all_cells = np.union1d(cell_ids_train, cell_ids_test)
    per_cell = {}
    for cell_id in sorted(all_cells):
        tr_mask = cell_ids_train == cell_id
        te_mask = cell_ids_test == cell_id
        per_cell[cell_id] = {
            'train_mse': float(np.mean(mse_train[tr_mask])) if tr_mask.any() else None,
            'test_mse': float(np.mean(mse_test[te_mask])) if te_mask.any() else None,
            'n_train': int(tr_mask.sum()),
            'n_test': int(te_mask.sum()),
        }
    return per_cell


def measure_mses_vectorized(betas, mus, X):
    """Vectorized MSE computation for contextualized models."""
    n_features = X.shape[1]
    # X[:, k] broadcasted: (n, 1, p) * betas(n, p, p) -> predicted X[:, j] from X[:, k]
    # residual[i,j,k] = X[i,j] - betas[i,j,k]*X[i,k] - mus[i,j,k]
    X_expanded_j = X[:, :, np.newaxis]  # (n, p, 1)
    X_expanded_k = X[:, np.newaxis, :]  # (n, 1, p)
    residuals = X_expanded_j - betas * X_expanded_k - mus  # (n, p, p)
    mses = (residuals ** 2).mean(axis=(1, 2))
    return mses


def save_results(mode_name, overall_train_mse, overall_test_mse, per_cell):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'mode': mode_name,
        'overall_train_mse': float(overall_train_mse),
        'overall_test_mse': float(overall_test_mse),
        'per_cell': per_cell,
    }
    path = RESULTS_DIR / f'table2_{mode_name}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return results


def print_results(mode_name, overall_train_mse, overall_test_mse, per_cell):
    print(f"\n{'=' * 50}")
    print(f"Performance: {mode_name}")
    print(f"Overall MSE on train set: {overall_train_mse:.4f}")
    print(f"Overall MSE on test set: {overall_test_mse:.4f}")
    print('=' * 50)
    print("\nPer-Cell Performance:")
    for cell_id, metrics in per_cell.items():
        tr = f"{metrics['train_mse']:8.4f}" if metrics['train_mse'] is not None else "     N/A"
        te = f"{metrics['test_mse']:8.4f}" if metrics['test_mse'] is not None else "     N/A"
        print(f"  {cell_id:<15} train={tr}  test={te}  n_train={metrics['n_train']:6d}  n_test={metrics['n_test']:6d}")


# ---- Mode runners ----

def run_population(split, X_train_norm, X_test_norm):
    print("\n--- Running Model: Population (CorrelationNetwork) ---")
    pop_cn = CorrelationNetwork()
    pop_cn.fit(X_train_norm)
    mse_train = pop_cn.measure_mses(X_train_norm)
    mse_test = pop_cn.measure_mses(X_test_norm)

    per_cell = compute_per_cell_mses(
        split['cell_ids_train'], split['cell_ids_test'], mse_train, mse_test
    )
    overall_train = float(np.mean(mse_train))
    overall_test = float(np.mean(mse_test))
    print_results('population', overall_train, overall_test, per_cell)
    return save_results('population', overall_train, overall_test, per_cell)


def _fast_corr_network_mse(X_train_group, X_test_group=None):
    """Vectorized CorrelationNetwork: fit on X_train_group, return MSEs.

    Each univariate regression X_i = beta_ij * X_j + intercept_ij is computed
    analytically via covariance, replacing 2500 sklearn calls with one matrix op.
    """
    n, p = X_train_group.shape
    means = X_train_group.mean(axis=0)  # (p,)
    X_c = X_train_group - means  # (n, p)
    cov = (X_c.T @ X_c) / n  # (p, p): cov[a, b] = cov(X_a, X_b)
    var = np.diag(cov)  # (p,)
    # Avoid division by zero for constant features
    var_safe = np.where(var == 0, 1e-12, var)
    # betas[i, j] = cov(X_j, X_i) / var(X_j) = cov[j, i] / cov[j, j]
    betas = cov.T / var_safe[np.newaxis, :]  # (p, p)
    intercepts = means[:, np.newaxis] - betas * means[np.newaxis, :]  # (p, p)

    def _compute_mse(X):
        # residual[n, i, j] = X[n, i] - betas[i, j] * X[n, j] - intercepts[i, j]
        # Compute as: X_i - (betas @ X_j^T)^T - intercepts, but per-sample
        # predicted[i, j] for sample n = betas[i,j]*X[n,j] + intercepts[i,j]
        # -> predicted_all = X @ betas.T + intercepts (broadcasted wrong dims)
        # Actually: predicted[n, i, j] = betas[i,j]*X[n,j] + intercepts[i,j]
        # For fixed i: predicted[n, i, :] = betas[i, :] * X[n, :] + intercepts[i, :]
        #   but we want residual = X[n, i] - predicted[n, i, j]
        # Sum over (i,j): sum_i sum_j (X[n,i] - betas[i,j]*X[n,j] - intercepts[i,j])^2
        # = sum_j [ sum_i (X[n,i] - betas[i,j]*X[n,j] - intercepts[i,j])^2 ]
        # For fixed j: residuals[:, :, j] = X - X[:, j:j+1] * betas[:, j].T - intercepts[:, j].T
        # Process one j at a time to keep memory O(n*p) instead of O(n*p*p)
        mses = np.zeros(len(X))
        for j in range(p):
            # predicted_i = betas[i, j] * X[n, j] + intercepts[i, j] for all i
            pred = np.outer(X[:, j], betas[:, j]) + intercepts[:, j]  # (n, p)
            residuals = X - pred  # (n, p) — residual for target i, predictor j
            mses += (residuals ** 2).sum(axis=1) / (p * p)
        return mses

    mse_train = _compute_mse(X_train_group)
    mse_test = _compute_mse(X_test_group) if X_test_group is not None else None
    return mse_train, mse_test


def run_cell_specific(split, X_train_norm, X_test_norm):
    print("\n--- Running Model: Cell-Specific (GroupedNetworks) ---")
    train_pairs = [f"{c}_{p}" for c, p in zip(split['cell_ids_train'], split['pert_ids_train'])]
    test_pairs = [f"{c}_{p}" for c, p in zip(split['cell_ids_test'], split['pert_ids_test'])]

    all_pairs = np.unique(np.concatenate((train_pairs, test_pairs)))
    pair_to_int = {p: i for i, p in enumerate(all_pairs)}
    train_labels = np.array([pair_to_int[p] for p in train_pairs])
    test_labels = np.array([pair_to_int[p] for p in test_pairs])

    unique_train_labels = np.unique(train_labels)
    print(f"Fitting {len(unique_train_labels)} context-specific models (vectorized)...")

    mse_train = np.zeros(len(X_train_norm))
    mse_test = np.zeros(len(X_test_norm))

    for label in tqdm(unique_train_labels, desc="Fitting groups"):
        train_mask = train_labels == label
        test_mask = test_labels == label
        X_te_group = X_test_norm[test_mask] if test_mask.any() else None

        mse_tr, mse_te = _fast_corr_network_mse(X_train_norm[train_mask], X_te_group)
        mse_train[train_mask] = mse_tr
        if mse_te is not None:
            mse_test[test_mask] = mse_te

    per_cell = compute_per_cell_mses(
        split['cell_ids_train'], split['cell_ids_test'], mse_train, mse_test
    )
    overall_train = float(np.mean(mse_train))
    overall_test = float(np.mean(mse_test))
    print_results('cell_specific', overall_train, overall_test, per_cell)
    return save_results('cell_specific', overall_train, overall_test, per_cell)


def run_contextualized(split, X_train_norm, X_test_norm, cell_context_mode, use_full_context):
    mode_suffix = cell_context_mode
    if use_full_context:
        mode_suffix += '_full_context'
    mode_name = f'contextualized_{mode_suffix}'

    print(f"\n--- Running Model: Contextualized ({mode_name}) ---")
    C_train = split['C_train']
    C_test = split['C_test']
    print(f"Context shapes: Train={C_train.shape}, Test={C_test.shape}")

    contextualized_model = ContextualizedCorrelation(
        context_dim=C_train.shape[1],
        x_dim=X_train_norm.shape[1],
        encoder_type='mlp',
        num_archetypes=25,
    )

    train_indices, val_indices = train_test_split(
        np.arange(len(X_train_norm)), test_size=0.2, random_state=RANDOM_STATE
    )

    datamodule = CorrelationDataModule(
        C_train=C_train[train_indices],
        X_train=X_train_norm[train_indices],
        C_val=C_train[val_indices],
        X_val=X_train_norm[val_indices],
        C_test=C_test,
        X_test=X_test_norm,
        C_predict=np.concatenate((C_train, C_test), axis=0),
        X_predict=np.concatenate((X_train_norm, X_test_norm), axis=0),
        batch_size=256,
    )

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1, filename='best_model',
    )
    trainer = Trainer(
        max_epochs=5, accelerator='auto', devices='auto',
        callbacks=[checkpoint_callback],
    )
    trainer.fit(contextualized_model, datamodule=datamodule)

    print("Testing model...")
    trainer.test(contextualized_model, datamodule.train_dataloader())
    trainer.test(contextualized_model, datamodule.test_dataloader())

    output_dir = Path(checkpoint_callback.best_model_path).parent / 'predictions'
    writer_callback = PredictionWriter(output_dir=output_dir, write_interval='batch')
    trainer_pred = Trainer(accelerator='auto', devices='auto', callbacks=[checkpoint_callback, writer_callback])

    print("Making predictions...")
    trainer_pred.predict(contextualized_model, datamodule=datamodule)

    print("Compiling predictions...")
    all_correlations, all_betas, all_mus = {}, {}, {}
    pred_files = glob.glob(str(output_dir / 'predictions_*.pt'))
    for file in pred_files:
        preds = torch.load(file)
        for context, correlation, beta, mu in zip(
            preds['contexts'], preds['correlations'], preds['betas'], preds['mus']
        ):
            key = tuple(context.to(torch.float32).tolist())
            all_correlations[key] = correlation.cpu().numpy()
            all_betas[key] = beta.cpu().numpy()
            all_mus[key] = mu.cpu().numpy()

    C_train_32 = C_train.astype(np.float32)
    C_test_32 = C_test.astype(np.float32)
    betas_train = np.array([all_betas[tuple(row)] for row in C_train_32])
    betas_test = np.array([all_betas[tuple(row)] for row in C_test_32])
    mus_train = np.array([all_mus[tuple(row)] for row in C_train_32])
    mus_test = np.array([all_mus[tuple(row)] for row in C_test_32])

    mse_train = measure_mses_vectorized(betas_train, mus_train, X_train_norm)
    mse_test = measure_mses_vectorized(betas_test, mus_test, X_test_norm)

    per_cell = compute_per_cell_mses(
        split['cell_ids_train'], split['cell_ids_test'], mse_train, mse_test
    )
    overall_train = float(np.mean(mse_train))
    overall_test = float(np.mean(mse_test))
    print_results(mode_name, overall_train, overall_test, per_cell)
    return save_results(mode_name, overall_train, overall_test, per_cell)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['population', 'cell_specific',
                                 'contextualized_onehot', 'contextualized_expression',
                                 'contextualized_expression_full'])
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override results directory')
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    print(f"--- Loading data ---")
    df = load_and_filter_data()
    X = extract_features(df)
    cell_ids = df['cell_id'].values
    pert_ids = df['pert_id'].values
    print(f"Data: {len(df)} samples, {len(np.unique(cell_ids))} cells, {len(np.unique(pert_ids))} perts")

    is_contextualized = args.mode.startswith('contextualized')

    if is_contextualized:
        if args.mode == 'contextualized_onehot':
            cell_context_mode, use_full_context = 'onehot', False
        elif args.mode == 'contextualized_expression':
            cell_context_mode, use_full_context = 'expression', False
        elif args.mode == 'contextualized_expression_full':
            cell_context_mode, use_full_context = 'expression', True

        C_global, filter_mask = build_context_matrix(df, cell_context_mode, use_full_context)
        if filter_mask is not None:
            df = df[filter_mask].reset_index(drop=True)
            X = X[filter_mask]
            cell_ids = df['cell_id'].values
            pert_ids = df['pert_id'].values
            C_global, _ = build_context_matrix(df, cell_context_mode, use_full_context)

        split = split_by_context_pairs(X, cell_ids, pert_ids, C_global)
    else:
        split = split_by_context_pairs(X, cell_ids, pert_ids)

    X_train_norm, X_test_norm = normalize_features(split['X_train'], split['X_test'])

    if args.mode == 'population':
        run_population(split, X_train_norm, X_test_norm)
    elif args.mode == 'cell_specific':
        run_cell_specific(split, X_train_norm, X_test_norm)
    else:
        if args.mode == 'contextualized_onehot':
            run_contextualized(split, X_train_norm, X_test_norm, 'onehot', False)
        elif args.mode == 'contextualized_expression':
            run_contextualized(split, X_train_norm, X_test_norm, 'expression', False)
        elif args.mode == 'contextualized_expression_full':
            run_contextualized(split, X_train_norm, X_test_norm, 'expression', True)


if __name__ == '__main__':
    main()
