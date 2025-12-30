import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
import torch
import lightning as pl
from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from lightning import seed_everything, Trainer
from contextualized.callbacks import PredictionWriter
from pathlib import Path
import glob
from contextualized.baselines.networks import GroupedNetworks, CorrelationNetwork
from tqdm import tqdm
from joblib import Parallel, delayed

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])

MODEL_MODE = 'cell_specific' # Options: 'contextualized', 'cell_specific', 'population'
# CELL_CONTEXT_MODE = 'expression' # Options: 'expression', 'onehot'  (only for 'contextualized' mode)
# USE_FULL_CONTEXT_FEATURES = True # Options: True, False for using dose, time as context features (only for 'contextualized' mode)

PATH_TRT_CP = DATA_DIR / 'pert_type_csvs' / 'trt_cp.csv'
PATH_CTLS = DATA_DIR / 'ctrls.csv'

N_DATA_PCS = 50
N_CTRL_PCS = 20
N_EMBEDDING_PCS = 20
RANDOM_STATE = 42
TEST_SIZE = 0.33

print(f"--- Running with settings ---")
print(f"MODEL_MODE: {MODEL_MODE}")
if MODEL_MODE == 'contextualized':
    print(f"CELL_CONTEXT_MODE: {CELL_CONTEXT_MODE}")
    print(f"USE_FULL_CONTEXT_FEATURES: {USE_FULL_CONTEXT_FEATURES}")
elif MODEL_MODE == 'cell_specific':
    print("INFO: 'cell_specific' mode will train one network per (cell, perturbation) pair.")
print("-----------------------------\n")

if not os.path.exists(PATH_TRT_CP):
    raise FileNotFoundError(f"Main data not found at {PATH_TRT_CP}")

if MODEL_MODE == 'contextualized':
    if CELL_CONTEXT_MODE == 'expression' and not os.path.exists(PATH_CTLS):
        raise FileNotFoundError(f"Controls data not found at {PATH_CTLS} (required for 'expression' mode)")
    if CELL_CONTEXT_MODE == 'embedding':
        if not os.path.exists(PATH_CTLS) or not os.path.exists(EMB_FILE):
            raise FileNotFoundError(f"ctrls.csv or embeddings.npy not found (required for 'embedding' mode)")

df = pd.read_csv(PATH_TRT_CP)

pert_to_fit_on = ['trt_cp']
mask = df['pert_type'].isin(pert_to_fit_on)
df = df[mask].reset_index(drop=True) 

condition = (
    (df['distil_cc_q75'] < 0.2) |
    (df['distil_cc_q75'] == -666) |
    (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) |
    (df['pct_self_rank_q25'] == -666) |
    (df['pct_self_rank_q25'].isna())
)
df = df[~condition].reset_index(drop=True)

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
columns_to_drop = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
feature_cols = [col for col in feature_cols if col not in columns_to_drop]
X = df[feature_cols].values

pert_dummies = pd.get_dummies(df['pert_id'], drop_first=True)

if MODEL_MODE == 'contextualized' and USE_FULL_CONTEXT_FEATURES:
    pert_unit_dummies = pd.get_dummies(df['pert_dose_unit'], drop_first=True)
    
    df['ignore_flag_pert_time'] = np.where(df['pert_time'] == -666, 1, 0)
    df['ignore_flag_pert_dose'] = np.where(df['pert_dose'] == -666, 1, 0)

    for col in ['pert_time', 'pert_dose']:
        mean_value = df[df[col] != -666][col].mean()
        df[col] = df[col].replace(-666, mean_value)

    pert_time = df['pert_time'].values.reshape(-1, 1)
    pert_dose = df['pert_dose'].values.reshape(-1, 1)
    ignore_time = df['ignore_flag_pert_time'].values.reshape(-1, 1)
    ignore_dose = df['ignore_flag_pert_dose'].values.reshape(-1, 1)

cell_ids = df['cell_id'].values
pert_ids = df['pert_id'].values
unique_cells = np.unique(cell_ids)

cell2vec = {} 
if MODEL_MODE == 'contextualized':
    if CELL_CONTEXT_MODE == 'expression':
        print("Loading 'expression' context (PCA of control expression)...")
        ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
        ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells)]
        
        scaler_ctrls = StandardScaler()
        ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

        n_cells = ctrls_scaled.shape[0]
        n_components_for_context = min(N_CTRL_PCS, n_cells)

        pca_ctrls = PCA(n_components=n_components_for_context, random_state=RANDOM_STATE)
        ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
        cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
        
    elif CELL_CONTEXT_MODE == 'embedding':
        print("Loading 'embedding' context (PCA of AIDO embeddings)...")
        all_embeddings_raw = np.load(EMB_FILE)
        ctrls_meta_df = pd.read_csv(PATH_CTLS, index_col=0)
        embedding_cell_ids_full = ctrls_meta_df.index.to_numpy()
        
        scaler_embeddings = StandardScaler()
        embeddings_scaled = scaler_embeddings.fit_transform(all_embeddings_raw)
        n_components_for_context = min(N_EMBEDDING_PCS, embeddings_scaled.shape[1])
        pca_embeddings = PCA(n_components=n_components_for_context, random_state=RANDOM_STATE)
        embeddings_pcs = pca_embeddings.fit_transform(embeddings_scaled)
        
        full_cell_embedding_map = dict(zip(embedding_cell_ids_full, embeddings_pcs))
        for cell_id in unique_cells:
            if cell_id in full_cell_embedding_map:
                cell2vec[cell_id] = full_cell_embedding_map[cell_id]
                
    elif CELL_CONTEXT_MODE == 'onehot':
        print("Using 'onehot' context...")
        cell_dummies_df = pd.get_dummies(unique_cells, prefix='cell', dtype=int, drop_first=True)
        cell2vec = dict(zip(unique_cells, cell_dummies_df.values))
        
    if CELL_CONTEXT_MODE != 'onehot':
        missing_cells = set(unique_cells) - set(cell2vec.keys())
        if missing_cells:
            print(f"Warning: Missing rich context for {len(missing_cells)} cells. Filtering them out.")
            df = df[~df['cell_id'].isin(missing_cells)]
            original_indices = df.index
            X = X[original_indices]
            cell_ids = df['cell_id'].values
            pert_ids = df['pert_id'].values
            unique_cells = np.unique(cell_ids)
            pert_dummies = pert_dummies.loc[original_indices].reset_index(drop=True)
            if USE_FULL_CONTEXT_FEATURES:
                pert_unit_dummies = pert_unit_dummies.loc[original_indices].reset_index(drop=True)
                pert_time = pert_time[original_indices]
                pert_dose = pert_dose[original_indices]
                ignore_time = ignore_time[original_indices]
                ignore_dose = ignore_dose[original_indices]

C_global = None
if MODEL_MODE == 'contextualized':
    print("Constructing Global Context Matrix (Fix for Scaling Issues)...")
    
    cell_context_matrix = np.array([cell2vec[cid] for cid in cell_ids])
    
    continuous_parts = []
    categorical_parts = [pert_dummies.values]
    
    if CELL_CONTEXT_MODE == 'onehot':
        categorical_parts.append(cell_context_matrix)
    else:
        continuous_parts.append(cell_context_matrix)
        
    if USE_FULL_CONTEXT_FEATURES:
        categorical_parts.extend([pert_unit_dummies.values, ignore_time, ignore_dose])
        continuous_parts.extend([pert_time, pert_dose])
        
    if continuous_parts:
        C_continuous_raw = np.hstack(continuous_parts)
        print(f"Global Scaling applied to {C_continuous_raw.shape[1]} continuous features.")
        scaler_global_ctx = StandardScaler()
        C_continuous_scaled = scaler_global_ctx.fit_transform(C_continuous_raw)
        
        C_categorical_raw = np.hstack(categorical_parts)
        C_global = np.hstack([C_continuous_scaled, C_categorical_raw]).astype(np.float32)
    else:
        C_global = np.hstack(categorical_parts).astype(np.float32)
        
    print(f"Global Context Matrix Constructed: {C_global.shape}")

X_train_list, X_test_list = [], []
C_train_list, C_test_list = [], []
cell_ids_train_list, cell_ids_test_list = [], []
pert_ids_train_list, pert_ids_test_list = [], []

context_pairs_tuples = list(zip(cell_ids, pert_ids))
unique_pairs = sorted(list(set(context_pairs_tuples)))

print(f"\nSplitting data within {len(unique_pairs)} unique (cell, pert) context pairs...")

for pair in tqdm(unique_pairs, desc="Splitting data by (cell, pert) pair"):
    cell_id, pert_id = pair
    
    pair_mask = (cell_ids == cell_id) & (pert_ids == pert_id)
    
    if np.sum(pair_mask) < 2:
        continue
        
    X_pair = X[pair_mask]
    cell_ids_pair = cell_ids[pair_mask]
    pert_ids_pair = pert_ids[pair_mask]
    
    if MODEL_MODE == 'contextualized':
        C_pair = C_global[pair_mask]
        
        (X_train_pair, X_test_pair, 
         C_train_pair, C_test_pair,
         ids_train_pair, ids_test_pair, 
         perts_train_pair, perts_test_pair) = train_test_split(
            X_pair, C_pair, cell_ids_pair, pert_ids_pair,
            test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        C_train_list.append(C_train_pair)
        C_test_list.append(C_test_pair)
    else:
        (X_train_pair, X_test_pair, 
         ids_train_pair, ids_test_pair, 
         perts_train_pair, perts_test_pair) = train_test_split(
            X_pair, cell_ids_pair, pert_ids_pair,
            test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    
    X_train_list.append(X_train_pair)
    X_test_list.append(X_test_pair)
    cell_ids_train_list.append(ids_train_pair)
    cell_ids_test_list.append(ids_test_pair)
    pert_ids_train_list.append(perts_train_pair)
    pert_ids_test_list.append(perts_test_pair)

if not X_train_list:
    raise RuntimeError("No (cell, pert) pairs had enough samples to train a model.")

X_train = np.vstack(X_train_list)
X_test = np.vstack(X_test_list)
cell_ids_train = np.concatenate(cell_ids_train_list)
cell_ids_test = np.concatenate(cell_ids_test_list)
pert_ids_train = np.concatenate(pert_ids_train_list)
pert_ids_test = np.concatenate(pert_ids_test_list)

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

if MODEL_MODE == 'cell_specific':
    print("\n--- Running Model: Context-Specific (GroupedNetworks) ---")
    train_context_pairs = [f"{c}_{p}" for c, p in zip(cell_ids_train, pert_ids_train)]
    test_context_pairs = [f"{c}_{p}" for c, p in zip(cell_ids_test, pert_ids_test)]

    all_unique_pairs_sorted = np.unique(np.concatenate((train_context_pairs, test_context_pairs)))
    pair_to_int_map = {pair: i for i, pair in enumerate(all_unique_pairs_sorted)}
    
    train_labels = np.array([pair_to_int_map[pair] for pair in train_context_pairs])
    test_labels = np.array([pair_to_int_map[pair] for pair in test_context_pairs])
    
    grouped_cn = GroupedNetworks(CorrelationNetwork)
    unique_train_labels = np.unique(train_labels)

    def fit_model_for_group(label):
        mask = train_labels == label
        X_group = X_train_norm[mask]
        model = CorrelationNetwork()
        model.fit(X_group)
        return (label, model)

    print(f"Fitting {len(unique_train_labels)} context-specific models in parallel...")
    jobs = [delayed(fit_model_for_group)(label) for label in unique_train_labels]
    results = Parallel(n_jobs=-1)(tqdm(jobs, desc="Training context-specific models"))
    grouped_cn.models = {label: model for label, model in results}

    mse_train_per_sample = grouped_cn.measure_mses(X_train_norm, train_labels)
    mse_test_per_sample = grouped_cn.measure_mses(X_test_norm, test_labels)

    print(f"Overall MSE on train set: {np.mean(mse_train_per_sample):.4f}")
    print(f"Overall MSE on test set: {np.mean(mse_test_per_sample):.4f}")

    print("\nPer-Context Performance (Aggregated by Cell Line):")
    all_unique_cells_in_split = np.unique(np.concatenate((cell_ids_train, cell_ids_test)))
    for cell_id in sorted(all_unique_cells_in_split):
        tr_mask = cell_ids_train == cell_id
        te_mask = cell_ids_test == cell_id
        tr_mse = np.mean(mse_train_per_sample[tr_mask]) if tr_mask.any() else np.nan
        te_mse = np.mean(mse_test_per_sample[te_mask]) if te_mask.any() else np.nan
        print(f'{cell_id:<15}   {tr_mse:8.4f}    {te_mse:8.4f}    {tr_mask.sum():6d}    {te_mask.sum():6d}')

elif MODEL_MODE == 'population':
    print("\n--- Running Model: Population (CorrelationNetwork) ---")
    pop_cn = CorrelationNetwork()
    pop_cn.fit(X_train_norm)
    mse_train = np.mean(pop_cn.measure_mses(X_train_norm))
    mse_test = np.mean(pop_cn.measure_mses(X_test_norm))
    print(f"Overall MSE on train set: {mse_train:.4f}")
    print(f"Overall MSE on test set: {mse_test:.4f}")

elif MODEL_MODE == 'contextualized':
    print("\n--- Running Model: Contextualized (Lightning) ---")
    
    C_train = np.vstack(C_train_list).astype(np.float32)
    C_test = np.vstack(C_test_list).astype(np.float32)
    
    print(f"Final Context Matrix Shapes: Train={C_train.shape}, Test={C_test.shape}")

    contextualized_model = ContextualizedCorrelation(
        context_dim=C_train.shape[1],
        x_dim=X_train_norm.shape[1],
        encoder_type='mlp',
        num_archetypes=50,
    )

    train_indices, val_indices = train_test_split(
        np.arange(len(X_train_norm)), test_size=0.2, random_state=RANDOM_STATE
    )
    C_val = C_train[val_indices]
    X_val = X_train_norm[val_indices]
    C_train_split = C_train[train_indices]
    X_train_split = X_train_norm[train_indices]

    datamodule = CorrelationDataModule(
        C_train=C_train_split,
        X_train=X_train_split,
        C_val=C_val,
        X_val=X_val,
        C_test=C_test,
        X_test=X_test_norm,
        C_predict=np.concatenate((C_train, C_test), axis=0),
        X_predict=np.concatenate((X_train_norm, X_test_norm), axis=0),
        batch_size=256,
    )

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best_model',
    )
    
    trainer = Trainer(
        max_epochs=25,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(contextualized_model, datamodule=datamodule)

    print(f"Testing model...")
    _ = trainer.test(contextualized_model, datamodule.train_dataloader())
    _ = trainer.test(contextualized_model, datamodule.test_dataloader())
    
    output_dir = Path(checkpoint_callback.best_model_path).parent / 'predictions'
    writer_callback = PredictionWriter(output_dir=output_dir, write_interval='batch')
    trainer_pred = Trainer(accelerator='auto', devices='auto', callbacks=[checkpoint_callback, writer_callback])
    
    print("Making predictions...")
    _ = trainer_pred.predict(contextualized_model, datamodule=datamodule)

    print("Compiling predictions...")
    C_train_32 = C_train.astype(np.float32)
    C_test_32 = C_test.astype(np.float32)

    all_correlations, all_betas, all_mus = {}, {}, {}
    pred_files = glob.glob(str(output_dir / 'predictions_*.pt'))
        
    for file in pred_files:
        preds = torch.load(file)
        for context, correlation, beta, mu in zip(preds['contexts'], preds['correlations'], preds['betas'], preds['mus']):
            context_tuple = tuple(context.to(torch.float32).tolist())
            all_correlations[context_tuple] = correlation.cpu().numpy()
            all_betas[context_tuple] = beta.cpu().numpy()
            all_mus[context_tuple] = mu.cpu().numpy()

    correlations_train = np.array([all_correlations[tuple(row)] for row in C_train_32])
    correlations_test = np.array([all_correlations[tuple(row)] for row in C_test_32])
    betas_train = np.array([all_betas[tuple(row)] for row in C_train_32])
    betas_test = np.array([all_betas[tuple(row)] for row in C_test_32])
    mus_train = np.array([all_mus[tuple(row)] for row in C_train_32])
    mus_test = np.array([all_mus[tuple(row)] for row in C_test_32])

    def measure_mses(betas, mus, X):
        mses = np.zeros(len(X))
        n_features = X.shape[-1]
        for i in range(len(X)):
            sample_mse = 0
            for j in range(n_features):
                for k in range(n_features):
                    residual = X[i, j] - betas[i, j, k] * X[i, k] - mus[i, j, k]
                    sample_mse += residual**2
            mses[i] = sample_mse / (n_features ** 2)
        return mses

    mse_train = measure_mses(betas_train, mus_train, X_train_norm)
    mse_test = measure_mses(betas_test, mus_test, X_test_norm)
    
    print("\n" + "=" * 50)
    print(f"Performance Metrics: CONTEXTUALIZED ({CELL_CONTEXT_MODE})")
    print(f"Overall MSE on train set: {np.mean(mse_train):.4f}")
    print(f"Overall MSE on test set: {np.mean(mse_test):.4f}")
    print("=" * 50)

    print("\nPer-Context Performance:")
    all_unique_cells = np.union1d(cell_ids_train, cell_ids_test)
    for cell_id in sorted(all_unique_cells):
        tr_mask = cell_ids_train == cell_id
        te_mask = cell_ids_test == cell_id
        tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
        te_mse = mse_test[te_mask].mean() if te_mask.any() else np.nan
        print(f'{cell_id:<15}   {tr_mse:8.4f}    {te_mse:8.4f}    {tr_mask.sum():6d}    {te_mask.sum():6d}')
