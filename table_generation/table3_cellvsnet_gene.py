import torch
import lightning as pl
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, ParameterGrid
import anndata as ad
import warnings
import os
from pathlib import Path
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from contextualized.baselines.networks import CorrelationNetwork

RANDOM_STATE = 10

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
seed_everything(RANDOM_STATE, workers=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')


def main():
    DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
    
    pert_to_fit_on = ['trt_lig']
    pert_name = pert_to_fit_on[0]
    EMBEDDINGS_TO_RUN = {
        'AIDOcell': DATA_DIR / 'gene_embeddings/AIDOcell_100M_Norman_Aligned_(D=640)',
        'AIDOdna': DATA_DIR / 'gene_embeddings/AIDOdna_(D=4352)',
        'AIDOprot': DATA_DIR / 'gene_embeddings/AIDOprot_mean_(D=384)',
        'AIDOprot_struct': DATA_DIR / 'gene_embeddings/AIDOprot_mean_(D=384)',
        'PCA': DATA_DIR / 'gene_embeddings/PCA_gene_embeddings.h5ad',
    }

    PATH_L1000 = DATA_DIR / 'pert_type_csvs' / f'{pert_name}.csv'
    if pert_to_fit_on == ['trt_sh']:
        PATH_L1000 = DATA_DIR / 'trt_sh_genes_qc.csv'
    if pert_to_fit_on == ['trt_cp']:
        PATH_L1000 = DATA_DIR / 'trt_cp_smiles_qc.csv'
    SPLIT_MAP_PATH = DATA_DIR / f'gene_embeddings/unseen_perturbation_splits/{pert_name}_split_map.csv'

    PATH_CTLS = DATA_DIR / 'ctrls.csv'
    PERT_INFO = DATA_DIR / 'gene_embeddings/perts_targets.csv'

    N_DATA_PCS = 50
    N_GENE_EMB_PCS = 256
    PERTURBATION_HOLDOUT_SIZE = 0.2
    SUBSAMPLE_FRACTION = None

    all_run_results = []

    print("Finding the intersection of all perturbations across all embedding files...")
    pert_info_df = pd.read_csv(PERT_INFO)
    union_of_pert_ids = set()
    intersection_of_pert_ids = None
    loaded_embs_cache = {}  # cache to avoid double-loading h5ad files

    for emb_name, emb_path in EMBEDDINGS_TO_RUN.items():
        try:
            embs = ad.read_h5ad(emb_path)
            embs.obs = embs.obs.set_index('symbol')
            loaded_embs_cache[emb_name] = embs
            available_symbols = set(embs.obs.index)

            pert_info_filtered = pert_info_df.dropna(subset=['pert_iname']).copy()
            pert_info_filtered['_targets'] = pert_info_filtered['pert_iname'].str.split(';').apply(
                lambda genes: {g.strip() for g in genes}
            )
            mask = pert_info_filtered['_targets'].apply(lambda t: not t.isdisjoint(available_symbols))
            pert_ids_in_emb = set(pert_info_filtered.loc[mask, 'pert_id'])

            print(f"  - Found {len(pert_ids_in_emb)} perturbations for '{emb_name}'")
            union_of_pert_ids.update(pert_ids_in_emb)

            if intersection_of_pert_ids is None:
                intersection_of_pert_ids = pert_ids_in_emb
            else:
                intersection_of_pert_ids.intersection_update(pert_ids_in_emb)

        except FileNotFoundError:
            print(f"  - WARNING: Embedding file not found at {emb_path}. Skipping.")
    
    if intersection_of_pert_ids is None:
        raise ValueError("No common perturbations found. Check embedding paths and data.")
        
    print(f"\nTotal unique perturbations found across all files (union): {len(union_of_pert_ids)}")
    print(f"Total perturbations available in ALL files (intersection): {len(intersection_of_pert_ids)}")

    valid_embeddings = {
        emb_name: EMBEDDINGS_TO_RUN[emb_name]
        for emb_name in EMBEDDINGS_TO_RUN
        if emb_name in loaded_embs_cache
    }

    def get_gene_embeddings(df, embs, pert_info_path):
        embs_symbols_set = set(embs.obs.index)

        # Build a fast symbol -> dense vector dict once
        X_dense = embs.X.toarray() if hasattr(embs.X, 'toarray') else np.asarray(embs.X)
        symbol_to_vec = {sym: X_dense[i] for i, sym in enumerate(embs.obs.index)}

        try:
            mapping = pd.read_csv(pert_info_path)
        except FileNotFoundError:
            print(f"Error: Perturbation info file not found at {pert_info_path}")
            return {}, []

        unique_perts = df[['pert_id']].drop_duplicates()
        # Explode pert_iname by ';' so each gene gets its own row
        mapping_exp = mapping.dropna(subset=['pert_iname']).copy()
        mapping_exp['gene'] = mapping_exp['pert_iname'].str.split(';')
        mapping_exp = mapping_exp.explode('gene')
        mapping_exp['gene'] = mapping_exp['gene'].str.strip()
        mapping_exp = mapping_exp[mapping_exp['gene'] != '']

        # Keep only perts in our dataset and genes we have embeddings for
        mapping_exp = mapping_exp[mapping_exp['pert_id'].isin(unique_perts['pert_id'])]
        mapping_exp = mapping_exp[mapping_exp['gene'].isin(embs_symbols_set)]

        # Average embeddings per pert_id
        if not mapping_exp.empty:
            vecs = np.vstack(mapping_exp['gene'].map(symbol_to_vec).values)
            mapping_exp = mapping_exp.copy()
            mapping_exp['_vec_idx'] = np.arange(len(mapping_exp))
            grouped = mapping_exp.groupby('pert_id')['_vec_idx'].apply(list)
            pert_embeddings = {pid: vecs[idxs].mean(axis=0) for pid, idxs in grouped.items()}
        else:
            pert_embeddings = {}

        not_found_pert_ids = [pid for pid in unique_perts['pert_id'] if pid not in pert_embeddings]
        return pert_embeddings, not_found_pert_ids

    df = pd.read_csv(PATH_L1000, engine='pyarrow')
    df = df[df['pert_type'].isin(pert_to_fit_on)]
    bad = (
        (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
        (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
    )
    df = df[~bad]
    df = df.dropna(subset=['pert_id'])
    df = df[df['pert_id'] != '']
    
    df = df[df['pert_id'].isin(intersection_of_pert_ids)]
    print(f"\nProcessing {len(df)} samples after filtering for the intersection of perturbations...")

    if SUBSAMPLE_FRACTION is not None:
        df = df.sample(frac=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE)
        print(f"Subsampled to {len(df)} samples ({SUBSAMPLE_FRACTION*100}% of data)")
    
    unique_pert_ids = df['pert_id'].unique()
    print(f"Found {len(unique_pert_ids)} unique perturbations (pert_id) for the unified dataset")

    if SPLIT_MAP_PATH is not None:
        print(f"Using split map file: {SPLIT_MAP_PATH}")
        split_map = pd.read_csv(SPLIT_MAP_PATH)[['inst_id', 'split']]
        df = df.merge(split_map, on='inst_id', how='inner')
        df_train_base = df[df['split'] == 'train'].drop(columns='split').copy()
        df_test_base  = df[df['split'] == 'test'].drop(columns='split').copy()
        df = df.drop(columns='split')
    else:
        pert_ids_train, pert_ids_test = train_test_split(
            unique_pert_ids,
            test_size=PERTURBATION_HOLDOUT_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"Perturbation split: {len(pert_ids_train)} train, {len(pert_ids_test)} test perturbations")
        df_train_base = df[df['pert_id'].isin(pert_ids_train)].copy()
        df_test_base  = df[df['pert_id'].isin(pert_ids_test)].copy()

    print(f"Sample split: {len(df_train_base)} train, {len(df_test_base)} test samples")
    n_total_samples = len(df)  # save before df is freed inside the loop

    for emb_name, EMB_H5AD in valid_embeddings.items():

        print(f"\n{'='*25} RUNNING FOR EMBEDDING: {emb_name} {'='*25}")
        
        RESULTS_DIR = f'{pert_name}_{emb_name}'
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"All results will be saved in: {RESULTS_DIR}")

        df_train = df_train_base.copy()
        df_test = df_test_base.copy()

        pert_time_mean, pert_dose_mean = None, None
        for df_split, split_name in [(df_train, 'train'), (df_test, 'test')]:
            df_split['ignore_flag_pert_time'] = (df_split['pert_time'] == -666).astype(int)
            df_split['ignore_flag_pert_dose'] = (df_split['pert_dose'] == -666).astype(int)
            for col in ['pert_time', 'pert_dose']:
                if split_name == 'train':
                    mean_val = df_split.loc[df_split[col] != -666, col].mean()
                    if col == 'pert_time':
                        pert_time_mean = mean_val
                    else:
                        pert_dose_mean = mean_val
                else:
                    mean_val = pert_time_mean if col == 'pert_time' else pert_dose_mean
                df_split[col] = df_split[col].replace(-666, mean_val)

        print("Loading gene embeddings...")
        embs = loaded_embs_cache[emb_name]
        all_pert_embeddings, not_found_list = get_gene_embeddings(
            pd.concat([df_train, df_test], ignore_index=True), embs, PERT_INFO
        )
        if not_found_list:
            print(f"Warning: {len(not_found_list)} perturbations in the unified set do not have an embedding in '{emb_name}'. They will be represented by zero vectors.")

        def process_data_split(df_split, split_name):
            numeric_cols = df_split.select_dtypes(include=[np.number]).columns
            drop_cols = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
            feature_cols = [c for c in numeric_cols if c not in drop_cols]
            X_raw = df_split[feature_cols].values

            print(f"Generating gene embeddings for {split_name} set...")
            emb_shape = next(iter(all_pert_embeddings.values())).shape[0] if all_pert_embeddings else 0
            rows = [all_pert_embeddings.get(pid, np.zeros(emb_shape)).flatten() for pid in df_split['pert_id']]
            gene_embs = np.array(rows) if rows else np.zeros((0, emb_shape))
            print(f"Generated gene embeddings for {split_name}: shape {gene_embs.shape}")
            
            pert_time = df_split['pert_time'].to_numpy().reshape(-1, 1)
            pert_dose = df_split['pert_dose'].to_numpy().reshape(-1, 1)
            ignore_time = df_split['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
            ignore_dose = df_split['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)
            return X_raw, gene_embs, pert_time, pert_dose, ignore_time, ignore_dose

        X_raw_train, gene_embs_train, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train = process_data_split(df_train, 'train')
        X_raw_test, gene_embs_test, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test = process_data_split(df_test, 'test')

        # Drop heavy feature columns — keep only metadata needed later
        _keep_cols = ['cell_id', 'inst_id', 'pert_time', 'pert_dose']
        df_train = df_train[[c for c in _keep_cols if c in df_train.columns]].copy()
        df_test  = df_test[[c for c in _keep_cols if c in df_test.columns]].copy()

        if gene_embs_train.shape[1] == 0:
            print(f"Warning: No gene embeddings were loaded for {emb_name}. Using zero vectors.")

        print("Applying scaling...")
        scaler_genes = StandardScaler()
        X_train_scaled = scaler_genes.fit_transform(X_raw_train)
        X_test_scaled = scaler_genes.transform(X_raw_test)
        del X_raw_train, X_raw_test  # ~2.9 GB freed

        scaler_embs = StandardScaler()
        if gene_embs_train.shape[1] > 0:
            gene_embs_train_scaled = scaler_embs.fit_transform(gene_embs_train.astype(float))
            gene_embs_test_scaled = scaler_embs.transform(gene_embs_test.astype(float))
        else:
            gene_embs_train_scaled = gene_embs_train
            gene_embs_test_scaled = gene_embs_test
        del gene_embs_train, gene_embs_test  # ~1.5 GB freed

        if emb_name != 'PCA' and gene_embs_train_scaled.shape[1] > N_GENE_EMB_PCS:
            print(f"Applying PCA to gene embeddings for '{emb_name}', reducing from {gene_embs_train_scaled.shape[1]} to {N_GENE_EMB_PCS} components...")
            pca_embs = PCA(n_components=N_GENE_EMB_PCS, random_state=RANDOM_STATE)
            gene_embs_train_scaled = pca_embs.fit_transform(gene_embs_train_scaled)
            gene_embs_test_scaled = pca_embs.transform(gene_embs_test_scaled)
        elif emb_name == 'PCA':
            print("Skipping gene embedding PCA because emb_name is 'PCA'.")
        elif gene_embs_train_scaled.shape[1] <= N_GENE_EMB_PCS:
            print(f"Skipping gene embedding PCA because dimension ({gene_embs_train_scaled.shape[1]}) is already <= {N_GENE_EMB_PCS}.")


        ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
        unique_cells_all = np.sort(np.union1d(df_train['cell_id'].unique(), df_test['cell_id'].unique()))
        ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_all)]
        
        scaler_ctrls = StandardScaler()
        ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
        n_ctrl_pcs = min(50, ctrls_scaled.shape[0])
        
        if n_ctrl_pcs == 0:
            print(f"ERROR: No control cell data found for cells in this split. Skipping {emb_name} run.")
            continue

        pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
        ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
        cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
        if not cell2vec:
            print(f"ERROR: No common cell IDs found after processing controls. Skipping {emb_name} run.")
            continue
        print(f"Loaded and processed control embeddings for {len(cell2vec)} unique cells.")

        def build_context_matrix(df_split, X_scaled_split, gene_embs_scaled, pert_time, pert_dose, ignore_time, ignore_dose, scaler_context=None, is_train=False):
            cell_ids = df_split['cell_id'].to_numpy()
            valid_cell_ids = np.sort([c for c in df_split['cell_id'].unique() if c in cell2vec])

            if not len(valid_cell_ids):
                if is_train:
                    return None, None, None, None
                return np.array([]), np.array([]), np.array([]), scaler_context

            # Build continuous context block in one pass
            masks = {c: (cell_ids == c) for c in valid_cell_ids}
            continuous_parts = [
                np.hstack([np.tile(cell2vec[c], (masks[c].sum(), 1)), pert_time[masks[c]], pert_dose[masks[c]]])
                for c in valid_cell_ids
            ]
            continuous_all = np.vstack(continuous_parts)

            if is_train:
                scaler_context = StandardScaler().fit(continuous_all)
                print(f"Fitted context scaler on {scaler_context.mean_.shape[0]} continuous context features")

            # Build final matrices, reusing already-computed continuous_parts
            X_final_parts, C_final_parts, cell_ids_final_parts = [], [], []
            offset = 0
            for c, cont in zip(valid_cell_ids, continuous_parts):
                n = cont.shape[0]
                mask = masks[c]
                C_final_parts.append(np.hstack([
                    scaler_context.transform(cont),
                    gene_embs_scaled[mask],
                    ignore_time[mask],
                    ignore_dose[mask],
                ]))
                X_final_parts.append(X_scaled_split[mask])
                cell_ids_final_parts.append(cell_ids[mask])
                offset += n

            if not C_final_parts:
                return np.array([]), np.array([]), np.array([]), scaler_context

            return np.vstack(X_final_parts), np.vstack(C_final_parts), np.concatenate(cell_ids_final_parts), scaler_context

        print("Building context matrices...")
        X_train, C_train, cell_ids_train, scaler_context = build_context_matrix(df_train, X_train_scaled, gene_embs_train_scaled, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train, is_train=True)

        if X_train is None:
            print(f"ERROR: Could not build training context matrix (e.g., no valid cells). Skipping {emb_name} run.")
            continue
            
        X_test, C_test, cell_ids_test, _ = build_context_matrix(df_test, X_test_scaled, gene_embs_test_scaled, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test, scaler_context=scaler_context)

        # Free scaled arrays no longer needed before training
        del X_train_scaled, X_test_scaled
        del gene_embs_train_scaled, gene_embs_test_scaled

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"ERROR: Train ({X_train.shape[0]}) or Test ({X_test.shape[0]}) set is empty after processing. Skipping {emb_name} run.")
            continue
            
        print(f'Context matrix:    train {C_train.shape}    test {C_test.shape}')
        print(f'Feature matrix:    train {X_train.shape}    test {X_test.shape}')

        print("Applying PCA...")
        pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
        X_train_pca = pca_data.fit_transform(X_train)
        del X_train  # free ~2.3 GB (979-col) before next allocations
        X_test_pca = pca_data.transform(X_test)
        del X_test

        pca_scaler = StandardScaler()
        X_train_norm = pca_scaler.fit_transform(X_train_pca)
        del X_train_pca
        X_test_norm = pca_scaler.transform(X_test_pca)
        del X_test_pca
        print(f'Normalized PCA features: train {X_train_norm.shape}    test {X_test_norm.shape}')

        X_train, X_test = X_train_norm, X_test_norm

        pop_model = CorrelationNetwork()
        pop_model.fit(X_train)
        pop_train_mse = pop_model.measure_mses(X_train).mean()
        pop_test_mse = pop_model.measure_mses(X_test).mean()
        print(f"Population Train MSE: {pop_train_mse:.4f}")
        print(f"Population Test MSE: {pop_test_mse:.4f}")

        contextualized_model = ContextualizedCorrelation(
            context_dim=C_train.shape[1],
            x_dim=X_train.shape[1],
            encoder_type='mlp',
            num_archetypes=50,
        )
        train_indices, val_indices = train_test_split(np.arange(len(X_train)), test_size=0.2, random_state=RANDOM_STATE)
        
        datamodule = CorrelationDataModule(
            C_train=C_train[train_indices], X_train=X_train[train_indices],
            C_val=C_train[val_indices], X_val=X_train[val_indices],
            C_test=C_test, X_test=X_test,
            C_predict=C_test, X_predict=X_test,  # inference is done manually; avoid extra concat copy
            batch_size=32,
        )

        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model')

        trainer = Trainer(
            default_root_dir=RESULTS_DIR, max_epochs=5, accelerator='auto', devices='auto',
            callbacks=[checkpoint_callback],
            deterministic=True
        )
        trainer.fit(contextualized_model, datamodule=datamodule)

        print(f"Best model path: {checkpoint_callback.best_model_path}")

        # Load best checkpoint weights into model, then run inference directly.
        # Outputs are streamed to memmap files to avoid OOM (~10 GB for full dataset).
        print("Loading best checkpoint and running direct inference...")
        ckpt = torch.load(checkpoint_callback.best_model_path, map_location='cpu', weights_only=False)
        contextualized_model.load_state_dict(ckpt['state_dict'])
        contextualized_model.cpu()  # run inference on CPU to avoid competing with MPS for RAM
        contextualized_model.eval()
        device = torch.device('cpu')

        n_features = X_train.shape[1]
        n_train, n_test = len(C_train), len(C_test)
        n_full = n_train + n_test
        out_shape = (n_full, n_features, n_features)

        corr_path  = os.path.join(RESULTS_DIR, 'full_dataset_correlations.npy')
        betas_path = os.path.join(RESULTS_DIR, 'full_dataset_betas.npy')
        mus_path   = os.path.join(RESULTS_DIR, 'full_dataset_mus.npy')

        # Pre-allocate proper .npy files on disk (loadable with np.load) — written in batches
        corrs_mm = np.lib.format.open_memmap(corr_path,  mode='w+', dtype='float32', shape=out_shape)
        betas_mm = np.lib.format.open_memmap(betas_path, mode='w+', dtype='float32', shape=out_shape)
        mus_mm   = np.lib.format.open_memmap(mus_path,   mode='w+', dtype='float32', shape=out_shape)

        def run_inference_to_mm(C_np, offset, batch_size=256):
            C_tensor = torch.tensor(C_np.astype(np.float32))
            with torch.no_grad():
                for start in range(0, len(C_tensor), batch_size):
                    end = min(start + batch_size, len(C_tensor))
                    out = contextualized_model.predict_step({"contexts": C_tensor[start:end].to(device)}, 0)
                    betas_mm[offset + start:offset + end] = out["betas"].cpu().numpy()
                    mus_mm[offset + start:offset + end]   = out["mus"].cpu().numpy()
                    corrs_mm[offset + start:offset + end] = out["correlations"].cpu().numpy()
                    if start % (batch_size * 100) == 0:
                        print(f"  {offset + end}/{n_full} samples done", flush=True)

        print("Running inference on train set...")
        run_inference_to_mm(C_train, offset=0)
        print("Running inference on test set...")
        run_inference_to_mm(C_test, offset=n_train)
        betas_mm.flush(); mus_mm.flush(); corrs_mm.flush()
        print(f"Saved betas/mus/correlations to {RESULTS_DIR}  shape={out_shape}")

        # Views into train/test slices (no copy — reads from disk as needed)
        betas_train       = betas_mm[:n_train]
        mus_train         = mus_mm[:n_train]
        correlations_train = corrs_mm[:n_train]
        betas_test        = betas_mm[n_train:]
        mus_test          = mus_mm[n_train:]
        correlations_test  = corrs_mm[n_train:]

        X_full = np.concatenate([X_train, X_test], axis=0)
        C_full = np.concatenate([C_train, C_test], axis=0)

        print(f"Full dataset predictions compiled:")
        print(f"  Betas/Mus/Correlations shape: {out_shape}")
        print(f"  Features shape: {X_full.shape}")
        print(f"  Context shape: {C_full.shape}")

        def measure_mses(betas, mus, X, batch_size=1000):
            mses = np.empty(len(X))
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                b = np.array(betas[start:end])
                m = np.array(mus[start:end])
                x = X[start:end]
                residuals = x[:, :, np.newaxis] - b * x[:, np.newaxis, :] - m
                chunk = np.sum(residuals ** 2, axis=(1, 2)) / (x.shape[-1] ** 2)
                chunk[np.isnan(b).any(axis=(1, 2))] = np.nan
                mses[start:end] = chunk
            return mses

        mse_train = measure_mses(betas_train, mus_train, X_train)
        mse_test  = measure_mses(betas_test,  mus_test,  X_test)
        mse_full  = measure_mses(betas_mm,    mus_mm,    X_full)
        
        context_train_mse = np.nanmean(mse_train)
        context_test_mse = np.nanmean(mse_test)

        print(f"Contextualized Train MSE: {context_train_mse:.4f}")
        print(f"Contextualized Test MSE: {context_test_mse:.4f}")
        print(f"Contextualized Full dataset MSE: {np.nanmean(mse_full):.4f}")

        # Free large arrays before building df_full to avoid OOM
        del C_full, X_full
        del C_train, C_test

        train_mask = df_train['cell_id'].isin(cell2vec.keys())
        test_mask = df_test['cell_id'].isin(cell2vec.keys())
        df_full = pd.concat([df_train[train_mask], df_test[test_mask]]).reset_index(drop=True)
        
        results_df = pd.DataFrame({
            'split': ['train'] * len(X_train) + ['test'] * len(X_test),
            'mse': mse_full,
            'cell_id': np.concatenate([cell_ids_train, cell_ids_test]),
        })
        
        if len(results_df) == len(df_full):
            results_df['inst_id'] = df_full['inst_id'].values
            results_df['pert_time'] = df_full['pert_time'].values
            results_df['pert_dose'] = df_full['pert_dose'].values
        else:
            print(f"Warning: Length mismatch between results ({len(results_df)}) and filtered df ({len(df_full)}). Skipping some column merges.")

        csv_path = os.path.join(RESULTS_DIR, 'full_dataset_predictions.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Saved predictions CSV to {csv_path}")

        print(f"\nPer-cell performance breakdown:")
        print("Cell ID                Train MSE    Test MSE     Train N  Test N")
        print("─" * 60)
        for cell_id in sorted(np.union1d(cell_ids_train, cell_ids_test)):
            tr_mask, te_mask = (cell_ids_train == cell_id), (cell_ids_test == cell_id)
            tr_mse = np.nanmean(mse_train[tr_mask]) if tr_mask.any() else np.nan
            te_mse = np.nanmean(mse_test[te_mask]) if te_mask.any() else np.nan
            print(f'{cell_id:<15}    {tr_mse:8.4f}    {te_mse:8.4f}     {tr_mask.sum():6d}     {te_mask.sum():6d}')
        
        current_run_summary = {
            'Embedding': emb_name,
            'Population Train MSE': pop_train_mse,
            'Population Test MSE': pop_test_mse,
            'Contextualized Train MSE': context_train_mse,
            'Contextualized Test MSE': context_test_mse,
            'Train Samples': len(X_train),
            'Test Samples': len(X_test)
        }
        all_run_results.append(current_run_summary)
        
        print(f"\n--- RESULTS SUMMARY FOR: {emb_name} ---")
        print(f"  Train Samples:                {len(X_train)}")
        print(f"  Test Samples:                 {len(X_test)}")
        print(f"  Population Model Train MSE:     {pop_train_mse:.4f}")
        print(f"  Population Model Test MSE:      {pop_test_mse:.4f}")
        print(f"  Contextualized Model Train MSE: {context_train_mse:.4f}")
        print(f"  Contextualized Model Test MSE: {context_test_mse:.4f}")
        print("-------------------------------------------\n")

    if all_run_results:
        print("\n\n" + "="*35 + " FINAL SUMMARY OF ALL RUNS " + "="*35)
        summary_df = pd.DataFrame(all_run_results)
        print(summary_df.to_string())
        
        BASE_SAVE_DIR = 'final_runs'
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)
        summary_csv_path = os.path.join(BASE_SAVE_DIR, 'all_runs_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary saved to: {summary_csv_path}")
        print("="*93 + "\n")
    else:
        print("\nNo runs were completed, so no summary to display.")

    print("\n" + "="*31 + " PERTURBATION SET DETAILS " + "="*31)
    filtered_out_perts = union_of_pert_ids - intersection_of_pert_ids
    
    print(f"Total Unique Perturbations Found (Union):            {len(union_of_pert_ids)}")
    print(f"Perturbations Common to All Embeddings (Intersection): {len(intersection_of_pert_ids)}")
    print(f"Perturbations Filtered Out (Not in all embeddings):  {len(filtered_out_perts)}")
    print(f"Perturbations the model was evaluated on (Post-QC):  {len(unique_pert_ids)}")
    print(f"Total Samples in Final Dataset:                      {n_total_samples}")

    print("="*87)


if __name__ == '__main__':
    main()
