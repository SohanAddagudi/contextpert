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
import glob
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from contextualized.baselines.networks import CorrelationNetwork
from contextualized.callbacks import PredictionWriter

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
        'PCA': DATA_DIR / 'gene_embeddings/PCA_gene_embeddings.h5ad',
        'AIDOcell': DATA_DIR / 'gene_embeddings/AIDOcell_100M_Norman_Aligned_(D=640)',
        'AIDOdna': DATA_DIR / 'gene_embeddings/AIDOdna_(D=4352)',
        'AIDOprot': DATA_DIR / 'gene_embeddings/AIDOprot_mean_(D=384)',
        'AIDOprot_struct': DATA_DIR / 'gene_embeddings/AIDOprot_mean_(D=384)',
    }

    PATH_L1000 = DATA_DIR / 'pert_type_csvs' / f'{pert_name}.csv'
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

    for emb_name, emb_path in EMBEDDINGS_TO_RUN.items():
        try:
            embs = ad.read_h5ad(emb_path)
            embs.obs = embs.obs.set_index('symbol')
            available_symbols = set(embs.obs.index)
            
            pert_ids_in_emb = set()
            for _, row in pert_info_df.dropna(subset=['pert_iname']).iterrows():
                targets = set(str(row['pert_iname']).split(';'))
                if not targets.isdisjoint(available_symbols):
                    pert_ids_in_emb.add(row['pert_id'])
            
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


    def get_gene_embeddings(df, emb_h5ad_path, pert_info_path):
        try:
            embs = ad.read_h5ad(emb_h5ad_path)
            embs.obs = embs.obs.set_index('symbol')
            embs_symbols_set = set(embs.obs.index)
        except FileNotFoundError:
            print(f"Error: AnnData file not found at {emb_h5ad_path}")
            return {}, [], None
        
        try:
            mapping = pd.read_csv(pert_info_path)
            merged_df = pd.merge(df, mapping, on='pert_id', how='left')
        except FileNotFoundError:
            print(f"Error: Perturbation info file not found at {pert_info_path}")
            return {}, [], None
        
        pert_embeddings = {}
        not_found_pert_ids = []
        
        for pert_id in merged_df['pert_id'].unique():
            pert_data = merged_df[merged_df['pert_id'] == pert_id]
            
            pert_iname_strings = pert_data['pert_iname'].unique()
            
            embeddings_list = []
            all_target_genes = set()
            
            for iname_str in pert_iname_strings:
                if pd.isna(iname_str):
                    continue
                
                for gene_name in iname_str.split(';'):
                    cleaned_name = gene_name.strip()
                    if cleaned_name:
                        all_target_genes.add(cleaned_name)
            
            for gene_name in all_target_genes:
                if gene_name in embs_symbols_set:
                    embedding = embs[embs.obs.index.get_loc(gene_name)].X
                    embeddings_list.append(embedding)
                else:
                    pass

            if embeddings_list:
                stacked_embs = np.vstack(embeddings_list)
                pert_embeddings[pert_id] = np.mean(stacked_embs, axis=0)
            else:
                not_found_pert_ids.append(pert_id)
                
        return pert_embeddings, not_found_pert_ids, embs

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
    pert_ids_train, pert_ids_test = train_test_split(
        unique_pert_ids,
        test_size=PERTURBATION_HOLDOUT_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"Perturbation split: {len(pert_ids_train)} train, {len(pert_ids_test)} test perturbations")
    
    df_train_base = df[df['pert_id'].isin(pert_ids_train)].copy()
    df_test_base = df[df['pert_id'].isin(pert_ids_test)].copy()
    print(f"Sample split: {len(df_train_base)} train, {len(df_test_base)} test samples")


    for emb_name, EMB_H5AD in EMBEDDINGS_TO_RUN.items():
        
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
        all_pert_embeddings, not_found_list, embs = get_gene_embeddings(df, EMB_H5AD, PERT_INFO)
        if not_found_list:
            print(f"Warning: {len(not_found_list)} perturbations in the unified set do not have an embedding in '{emb_name}'. They will be represented by zero vectors.")

        def process_data_split(df_split, split_name, embs):
            numeric_cols = df_split.select_dtypes(include=[np.number]).columns
            drop_cols = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
            feature_cols = [c for c in numeric_cols if c not in drop_cols]
            X_raw = df_split[feature_cols].values
            
            print(f"Generating gene embeddings for {split_name} set...")
            emb_shape = next(iter(all_pert_embeddings.values())).shape[0] if all_pert_embeddings else 0
            gene_embs = np.array([
                all_pert_embeddings.get(pid, np.zeros(emb_shape)).flatten()
                for pid in df_split['pert_id']
            ])
            print(f"Generated gene embeddings for {split_name}: shape {gene_embs.shape}")
            
            pert_time = df_split['pert_time'].to_numpy().reshape(-1, 1)
            pert_dose = df_split['pert_dose'].to_numpy().reshape(-1, 1)
            ignore_time = df_split['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
            ignore_dose = df_split['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)
            return X_raw, gene_embs, pert_time, pert_dose, ignore_time, ignore_dose

        X_raw_train, gene_embs_train, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train = process_data_split(df_train, 'train', embs)
        X_raw_test, gene_embs_test, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test = process_data_split(df_test, 'test', embs)
        
        if gene_embs_train.shape[1] == 0:
            print(f"Warning: No gene embeddings were loaded for {emb_name}. Using zero vectors.")
            pass

        print("Applying scaling...")
        scaler_genes = StandardScaler()
        X_train_scaled = scaler_genes.fit_transform(X_raw_train)
        X_test_scaled = scaler_genes.transform(X_raw_test)
        
        scaler_embs = StandardScaler()
        if gene_embs_train.shape[1] > 0:
            gene_embs_train_scaled = scaler_embs.fit_transform(gene_embs_train.astype(float))
            gene_embs_test_scaled = scaler_embs.transform(gene_embs_test.astype(float))
        else:
            gene_embs_train_scaled = gene_embs_train
            gene_embs_test_scaled = gene_embs_test

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
            continuous_context_parts, X_final_parts, C_final_parts, cell_ids_final_parts = [], [], [], []

            if is_train:
                for cell_id in np.sort(df_split['cell_id'].unique()):
                    if cell_id not in cell2vec: continue
                    mask = (cell_ids == cell_id)
                    if not mask.any(): continue
                    continuous_context_parts.append(np.hstack([
                        np.tile(cell2vec[cell_id], (mask.sum(), 1)),
                        pert_time[mask],
                        pert_dose[mask],
                    ]))
                
                if not continuous_context_parts:
                    return None, None, None, None
                    
                scaler_context = StandardScaler().fit(np.vstack(continuous_context_parts))
                print(f"Fitted context scaler on {scaler_context.mean_.shape[0]} continuous context features")

            for cell_id in np.sort(df_split['cell_id'].unique()):
                if cell_id not in cell2vec:
                    print(f"Warning: Cell {cell_id} not found in control embeddings, skipping...")
                    continue
                mask = (cell_ids == cell_id)
                if not mask.any(): continue
                
                C_continuous = np.hstack([
                    np.tile(cell2vec[cell_id], (mask.sum(), 1)),
                    pert_time[mask],
                    pert_dose[mask],
                ])
                C_continuous_scaled = scaler_context.transform(C_continuous)
                
                C_final_parts.append(np.hstack([
                    C_continuous_scaled,
                    gene_embs_scaled[mask],
                    ignore_time[mask],
                    ignore_dose[mask],
                ]))
                X_final_parts.append(X_scaled_split[mask])
                cell_ids_final_parts.append(cell_ids[mask])
            
            if not C_final_parts:
                return np.array([]), np.array([]), np.array([]), scaler_context

            return np.vstack(X_final_parts), np.vstack(C_final_parts), np.concatenate(cell_ids_final_parts), scaler_context

        print("Building context matrices...")
        X_train, C_train, cell_ids_train, scaler_context = build_context_matrix(df_train, X_train_scaled, gene_embs_train_scaled, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train, is_train=True)

        if X_train is None:
            print(f"ERROR: Could not build training context matrix (e.g., no valid cells). Skipping {emb_name} run.")
            continue
            
        X_test, C_test, cell_ids_test, _ = build_context_matrix(df_test, X_test_scaled, gene_embs_test_scaled, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test, scaler_context=scaler_context)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"ERROR: Train ({X_train.shape[0]}) or Test ({X_test.shape[0]}) set is empty after processing. Skipping {emb_name} run.")
            continue
            
        print(f'Context matrix:    train {C_train.shape}    test {C_test.shape}')
        print(f'Feature matrix:    train {X_train.shape}    test {X_test.shape}')

        print("Applying PCA...")
        pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
        X_train_pca = pca_data.fit_transform(X_train)
        X_test_pca = pca_data.transform(X_test)
        
        pca_scaler = StandardScaler()
        X_train_norm = pca_scaler.fit_transform(X_train_pca)
        X_test_norm = pca_scaler.transform(X_test_pca)
        print(f'Normalized PCA features: train {X_train_norm.shape}    test {X_test_norm.shape}')
        
        train_group_ids = cell_ids_train
        test_group_ids = cell_ids_test
        
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
            C_predict=np.concatenate((C_train, C_test)), X_predict=np.concatenate((X_train, X_test)),
            batch_size=32,
        )

        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model')

        # CHANGE 2: Added deterministic=True to Trainer
        trainer = Trainer(
            default_root_dir=RESULTS_DIR, max_epochs=10, accelerator='auto', devices='auto',
            callbacks=[checkpoint_callback],
            deterministic=True
        )
        trainer.fit(contextualized_model, datamodule=datamodule)

        print(f"Best model path: {checkpoint_callback.best_model_path}")
        print("Testing model on full training data...")
        trainer.test(contextualized_model, datamodule.train_dataloader())
        print("Testing model on test data...")
        trainer.test(contextualized_model, datamodule.test_dataloader())

        output_dir = Path(checkpoint_callback.best_model_path).parent / 'predictions'
        writer_callback = PredictionWriter(output_dir=output_dir, write_interval='batch')

        # CHANGE 3: Added deterministic=True to prediction Trainer
        pred_trainer = Trainer(
            default_root_dir=RESULTS_DIR, accelerator='auto', devices='auto', callbacks=[writer_callback],
            deterministic=True
        )
        print("Making predictions on full dataset (train + test)...")
        pred_trainer.predict(contextualized_model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
        
        C_train_32 = C_train.astype(np.float32)
        C_test_32 = C_test.astype(np.float32)
        
        all_correlations, all_betas, all_mus = {}, {}, {}
        
        pred_files = glob.glob(str(output_dir / 'predictions_*.pt'))
        if not pred_files:
            print(f"ERROR: No prediction files found in {output_dir}. Skipping post-processing for {emb_name}.")
            continue
            
        for file in pred_files:
            preds = torch.load(file)
            for context, correlation, beta, mu in zip(preds['contexts'], preds['correlations'], preds['betas'], preds['mus']):
                context_tuple = tuple(context.to(torch.float32).tolist())
                all_correlations[context_tuple] = correlation.cpu().numpy()
                all_betas[context_tuple] = beta.cpu().numpy()
                all_mus[context_tuple] = mu.cpu().numpy()
        
        def get_preds(contexts, pred_dict, default_val):
            return np.array([pred_dict.get(tuple(c), default_val) for c in [tuple(row) for row in contexts]])

        corr_shape = (X_train.shape[1], X_train.shape[1])
        beta_shape = (X_train.shape[1], X_train.shape[1])
        mu_shape = (X_train.shape[1], X_train.shape[1])
        default_corr = np.full(corr_shape, np.nan)
        default_beta = np.full(beta_shape, np.nan)
        default_mu = np.full(mu_shape, np.nan)

        correlations_train = get_preds(C_train_32, all_correlations, default_corr)
        correlations_test = get_preds(C_test_32, all_correlations, default_corr)
        betas_train = get_preds(C_train_32, all_betas, default_beta)
        betas_test = get_preds(C_test_32, all_betas, default_beta)
        mus_train = get_preds(C_train_32, all_mus, default_mu)
        mus_test = get_preds(C_test_32, all_mus, default_mu)

        correlations_full = np.concatenate([correlations_train, correlations_test], axis=0)
        betas_full = np.concatenate([betas_train, betas_test], axis=0)
        mus_full = np.concatenate([mus_train, mus_test], axis=0)

        X_full = np.concatenate([X_train, X_test], axis=0)
        C_full = np.concatenate([C_train, C_test], axis=0)
        train_indices_full = np.arange(len(X_train))
        test_indices_full = np.arange(len(X_train), len(X_train) + len(X_test))
        
        print(f"Full dataset predictions compiled:")
        print(f"  Correlations shape: {correlations_full.shape}")
        print(f"  Betas shape: {betas_full.shape}")
        print(f"  Mus shape: {mus_full.shape}")
        print(f"  Features shape: {X_full.shape}")
        print(f"  Context shape: {C_full.shape}")

        def measure_mses(betas, mus, X):
            mses = np.zeros(len(X))
            for i in range(len(X)):
                if np.isnan(betas[i]).any():
                    mses[i] = np.nan
                    continue
                sample_mse = 0
                for j in range(X.shape[-1]):
                    for k in range(X.shape[-1]):
                        residual = X[i, j] - betas[i, j, k] * X[i, k] - mus[i, j, k]
                        sample_mse += residual**2
                mses[i] = sample_mse / (X.shape[-1] ** 2)
            return mses

        mse_train = measure_mses(betas_train, mus_train, X_train)
        mse_test = measure_mses(betas_test, mus_test, X_test)
        mse_full = measure_mses(betas_full, mus_full, X_full)
        
        context_train_mse = np.nanmean(mse_train)
        context_test_mse = np.nanmean(mse_test)

        print(f"Contextualized Train MSE: {context_train_mse:.4f}")
        print(f"Contextualized Test MSE: {context_test_mse:.4f}")
        print(f"Contextualized Full dataset MSE: {np.nanmean(mse_full):.4f}")
        
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

        
        print(f"\nPer-cell performance breakdown:")
        print("Cell ID                Train MSE    Test MSE     Train N  Test N")
        print("─" * 60)
        for cell_id in sorted(np.union1d(cell_ids_train, cell_ids_test)):
            tr_mask, te_mask = (cell_ids_train == cell_id), (cell_ids_test == cell_id)
            tr_mse = np.nanmean(mse_train[tr_mask]) if tr_mask.any() else np.nan
            te_mse = np.nanmean(mse_test[te_mask]) if te_mask.any() else np.nan
            print(f'{cell_id:<15}    {tr_mse:8.4f}    {te_mse:8.4f}     {tr_mask.sum():6d}     {te_mask.sum():6d}')

        csv_path = os.path.join(RESULTS_DIR, 'full_dataset_predictions.csv')
        corr_path = os.path.join(RESULTS_DIR, 'full_dataset_correlations.npy')
        betas_path = os.path.join(RESULTS_DIR, 'full_dataset_betas.npy')
        mus_path = os.path.join(RESULTS_DIR, 'full_dataset_mus.npy')

        results_df.to_csv(csv_path, index=False)
        
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
    print(f"Total Samples in Final Dataset:                      {len(df)}")

    print("="*87)


if __name__ == '__main__':
    main()
