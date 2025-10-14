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
torch.set_float32_matmul_precision('medium')
# -----------------------------------------------------------


def main():
    ## Configuration and Data Loading
    pert_to_fit_on = ['trt_cp']
    pert_name = pert_to_fit_on[0]

    EMBEDDINGS_TO_RUN = {
        'AIDOcell': '/home/user/screening2/contextpert/gene_embs/AIDOcell_100M_Norman_Aligned_(D=640)',
        'AIDOdna': '/home/user/screening2/contextpert/gene_embs/AIDOdna_(D=4352)',
        'AIDOprot': '/home/user/screening2/contextpert/gene_embs/AIDOprot_mean_(D=384)',
        'AIDOprot_struct': '/home/user/screening2/contextpert/gene_embs/AIDOprot_seq+struct_(D=1024)'
    }

    PATH_L1000 = f'/home/user/screening2/contextpert/data/{pert_name}.csv'
    PATH_CTLS = '/home/user/contextulized/old/ctrls.csv'
    PERT_INFO = '/home/user/screening2/contextpert/cp_gene/test.csv'

    N_DATA_PCS = 50
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
            # Find which pert_ids correspond to the symbols in this embedding file
            pert_ids_in_emb = set(pert_info_df[pert_info_df['pert_iname'].isin(available_symbols)]['pert_id'])
            
            print(f"  - Found {len(pert_ids_in_emb)} perturbations for '{emb_name}'")
            union_of_pert_ids.update(pert_ids_in_emb)
            
            # Update the intersection
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
            gene_names = pert_data['pert_iname'].unique()
            embeddings_list = []
            
            if len(gene_names) > 0 and gene_names[0] in embs_symbols_set:
                gene_name = gene_names[0]
                embedding = embs[embs.obs.index.get_loc(gene_name)].X
                embeddings_list.append(embedding)
            else:
                not_found_pert_ids.append(pert_id)

            if embeddings_list:
                stacked_embs = np.vstack(embeddings_list)
                pert_embeddings[pert_id] = np.mean(stacked_embs, axis=0)
                
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
    
    # --- This train/test split is now consistent for all embedding runs ---
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

        # --- Create copies to work with in this iteration ---
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

        print("Applying scaling...")
        scaler_genes = StandardScaler()
        X_train_scaled = scaler_genes.fit_transform(X_raw_train)
        X_test_scaled = scaler_genes.transform(X_raw_test)
        
        scaler_embs = StandardScaler()
        gene_embs_train_scaled = scaler_embs.fit_transform(gene_embs_train.astype(float))
        gene_embs_test_scaled = scaler_embs.transform(gene_embs_test.astype(float))

        ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
        unique_cells_all = np.sort(np.union1d(df_train['cell_id'].unique(), df_test['cell_id'].unique()))
        ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_all)]
        
        scaler_ctrls = StandardScaler()
        ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
        n_ctrl_pcs = min(50, ctrls_scaled.shape[0])
        
        pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
        ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
        cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
        if not cell2vec:
            raise ValueError("No common cell IDs found.")
        print(f"Loaded and processed control embeddings for {len(cell2vec)} unique cells.")

        def build_context_matrix(df_split, X_scaled_split, gene_embs_scaled, pert_time, pert_dose, ignore_time, ignore_dose, scaler_context=None, is_train=False):
            cell_ids = df_split['cell_id'].to_numpy()
            continuous_context_parts, X_final_parts, C_final_parts, cell_ids_final_parts = [], [], [], []

            if is_train:
                for cell_id in np.sort(df_split['cell_id'].unique()):
                    if cell_id not in cell2vec: continue
                    mask = (cell_ids == cell_id)
                    continuous_context_parts.append(np.hstack([
                        np.tile(cell2vec[cell_id], (mask.sum(), 1)),
                        pert_time[mask],
                        pert_dose[mask],
                    ]))
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
            
            return np.vstack(X_final_parts), np.vstack(C_final_parts), np.concatenate(cell_ids_final_parts), scaler_context

        print("Building context matrices...")
        X_train, C_train, cell_ids_train, scaler_context = build_context_matrix(df_train, X_train_scaled, gene_embs_train_scaled, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train, is_train=True)
        X_test, C_test, cell_ids_test, _ = build_context_matrix(df_test, X_test_scaled, gene_embs_test_scaled, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test, scaler_context=scaler_context)
        
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
            num_archetypes=30,
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

        trainer = Trainer(
            default_root_dir=RESULTS_DIR, max_epochs=10, accelerator='auto', devices='auto',
            callbacks=[checkpoint_callback],
        )
        trainer.fit(contextualized_model, datamodule=datamodule)

        print(f"Best model path: {checkpoint_callback.best_model_path}")
        print("Testing model on full training data...")
        trainer.test(contextualized_model, datamodule.train_dataloader())
        print("Testing model on test data...")
        trainer.test(contextualized_model, datamodule.test_dataloader())

        output_dir = Path(checkpoint_callback.best_model_path).parent / 'predictions'
        writer_callback = PredictionWriter(output_dir=output_dir, write_interval='batch')

        pred_trainer = Trainer(
            default_root_dir=RESULTS_DIR, accelerator='auto', devices='auto', callbacks=[writer_callback],
        )
        print("Making predictions on full dataset (train + test)...")
        pred_trainer.predict(contextualized_model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
        
        C_train_32 = C_train.astype(np.float32)
        C_test_32 = C_test.astype(np.float32)
        
        all_correlations, all_betas, all_mus = {}, {}, {}
        for file in glob.glob(str(output_dir / 'predictions_*.pt')):
            preds = torch.load(file)
            for context, correlation, beta, mu in zip(preds['contexts'], preds['correlations'], preds['betas'], preds['mus']):
                context_tuple = tuple(context.to(torch.float32).tolist())
                all_correlations[context_tuple] = correlation.cpu().numpy()
                all_betas[context_tuple] = beta.cpu().numpy()
                all_mus[context_tuple] = mu.cpu().numpy()
                
        correlations_train = np.array([all_correlations[c] for c in [tuple(row) for row in C_train_32]])
        correlations_test = np.array([all_correlations[c] for c in [tuple(row) for row in C_test_32]])
        betas_train = np.array([all_betas[c] for c in [tuple(row) for row in C_train_32]])
        betas_test = np.array([all_betas[c] for c in [tuple(row) for row in C_test_32]])
        mus_train = np.array([all_mus[c] for c in [tuple(row) for row in C_train_32]])
        mus_test = np.array([all_mus[c] for c in [tuple(row) for row in C_test_32]])

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
        
        context_train_mse = mse_train.mean()
        context_test_mse = mse_test.mean()

        print(f"Contextualized Train MSE: {context_train_mse:.4f}")
        print(f"Contextualized Test MSE: {context_test_mse:.4f}")
        print(f"Contextualized Full dataset MSE: {mse_full.mean():.4f}")
        
        df_full = pd.concat([df_train, df_test]).reset_index(drop=True)
        results_df = pd.DataFrame({
            'split': ['train'] * len(X_train) + ['test'] * len(X_test),
            'mse': mse_full,
            'cell_id': np.concatenate([cell_ids_train, cell_ids_test]),
            'inst_id': df_full['inst_id'].values,
            'pert_time': df_full['pert_time'].values,
            'pert_dose': df_full['pert_dose'].values,
        })
        
        print(f"\nPer-cell performance breakdown:")
        print("Cell ID               Train MSE    Test MSE     Train N  Test N")
        print("─" * 60)
        for cell_id in sorted(np.union1d(cell_ids_train, cell_ids_test)):
            tr_mask, te_mask = (cell_ids_train == cell_id), (cell_ids_test == cell_id)
            tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
            te_mse = mse_test[te_mask].mean() if te_mask.any() else np.nan
            print(f'{cell_id:<15}   {tr_mse:8.4f}   {te_mse:8.4f}    {tr_mask.sum():6d}    {te_mask.sum():6d}')

        csv_path = os.path.join(RESULTS_DIR, 'full_dataset_predictions.csv')
        corr_path = os.path.join(RESULTS_DIR, 'full_dataset_correlations.npy')
        betas_path = os.path.join(RESULTS_DIR, 'full_dataset_betas.npy')
        mus_path = os.path.join(RESULTS_DIR, 'full_dataset_mus.npy')

        results_df.to_csv(csv_path, index=False)
        np.save(corr_path, correlations_full)
        np.save(betas_path, betas_full)
        np.save(mus_path, mus_full)

        print(f"\nSaved final results to '{RESULTS_DIR}':")
        print(f"  - {csv_path} (sample-level results)")
        print(f"  - {corr_path} (correlation matrices)")
        print(f"  - {betas_path} (beta coefficients)")
        print(f"  - {mus_path} (mu parameters)")
        
        current_run_summary = {
            'Embedding': emb_name,
            'Population Train MSE': pop_train_mse,
            'Population Test MSE': pop_test_mse,
            'Contextualized Train MSE': context_train_mse,
            'Contextualized Test MSE': context_test_mse
        }
        all_run_results.append(current_run_summary)

        print(f"\n--- RESULTS SUMMARY FOR: {emb_name} ---")
        print(f"  Population Model Train MSE:     {pop_train_mse:.4f}")
        print(f"  Population Model Test MSE:      {pop_test_mse:.4f}")
        print(f"  Contextualized Model Train MSE: {context_train_mse:.4f}")
        print(f"  Contextualized Model Test MSE:  {context_test_mse:.4f}")
        print("-------------------------------------------\n")

    if all_run_results:
        print("\n\n" + "="*35 + " FINAL SUMMARY OF ALL RUNS " + "="*35)
        summary_df = pd.DataFrame(all_run_results)
        print(summary_df.to_string())
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
