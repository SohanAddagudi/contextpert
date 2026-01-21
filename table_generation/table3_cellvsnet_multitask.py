import torch
import lightning as pl
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
from pathlib import Path

from contextpert.cellvsnet.multitask.data import MultitaskCorrelationDataModule
from contextpert.cellvsnet.multitask.model import MultitaskContextualizedUnivariateRegression
from contextpert.cellvsnet.multitask.callbacks import MmapEdgeWriter
from contextpert.cellvsnet.multitask.utils import measure_mses, regression_to_correlation

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

RANDOM_STATE = 10

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
seed_everything(RANDOM_STATE, workers=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('highest')

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])

# Paths
PATH_L1000 = DATA_DIR / 'trt_cp_smiles_qc.csv'  # LINCS data with SMILES
PATH_CTLS = DATA_DIR / 'ctrls.csv'              # Control expression for cell context

OUTPUT_DIR = DATA_DIR / 'sm_cohesion_network' / 'trt_cp_multitask'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to {OUTPUT_DIR.resolve()}")

# Model hyperparameters
N_DATA_PCS = 50
N_CTRL_PCS = 50
FEATURE_EMBEDDING_DIM = 32
NUM_ARCHETYPES = 50
ENCODER_HIDDEN_DIMS = [128, 64]
BATCH_SIZE = 64
MAX_EPOCHS = 25
LEARNING_RATE = 1e-3
PERTURBATION_HOLDOUT_SIZE = 0.2
SUBSAMPLE_FRACTION = None

class SubtypeMLPEncoder(torch.nn.Module):
    def __init__(self, context_dim, feature_embedding_dim, num_archetypes=10, 
                 encoder_hidden_dims=[64, 32], activation=torch.nn.ReLU):
        super().__init__()
        
        # Encoder: maps context + task embeddings to archetype weights
        input_dim = context_dim + feature_embedding_dim + feature_embedding_dim
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, num_archetypes))
        layers.append(torch.nn.Softmax(dim=-1))
        self.context_encoder = torch.nn.Sequential(*layers)
        
        # Archetypes: learned prototypes for (beta, mu)
        self.archetypes = torch.nn.Parameter(torch.randn(num_archetypes, 2) * 0.01)

    def forward(self, contexts, predictor_embeddings, outcome_embeddings):
        x = torch.cat([contexts, predictor_embeddings, outcome_embeddings], dim=-1)
        weights = self.context_encoder(x)  # (batch, num_archetypes)
        outputs = weights @ self.archetypes  # (batch, 2)
        return {
            'betas': outputs[:, 0],
            'mus': outputs[:, 1],
        }


print("=" * 80)
print("MULTITASK NETWORK TRAINING")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Data PCs (network size): {N_DATA_PCS}")
print(f"  Feature embedding dim: {FEATURE_EMBEDDING_DIM}")
print(f"  Hidden dims: {ENCODER_HIDDEN_DIMS}")
print(f"  Num archetypes: {NUM_ARCHETYPES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {MAX_EPOCHS}")

# Load LINCS data
print(f"\nLoading LINCS data from: {PATH_L1000}")
df = pd.read_csv(PATH_L1000, engine='pyarrow')

# Filter perturbation type
pert_to_fit_on = ['trt_cp']
df = df[df['pert_type'].isin(pert_to_fit_on)]

# Quality filters (same as existing scripts)
bad = (
    (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
)
df = df[~bad]

# Filter valid SMILES
df = df.dropna(subset=['canonical_smiles'])
df = df[df['canonical_smiles'] != '']
df = df[~df['canonical_smiles'].isin(['-666', 'restricted'])]

print(f"Processing {len(df)} samples after quality filtering...")

if SUBSAMPLE_FRACTION is not None:
    df = df.sample(frac=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE)
    print(f"Subsampled to {len(df)} samples ({SUBSAMPLE_FRACTION*100}% of data)")

# Create perturbation-based train/test split
unique_pert_ids = df['pert_id'].unique()
print(f"Found {len(unique_pert_ids)} unique perturbations (pert_id)")

pert_ids_train, pert_ids_test = train_test_split(
    unique_pert_ids,
    test_size=PERTURBATION_HOLDOUT_SIZE,
    random_state=RANDOM_STATE
)
print(f"Perturbation split: {len(pert_ids_train)} train, {len(pert_ids_test)} test perturbations")

df_train = df[df['pert_id'].isin(pert_ids_train)].copy()
df_test = df[df['pert_id'].isin(pert_ids_test)].copy()
print(f"Sample split: {len(df_train)} train, {len(df_test)} test samples")

# Handle missing values for pert_time and pert_dose
pert_time_mean = None
pert_dose_mean = None

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

def process_data_split(df_split, split_name):
    """Process a data split to extract features and metadata."""
    numeric_cols = df_split.select_dtypes(include=[np.number]).columns
    drop_cols = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    X_raw = df_split[feature_cols].values
    
    pert_time = df_split['pert_time'].to_numpy().reshape(-1, 1)
    pert_dose = df_split['pert_dose'].to_numpy().reshape(-1, 1)
    ignore_time = df_split['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
    ignore_dose = df_split['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)
    
    return X_raw, pert_time, pert_dose, ignore_time, ignore_dose

X_raw_train, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train = process_data_split(df_train, 'train')
X_raw_test, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test = process_data_split(df_test, 'test')

# Scale gene expression
print("\nApplying scaling...")
scaler_genes = StandardScaler()
X_train_scaled = scaler_genes.fit_transform(X_raw_train)
X_test_scaled = scaler_genes.transform(X_raw_test)
print(f"Gene expression scaled: train {X_train_scaled.shape}, test {X_test_scaled.shape}")

print("\nLoading cell context from control expression...")
ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
unique_cells_train = np.sort(df_train['cell_id'].unique())
unique_cells_test = np.sort(df_test['cell_id'].unique())
unique_cells_all = np.sort(np.union1d(unique_cells_train, unique_cells_test))

ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_all)]

scaler_ctrls = StandardScaler()
ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

n_cells = ctrls_scaled.shape[0]
n_ctrl_pcs = min(N_CTRL_PCS, n_cells)

pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)

cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))

if not cell2vec:
    raise ValueError("No common cell IDs found. Cannot proceed.")

print(f"Loaded and processed control embeddings for {len(cell2vec)} unique cells.")

def build_context_matrix(df_split, X_scaled_split, pert_time, pert_dose, 
                         ignore_time, ignore_dose, split_name, scaler_context=None, is_train=False):
    """Build context matrix from cell embeddings and perturbation info."""
    cell_ids = df_split['cell_id'].to_numpy()
    unique_cells_split = np.sort(df_split['cell_id'].unique())
    
    all_continuous_context = []
    valid_cells = []
    
    for cell_id in unique_cells_split:
        if cell_id not in cell2vec:
            print(f"Warning: Cell {cell_id} not found in control embeddings, skipping...")
            continue
            
        mask = cell_ids == cell_id
        if mask.sum() == 0:
            continue
            
        valid_cells.append(cell_id)
        
        C_continuous = np.hstack([
            np.tile(cell2vec[cell_id], (mask.sum(), 1)),
            pert_time[mask],
            pert_dose[mask],
        ])
        all_continuous_context.append(C_continuous)
    
    if is_train:
        all_continuous_combined = np.vstack(all_continuous_context)
        scaler_context = StandardScaler()
        scaler_context.fit(all_continuous_combined)
        print(f"Fitted context scaler on {all_continuous_combined.shape} continuous context features")
    
    if scaler_context is None:
        raise ValueError("scaler_context must be provided for non-training data")
    
    X_lst, C_lst, cell_lst, inst_ids_lst = [], [], [], []
    
    for i, cell_id in enumerate(valid_cells):
        mask = cell_ids == cell_id
        X_cell = X_scaled_split[mask]
        
        C_continuous_scaled = scaler_context.transform(all_continuous_context[i])
        
        C_cell = np.hstack([
            C_continuous_scaled,
            ignore_time[mask],
            ignore_dose[mask],
        ])

        X_lst.append(X_cell)
        C_lst.append(C_cell)
        cell_lst.append(cell_ids[mask])
        inst_ids_lst.append(df_split.loc[mask, 'inst_id'].values)

    if not X_lst:
        raise RuntimeError(f"No data collected for {split_name} set.")
    
    X_final = np.vstack(X_lst)
    C_final = np.vstack(C_lst)
    cell_ids_final = np.concatenate(cell_lst)
    inst_ids_final = np.concatenate(inst_ids_lst)
    
    return X_final, C_final, cell_ids_final, inst_ids_final, scaler_context

print("\nBuilding context matrices...")

X_train, C_train, cell_ids_train, inst_ids_train, scaler_context = build_context_matrix(
    df_train, X_train_scaled, pert_time_train, pert_dose_train,
    ignore_time_train, ignore_dose_train, 'train', is_train=True
)

X_test, C_test, cell_ids_test, inst_ids_test, _ = build_context_matrix(
    df_test, X_test_scaled, pert_time_test, pert_dose_test,
    ignore_time_test, ignore_dose_test, 'test', scaler_context=scaler_context
)

print(f'Context matrix:   train {C_train.shape}   test {C_test.shape}')
print(f'Feature matrix:   train {X_train.shape}   test {X_test.shape}')

print("\nApplying PCA to gene expression...")

pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
X_train_pca = pca_data.fit_transform(X_train)
X_test_pca = pca_data.transform(X_test)

pca_scaler = StandardScaler()
X_train_norm = pca_scaler.fit_transform(X_train_pca).astype(np.float32)
X_test_norm = pca_scaler.transform(X_test_pca).astype(np.float32)

print(f'Normalized PCA features: train {X_train_norm.shape}   test {X_test_norm.shape}')

# Convert context to float32
C_train = C_train.astype(np.float32)
C_test = C_test.astype(np.float32)


print("\nCreating gene feature embeddings...")

# Initialize learnable feature embeddings for genes (nodes in the network)
feature_embeddings = np.random.randn(N_DATA_PCS, FEATURE_EMBEDDING_DIM).astype(np.float32)
feature_embeddings = feature_embeddings / np.linalg.norm(feature_embeddings, axis=1, keepdims=True)
print(f"Feature embedding shape: {feature_embeddings.shape}")

print("\nPreparing data module...")

# Split training data into train/val
train_indices, val_indices = train_test_split(
    np.arange(len(X_train_norm)), test_size=0.2, random_state=RANDOM_STATE
)

C_val = C_train[val_indices]
X_val = X_train_norm[val_indices]
C_train_split = C_train[train_indices]
X_train_split = X_train_norm[train_indices]

# Full dataset for prediction
X_full = np.concatenate([X_train_norm, X_test_norm], axis=0)
C_full = np.concatenate([C_train, C_test], axis=0)
inst_ids_full = np.concatenate([inst_ids_train, inst_ids_test])
cell_ids_full = np.concatenate([cell_ids_train, cell_ids_test])

datamodule = MultitaskCorrelationDataModule(
    C_train=C_train_split,
    X_train=X_train_split,
    train_feature_embeddings=feature_embeddings,
    C_val=C_val,
    X_val=X_val,
    val_feature_embeddings=feature_embeddings,
    C_test=C_test,
    X_test=X_test_norm,
    test_feature_embeddings=feature_embeddings,
    C_predict=C_full,
    X_predict=X_full,
    predict_feature_embeddings=feature_embeddings,
    batch_size=BATCH_SIZE,
)

print("\nSetting up model...")

context_dim = C_train.shape[1]
encoder = SubtypeMLPEncoder(
    context_dim=context_dim,
    feature_embedding_dim=FEATURE_EMBEDDING_DIM,
    num_archetypes=NUM_ARCHETYPES,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
)

model = MultitaskContextualizedUnivariateRegression(
    encoder=encoder,
    learning_rate=LEARNING_RATE,
)

print(f"  Context dim: {context_dim}")
print(f"  Feature embedding dim: {FEATURE_EMBEDDING_DIM}")
print(f"  Network size: {N_DATA_PCS}x{N_DATA_PCS}")
print(f"  Num archetypes: {NUM_ARCHETYPES}")
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR / 'checkpoints',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    filename='best_model',
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
)

trainer = Trainer(
    default_root_dir=str(OUTPUT_DIR),
    max_epochs=MAX_EPOCHS,
    accelerator='auto',
    devices='auto',
    callbacks=[checkpoint_callback, early_stop_callback],
    deterministic=True,
)

print("\nStarting training...")
trainer.fit(model, datamodule)

print(f"\nBest model path: {checkpoint_callback.best_model_path}")

print("\nTesting on train data...")
trainer.test(model, datamodule.train_dataloader())
print("Testing on test data...")
trainer.test(model, datamodule.test_dataloader())

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS ON FULL DATASET")
print("=" * 80)

# Load best model
best_model_path = checkpoint_callback.best_model_path
print(f"Loading best model from: {best_model_path}")

# Recreate encoder for loading
encoder_for_load = SubtypeMLPEncoder(
    context_dim=context_dim,
    feature_embedding_dim=FEATURE_EMBEDDING_DIM,
    num_archetypes=NUM_ARCHETYPES,
    encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
)
model = MultitaskContextualizedUnivariateRegression.load_from_checkpoint(
    best_model_path,
    encoder=encoder_for_load,
)

# Set up prediction writer
pred_output_dir = OUTPUT_DIR / 'predictions'
pred_output_dir.mkdir(parents=True, exist_ok=True)

writer_callback = MmapEdgeWriter(
    mmap_dir=str(pred_output_dir),
    n_samples=len(X_full),
    x_dim=N_DATA_PCS,
    y_dim=N_DATA_PCS,
    dtype=np.float32,
    write_interval='batch',
)

pred_trainer = Trainer(
    accelerator='auto',
    devices='auto',
    callbacks=[writer_callback],
    deterministic=True,
)

print("Running predictions on full dataset...")
_ = pred_trainer.predict(model, datamodule.predict_dataloader())

print("\n" + "=" * 80)
print("PROCESSING PREDICTIONS")
print("=" * 80)

# Load predictions from memory map
mmap_path = pred_output_dir / "edges_rank0.dat"
print(f"Loading predictions from: {mmap_path}")

preds = np.memmap(
    mmap_path,
    dtype=np.float32,
    mode="r",
    shape=(len(X_full), N_DATA_PCS, N_DATA_PCS, 2)
)

betas_full = np.array(preds[:, :, :, 0])
mus_full = np.array(preds[:, :, :, 1])

print(f"  Betas shape: {betas_full.shape}")
print(f"  Mus shape: {mus_full.shape}")

n_train = len(X_train_norm)
betas_train = betas_full[:n_train]
betas_test = betas_full[n_train:]
mus_train = mus_full[:n_train]
mus_test = mus_full[n_train:]

print("Converting to correlations...")
correlations_full = regression_to_correlation(betas_full)
print(f"  Correlations shape: {correlations_full.shape}")

print("\nComputing MSE...")
mse_train = measure_mses(betas_train, mus_train, X_train_norm)
mse_test = measure_mses(betas_test, mus_test, X_test_norm)
mse_full = measure_mses(betas_full, mus_full, X_full)

print(f"Train MSE: {np.mean(mse_train):.4f}")
print(f"Test MSE: {np.mean(mse_test):.4f}")
print(f"Full dataset MSE: {np.mean(mse_full):.4f}")

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Create results dataframe
results_df = pd.DataFrame({
    'split': ['train'] * n_train + ['test'] * len(X_test_norm),
    'inst_id': inst_ids_full,
    'cell_id': cell_ids_full,
    'mse': mse_full,
})

# Merge with original df to get pert_id
df_full_orig = pd.concat([df_train.reset_index(drop=True), df_test.reset_index(drop=True)], ignore_index=True)

# Save files
results_df.to_csv(OUTPUT_DIR / 'full_dataset_predictions.csv', index=False)
np.save(OUTPUT_DIR / 'full_dataset_betas.npy', betas_full)
np.save(OUTPUT_DIR / 'full_dataset_correlations.npy', correlations_full)
np.save(OUTPUT_DIR / 'full_dataset_mus.npy', mus_full)

print("\n" + "=" * 80)
print("PER-CELL PERFORMANCE BREAKDOWN")
print("=" * 80)
print("Cell ID           Train MSE    Test MSE      Train N  Test N")
print("─" * 60)

all_unique_cells = np.union1d(cell_ids_train, cell_ids_test)

for cell_id in sorted(all_unique_cells):
    tr_mask = cell_ids_train == cell_id
    te_mask = cell_ids_test == cell_id
    
    tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
    te_mse = mse_test[te_mask].mean() if te_mask.any() else np.nan
    tr_n = tr_mask.sum()
    te_n = te_mask.sum()
    
    if tr_n > 0 or te_n > 0:
        print(f'{cell_id:<15}  {tr_mse:8.4f}    {te_mse:8.4f}    {tr_n:6d}    {te_n:6d}')

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel: MultitaskContextualizedUnivariateRegression")
print(f"Encoder: SubtypeMLPEncoder (num_archetypes={NUM_ARCHETYPES})")
print(f"Network size: {N_DATA_PCS} x {N_DATA_PCS}")
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"\nTo evaluate on the drug-disease cohesion benchmark, run:")
print(f"  python example_submissions/sm_cohesion_multitask_networks_submission.py")
