import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import torch
import lightning as pl
from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import shutil

seed_everything(10, workers=True)

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])

PATH_FULL_DATA = DATA_DIR / 'lincs.csv'

"""
Possible MODEL_MODE options:
1. 'population': A single global model shared across all contexts (ignores context features).
2. 'contextualized': Model parameters are modulated dynamically based on the context vector (C).
"""
MODEL_MODE = 'population' 
CELL_CONTEXT_MODE = 'onehot' 
USE_FULL_CONTEXT_FEATURES = False 

N_DATA_PCS = 50
RANDOM_STATE = 42
TEST_SIZE = 0.33
BATCH_SIZE = 32

seed_everything(RANDOM_STATE, workers=True)

print(f"--- Running Table 1 Reconstruction for Controls ---")
print(f"MODEL_MODE: {MODEL_MODE}")
print(f"Targeting Controls: ['ctl_vehicle', 'ctl_vector', 'ctl_untrt']")
print("---------------------------------------------------\n")

if not os.path.exists(PATH_FULL_DATA):
    raise FileNotFoundError(f"Data file not found at {PATH_FULL_DATA}")

print("Loading data...")
df = pd.read_csv(PATH_FULL_DATA, engine='pyarrow')

controls_to_fit = ['ctl_vehicle', 'ctl_vector', 'ctl_untrt']
mask = df['pert_type'].isin(controls_to_fit)
df = df[mask].reset_index(drop=True)

if 'distil_cc_q75' in df.columns and 'pct_self_rank_q25' in df.columns:
    condition = (
        (df['distil_cc_q75'] < 0.2) |
        (df['pct_self_rank_q25'] > 5)
    )
    df = df[~condition].reset_index(drop=True)

cell_counts = df['cell_id'].value_counts()
df['n_c'] = df['cell_id'].map(cell_counts)

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
columns_to_drop = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25', 'n_c']
feature_cols = [col for col in feature_cols if col not in columns_to_drop and col in df.columns]
X = df[feature_cols].values

cell_ids = df['cell_id'].values
unique_cells = np.unique(cell_ids)

print("Using 'onehot' context...")
cell_dummies = pd.get_dummies(unique_cells, prefix='cell', dtype=int)
cell2vec = dict(zip(unique_cells, cell_dummies.values))

cell_context_matrix = np.array([cell2vec[cid] for cid in cell_ids])
pert_dummies = pd.get_dummies(df['pert_id'], drop_first=True).values
C_global = np.hstack([pert_dummies, cell_context_matrix]).astype(np.float32)

X_train_list, X_test_list = [], []
C_train_list, C_test_list = [], []
nc_test_list = [] 

print("Splitting data...")
for cell in tqdm(unique_cells, desc="Splitting by cell line"):
    cell_mask = (cell_ids == cell)
    X_cell = X[cell_mask]
    n_samples = len(X_cell)
    current_nc = df.loc[cell_mask, 'n_c'].iloc[0]
    
    if n_samples < 2:
        continue

    indices = np.arange(n_samples)
    if n_samples == 2:
        train_idx, test_idx = [0], [1]
    elif n_samples == 3:
        train_idx, test_idx = [0, 1], [2]
    else:
        train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    X_train_list.append(X_cell[train_idx])
    X_test_list.append(X_cell[test_idx])
    nc_test_list.append(np.full(len(test_idx), current_nc))
    
    C_cell = C_global[cell_mask]
    C_train_list.append(C_cell[train_idx])
    C_test_list.append(C_cell[test_idx])

X_train = np.vstack(X_train_list)
X_test = np.vstack(X_test_list)
nc_test = np.concatenate(nc_test_list)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=min(N_DATA_PCS, X_train_scaled.shape[1]))
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_mean = X_train_pca.mean(axis=0)
X_std = X_train_pca.std(axis=0)
X_std = np.where(X_std == 0, 1e-6, X_std)
X_train_norm = (X_train_pca - X_mean) / X_std
X_test_norm = (X_test_pca - X_mean) / X_std

C_train = np.vstack(C_train_list).astype(np.float32)
C_test = np.vstack(C_test_list).astype(np.float32)

print("\n--- Running Model: Contextualized (Lightning) ---")

contextualized_model = ContextualizedCorrelation(
    context_dim=C_train.shape[1],
    x_dim=X_train_norm.shape[1],
    encoder_type='mlp',
    num_archetypes=16,
)

train_idx, val_idx = train_test_split(np.arange(len(X_train_norm)), test_size=0.1, random_state=RANDOM_STATE)

datamodule = CorrelationDataModule(
    C_train=C_train[train_idx], X_train=X_train_norm[train_idx],
    C_val=C_train[val_idx],     X_val=X_train_norm[val_idx],    
    C_test=C_test,              X_test=X_test_norm,             
    C_predict=C_test,           X_predict=X_test_norm, 
    batch_size=BATCH_SIZE,
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', mode='min', save_top_k=1, filename='best_model'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss', 
    patience=1,          
    mode='min'           
)

trainer = Trainer(
    max_epochs=15, 
    accelerator='auto', devices='auto',
    callbacks=[checkpoint_callback, early_stop_callback],
    enable_progress_bar=True
)

trainer.fit(contextualized_model, datamodule=datamodule)

print(f"Testing model on training data...")
trainer.test(contextualized_model, datamodule.train_dataloader())

print(f"Testing model on test data...")
trainer.test(contextualized_model, datamodule.test_dataloader())

print("\n" + "="*50)
print("RUNNING TABLE 1 GENERATION")
print("="*50)

def run_subset_test(name, X_subset, C_subset):
    if len(X_subset) == 0:
        return np.nan
    
    datamodule.X_test = X_subset
    datamodule.C_test = C_subset
    
    results = trainer.test(contextualized_model, datamodule=datamodule, verbose=False)
    
    score = results[0]['test_loss']
    print(f">> {name}: {score:.4f}")
    return score

score_train = run_subset_test("Train Data (Full)", X_train_norm, C_train)

score_test_full = run_subset_test("Test Data (Full)", X_test_norm, C_test)

mask_high = nc_test > 3
score_test_high = run_subset_test("Test Data (nc > 3)", 
                                  X_test_norm[mask_high], 
                                  C_test[mask_high])

mask_low = nc_test <= 3
score_test_low = run_subset_test("Test Data (nc <= 3)", 
                                 X_test_norm[mask_low], 
                                 C_test[mask_low])

print("\n" + "="*45)
print(f"TABLE 1 RECREATION (MSE)")
print("-" * 45)
print(f"{'Condition':<20} | {'MSE':<20}")
print("-" * 45)
print(f"{'Train (Full)':<20} | {score_train:.4f}")
print(f"{'Test (Full)':<20} | {score_test_full:.4f}")
print("-" * 45)
print(f"{'Test (nc > 3)':<20} | {score_test_high:.4f}")
print(f"{'Test (nc <= 3)':<20} | {score_test_low:.4f}")
print("="*45 + "\n")
