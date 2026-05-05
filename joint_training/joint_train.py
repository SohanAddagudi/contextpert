"""
Jointly train a single ContextualizedCorrelation network across all four
perturbation types (trt_cp, trt_sh, trt_oe, trt_lig) 
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

DATA_DIR = Path(os.environ['CONTEXTPERT_DATA_DIR'])

PATH_CTLS      = DATA_DIR / 'ctrls.csv'
PATH_PERT_INFO = DATA_DIR / 'gene_embeddings' / 'perts_targets.csv'
SPLIT_DIR      = DATA_DIR / 'gene_embeddings' / 'unseen_perturbation_splits'

EXPR_CSV = {
    'trt_cp':  DATA_DIR / 'trt_cp_smiles_qc.csv',
    'trt_sh':  DATA_DIR / 'trt_sh_genes_qc.csv',
    'trt_oe':  DATA_DIR / 'pert_type_csvs' / 'trt_oe.csv',
    'trt_lig': DATA_DIR / 'pert_type_csvs' / 'trt_lig.csv',
}

PERT_EMB_PATH = {
    'trt_cp':  DATA_DIR / 'gene_embeddings' / 'chemberta_embeddings.npz',
    'trt_sh':  DATA_DIR / 'gene_embeddings' / 'AIDOprot_mean_(D=384)',
    'trt_oe':  DATA_DIR / 'gene_embeddings' / 'AIDOprot_mean_(D=384)',
    'trt_lig': DATA_DIR / 'gene_embeddings' / 'AIDOprot_seq+struct_(D=1024)',
}
SPLIT_MAP = {pt: SPLIT_DIR / f'{pt}_split_map.csv' for pt in EXPR_CSV}

SAVE_TYPES = ('trt_cp', 'trt_sh')
ALL_TYPES = ('trt_cp', 'trt_sh', 'trt_oe', 'trt_lig')
GENE_TARGET_TYPES = ('trt_sh', 'trt_oe', 'trt_lig')

N_DATA_PCS         = 50
N_PERT_EMB_PCS     = 16
NUM_ARCHETYPES     = 100
MAX_EPOCHS         = 7
BATCH_SIZE         = 64
VAL_FRAC           = 0.1
BALANCE_PER_TYPE   = 30_000
LEARNING_RATE      = 1e-3
ENCODER_KWARGS     = {'width': 128, 'layers': 2, 'link_fn': 'identity'}

OUTPUT_DIR = Path(__file__).parent / 'best_joint_model_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Outputs will be saved to: {OUTPUT_DIR.resolve()}')


# =============================================================================
# STEP 1: Load per-type expression data + pert embeddings
# =============================================================================

def qc_filter(df):
    bad = (
        (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
        (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
    )
    return df[~bad].copy()


def fill_pert_time_dose(df_train, df_test):
    for d in (df_train, df_test):
        d['ignore_flag_pert_time'] = (d['pert_time'] == -666).astype(int)
        d['ignore_flag_pert_dose'] = (d['pert_dose'] == -666).astype(int)
    means = {}
    for col in ('pert_time', 'pert_dose'):
        m = df_train.loc[df_train[col] != -666, col].mean()
        means[col] = m
        df_train[col] = df_train[col].replace(-666, m)
        df_test[col]  = df_test[col].replace(-666, m)
    return means


def load_gene_embedding(emb_path):
    embs = ad.read_h5ad(emb_path)
    embs.obs = embs.obs.set_index('symbol')
    X = embs.X.toarray() if hasattr(embs.X, 'toarray') else np.asarray(embs.X)
    sym2vec = {sym: X[i] for i, sym in enumerate(embs.obs.index)}
    return sym2vec, X.shape[1]


def get_pert_embeddings_gene(unique_pert_ids, sym2vec, pert_info_df):
    mapping = pert_info_df.dropna(subset=['pert_iname']).copy()
    mapping['gene'] = mapping['pert_iname'].str.split(';')
    mapping = mapping.explode('gene')
    mapping['gene'] = mapping['gene'].str.strip()
    mapping = mapping[mapping['gene'] != '']
    mapping = mapping[mapping['pert_id'].isin(unique_pert_ids)]
    mapping = mapping[mapping['gene'].isin(sym2vec)]
    pert_embs = {}
    if not mapping.empty:
        vecs = np.vstack(mapping['gene'].map(sym2vec).values)
        mapping = mapping.copy()
        mapping['_idx'] = np.arange(len(mapping))
        for pid, idxs in mapping.groupby('pert_id')['_idx']:
            pert_embs[pid] = vecs[list(idxs)].mean(axis=0)
    missing = [p for p in unique_pert_ids if p not in pert_embs]
    if missing:
        warnings.warn(f'{len(missing)} pert_ids have no gene embedding (zero-padded).')
    return pert_embs


def load_chemberta_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return dict(zip(data['inst_ids'], data['embeddings']))


print('=' * 78)
print('STEP 1: Loading per-type expression data + pert embeddings')
print('=' * 78)

pert_info_df = pd.read_csv(PATH_PERT_INFO)


def perts_with_target_in_embedding(emb_path, pert_info):
    embs = ad.read_h5ad(emb_path)
    available = set(embs.obs.set_index('symbol').index)
    pinfo = pert_info.dropna(subset=['pert_iname']).copy()
    pinfo['_t'] = pinfo['pert_iname'].str.split(';').apply(lambda gs: {g.strip() for g in gs})
    return set(pinfo.loc[pinfo['_t'].apply(lambda t: not t.isdisjoint(available)), 'pert_id'])


loaded_chem_embs = {}
loaded_gene_embs = {}
for pt in ALL_TYPES:
    p = PERT_EMB_PATH[pt]
    if pt == 'trt_cp':
        if p not in loaded_chem_embs:
            iid2vec = load_chemberta_embeddings(p)
            sample_vec = next(iter(iid2vec.values()))
            dim = int(np.asarray(sample_vec).shape[0])
            loaded_chem_embs[p] = (iid2vec, dim)
            print(f'  loaded ChemBERTa: {p.name}  n_inst={len(iid2vec):,}  dim={dim}')
    else:
        if p not in loaded_gene_embs:
            loaded_gene_embs[p] = load_gene_embedding(p)
            sym2vec, dim = loaded_gene_embs[p]
            print(f'  loaded gene embeddings: {p.name}  n_genes={len(sym2vec):,}  dim={dim}')


per_type_pert_filter = {}
for pt in GENE_TARGET_TYPES:
    per_type_pert_filter[pt] = perts_with_target_in_embedding(PERT_EMB_PATH[pt], pert_info_df)
    print(f'  {pt}: {len(per_type_pert_filter[pt]):,} perts whose targets are in {PERT_EMB_PATH[pt].name}')


per_type_data = {}

for pt in ALL_TYPES:
    print(f'\n--- {pt} ---')
    df = pd.read_csv(EXPR_CSV[pt], engine='pyarrow')
    df = df[df['pert_type'] == pt]
    df = qc_filter(df)
    df = df.dropna(subset=['pert_id'])
    df = df[df['pert_id'] != '']

    if pt == 'trt_cp':
        df = df.dropna(subset=['canonical_smiles'])
        df = df[df['canonical_smiles'] != '']
        iid2vec, emb_dim = loaded_chem_embs[PERT_EMB_PATH[pt]]
        before = len(df)
        df = df[df['inst_id'].isin(iid2vec)]
        print(f'  rows with ChemBERTa embedding: {len(df):,}/{before:,}')
    else:
        df = df[df['pert_id'].isin(per_type_pert_filter[pt])]

    sm = pd.read_csv(SPLIT_MAP[pt])[['inst_id', 'split']]
    df = df.merge(sm, on='inst_id', how='inner')

    df_train = df[df['split'] == 'train'].drop(columns='split').copy()
    df_test  = df[df['split'] == 'test' ].drop(columns='split').copy()
    print(f'  rows: train={len(df_train):,}  test={len(df_test):,}')

    fill_pert_time_dose(df_train, df_test)

    if pt == 'trt_cp':
        iid2vec, emb_dim = loaded_chem_embs[PERT_EMB_PATH[pt]]
        emb_train = np.array([iid2vec[iid] for iid in df_train['inst_id']], dtype=np.float32)
        emb_test  = np.array([iid2vec[iid] for iid in df_test['inst_id']],  dtype=np.float32)
    else:
        sym2vec, emb_dim = loaded_gene_embs[PERT_EMB_PATH[pt]]
        all_pids = pd.concat([df_train['pert_id'], df_test['pert_id']]).unique()
        pid2vec = get_pert_embeddings_gene(all_pids, sym2vec, pert_info_df)
        zero = np.zeros(emb_dim, dtype=np.float32)
        emb_train = np.array([pid2vec.get(p, zero) for p in df_train['pert_id']], dtype=np.float32)
        emb_test  = np.array([pid2vec.get(p, zero) for p in df_test['pert_id']],  dtype=np.float32)

    print(f'  pert emb dim raw = {emb_dim}')

    per_type_data[pt] = dict(
        df_train=df_train, df_test=df_test,
        emb_train=emb_train, emb_test=emb_test,
    )


# =============================================================================
# STEP 2: StandardScaler + PCA on pert embeddings
# =============================================================================

print('\n' + '=' * 78)
print(f'STEP 2: StandardScaler+PCA on pert embeddings -> {N_PERT_EMB_PCS} dims')
print('=' * 78)

PE_FIT_N = 20_000
rng_pe = np.random.default_rng(RANDOM_STATE)
groups = {}
for pt in ALL_TYPES:
    groups.setdefault(PERT_EMB_PATH[pt], []).append(pt)

for emb_path, pt_group in groups.items():
    src_name = emb_path.name
    print(f'\n  Group [{", ".join(pt_group)}] sharing {src_name}:')
    fit_blocks = []
    for pt in pt_group:
        e_tr = per_type_data[pt]['emb_train'].astype(float)
        n = min(PE_FIT_N, len(e_tr))
        idx = rng_pe.choice(len(e_tr), size=n, replace=False)
        fit_blocks.append(e_tr[idx])
    fit_set = np.vstack(fit_blocks)
    print(f'    balanced fit set: {fit_set.shape}')

    scaler = StandardScaler().fit(fit_set)
    fit_scaled = scaler.transform(fit_set).astype(np.float32)
    del fit_set, fit_blocks
    n_comp = min(N_PERT_EMB_PCS, fit_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(fit_scaled)
    del fit_scaled
    print(f'    fitted scaler + PCA(n_components={n_comp})')

    for pt in pt_group:
        d = per_type_data[pt]
        et = pca.transform(scaler.transform(d['emb_train'].astype(float))).astype(np.float32)
        es = pca.transform(scaler.transform(d['emb_test'].astype(float))).astype(np.float32)
        if et.shape[1] < N_PERT_EMB_PCS:
            pad_tr = np.zeros((et.shape[0], N_PERT_EMB_PCS - et.shape[1]), dtype=et.dtype)
            pad_te = np.zeros((es.shape[0], N_PERT_EMB_PCS - es.shape[1]), dtype=es.dtype)
            et = np.hstack([et, pad_tr])
            es = np.hstack([es, pad_te])
        d['emb_train_pca'] = et
        d['emb_test_pca']  = es
        print(f'    {pt}: train {et.shape}  test {es.shape}')


# =============================================================================
# STEP 3: Build expression feature matrix
# =============================================================================

print('\n' + '=' * 78)
print(f'STEP 3: Build expression feature matrix (-> StandardScaler -> PCA({N_DATA_PCS}) -> StandardScaler)')
print('=' * 78)

DROP = {'pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25',
        'ignore_flag_pert_time', 'ignore_flag_pert_dose'}

def numeric_feature_cols(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in DROP]

shared_cols = None
for pt in ALL_TYPES:
    cols = set(numeric_feature_cols(per_type_data[pt]['df_train']))
    shared_cols = cols if shared_cols is None else (shared_cols & cols)
shared_cols = sorted(shared_cols)
print(f'  Shared expression feature columns: {len(shared_cols)}')

rng_bal = np.random.default_rng(RANDOM_STATE)
SCALER_FIT_N = 20_000
balanced_fit_blocks = []
for pt in ALL_TYPES:
    d = per_type_data[pt]
    X_tr = d['df_train'][shared_cols].to_numpy(dtype=np.float32)
    n = min(SCALER_FIT_N, len(X_tr))
    idx = rng_bal.choice(len(X_tr), size=n, replace=False)
    balanced_fit_blocks.append(X_tr[idx])
X_balanced_fit = np.vstack(balanced_fit_blocks)
print(f'  Balanced fit set for scaler+PCA: {X_balanced_fit.shape}  (~{SCALER_FIT_N:,}/type)')

scaler_genes = StandardScaler().fit(X_balanced_fit)

X_train_blocks, X_test_blocks_raw = [], []
for pt in ALL_TYPES:
    d = per_type_data[pt]
    X_train_blocks.append(d['df_train'][shared_cols].to_numpy(dtype=np.float32))
    X_test_blocks_raw.append(d['df_test'][shared_cols].to_numpy(dtype=np.float32))
X_train_scaled = scaler_genes.transform(np.vstack(X_train_blocks)).astype(np.float32)
X_test_scaled  = scaler_genes.transform(np.vstack(X_test_blocks_raw)).astype(np.float32)
del X_train_blocks, X_test_blocks_raw, balanced_fit_blocks

X_balanced_scaled = scaler_genes.transform(X_balanced_fit).astype(np.float32)
del X_balanced_fit
pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE).fit(X_balanced_scaled)
del X_balanced_scaled

X_train_pca = pca_data.transform(X_train_scaled)
X_test_pca  = pca_data.transform(X_test_scaled)
del X_train_scaled, X_test_scaled

pca_scaler = StandardScaler()
X_train = pca_scaler.fit_transform(X_train_pca).astype(np.float32)
X_test  = pca_scaler.transform(X_test_pca).astype(np.float32)
del X_train_pca, X_test_pca
print(f'  Final expression matrix: train {X_train.shape}  test {X_test.shape}')


# =============================================================================
# STEP 4: Control-cell PCA
# =============================================================================

print('\n' + '=' * 78)
print('STEP 4: Control-cell embeddings via PCA(50) on ctrls.csv')
print('=' * 78)

ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
all_cells = set()
for pt in ALL_TYPES:
    all_cells |= set(per_type_data[pt]['df_train']['cell_id'].unique())
    all_cells |= set(per_type_data[pt]['df_test']['cell_id'].unique())
ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(sorted(all_cells))]

scaler_ctrls = StandardScaler()
ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
n_ctrl_pcs = min(50, ctrls_scaled.shape[0])
pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
print(f'  cell2vec built for {len(cell2vec)} cells (PCs={n_ctrl_pcs})')


# =============================================================================
# STEP 5: Assemble row-aligned context block
# =============================================================================

print('\n' + '=' * 78)
print('STEP 5: Assembling context matrix')
print('=' * 78)

PT_INDEX = {pt: i for i, pt in enumerate(ALL_TYPES)}

keep_per_type = {}
for pt in ALL_TYPES:
    d = per_type_data[pt]
    keep_train = np.array([c in cell2vec for c in d['df_train']['cell_id']])
    keep_test  = np.array([c in cell2vec for c in d['df_test']['cell_id']])
    keep_per_type[pt] = (keep_train, keep_test)
    print(f'  {pt}: kept train {keep_train.sum()}/{len(keep_train)}  '
          f'test {keep_test.sum()}/{len(keep_test)}')


def stitch_keep(blocks_per_type, keep_per_split):
    out_blocks = []
    offsets = []
    o = 0
    for pt, blk in zip(ALL_TYPES, blocks_per_type):
        k = keep_per_type[pt][keep_per_split]
        out_blocks.append(blk[k])
        offsets.append((pt, o, o + k.sum()))
        o += k.sum()
    return np.vstack(out_blocks), offsets


X_train_blocks = []
X_test_blocks  = []
o_tr = 0
o_te = 0
for pt in ALL_TYPES:
    n_tr = len(per_type_data[pt]['df_train'])
    n_te = len(per_type_data[pt]['df_test'])
    X_train_blocks.append(X_train[o_tr:o_tr + n_tr])
    X_test_blocks.append( X_test [o_te:o_te + n_te])
    o_tr += n_tr
    o_te += n_te

X_train_kept, train_offsets = stitch_keep(X_train_blocks, 0)
X_test_kept,  test_offsets  = stitch_keep(X_test_blocks,  1)
del X_train, X_test


cont_train_blocks, cont_test_blocks = [], []
for pt in ALL_TYPES:
    d = per_type_data[pt]
    k_tr, k_te = keep_per_type[pt]
    df_tr = d['df_train'].loc[k_tr]
    df_te = d['df_test'].loc[k_te]
    cells_tr = df_tr['cell_id'].to_numpy()
    cells_te = df_te['cell_id'].to_numpy()
    cell_arr_tr = np.vstack([cell2vec[c] for c in cells_tr]).astype(np.float32)
    cell_arr_te = np.vstack([cell2vec[c] for c in cells_te]).astype(np.float32)
    extras_tr = np.column_stack([
        df_tr['pert_time'].to_numpy(dtype=np.float32),
        df_tr['pert_dose'].to_numpy(dtype=np.float32),
        df_tr['ignore_flag_pert_time'].to_numpy(dtype=np.float32),
        df_tr['ignore_flag_pert_dose'].to_numpy(dtype=np.float32),
    ])
    extras_te = np.column_stack([
        df_te['pert_time'].to_numpy(dtype=np.float32),
        df_te['pert_dose'].to_numpy(dtype=np.float32),
        df_te['ignore_flag_pert_time'].to_numpy(dtype=np.float32),
        df_te['ignore_flag_pert_dose'].to_numpy(dtype=np.float32),
    ])
    cont_train_blocks.append(np.hstack([cell_arr_tr, extras_tr]))
    cont_test_blocks .append(np.hstack([cell_arr_te, extras_te]))

cont_train = np.vstack(cont_train_blocks)
cont_test  = np.vstack(cont_test_blocks)

cont_scale_cols = list(range(n_ctrl_pcs)) + [n_ctrl_pcs, n_ctrl_pcs + 1]
sc_cont = StandardScaler().fit(cont_train[:, cont_scale_cols])
cont_train[:, cont_scale_cols] = sc_cont.transform(cont_train[:, cont_scale_cols]).astype(np.float32)
cont_test [:, cont_scale_cols] = sc_cont.transform(cont_test [:, cont_scale_cols]).astype(np.float32)
print(f'  Continuous context (cell_pcs + pert_time + pert_dose + flags): '
      f'train {cont_train.shape}  test {cont_test.shape}')


pemb_train_blocks, pemb_test_blocks = [], []
oh_train_blocks,  oh_test_blocks  = [], []
pt_label_train_blocks, pt_label_test_blocks = [], []

for pt in ALL_TYPES:
    d = per_type_data[pt]
    k_tr, k_te = keep_per_type[pt]
    n_tr = int(k_tr.sum()); n_te = int(k_te.sum())
    pemb_train_blocks.append(d['emb_train_pca'][k_tr])
    pemb_test_blocks.append( d['emb_test_pca'][k_te])
    oh = np.zeros((n_tr, len(ALL_TYPES)), dtype=np.float32); oh[:, PT_INDEX[pt]] = 1.; oh_train_blocks.append(oh)
    oh = np.zeros((n_te, len(ALL_TYPES)), dtype=np.float32); oh[:, PT_INDEX[pt]] = 1.; oh_test_blocks.append(oh)
    pt_label_train_blocks.append(np.full(n_tr, PT_INDEX[pt], dtype=np.int32))
    pt_label_test_blocks.append( np.full(n_te, PT_INDEX[pt], dtype=np.int32))

pemb_train = np.vstack(pemb_train_blocks).astype(np.float32)
pemb_test  = np.vstack(pemb_test_blocks).astype(np.float32)
oh_train   = np.vstack(oh_train_blocks)
oh_test    = np.vstack(oh_test_blocks)
pt_label_train = np.concatenate(pt_label_train_blocks)
pt_label_test  = np.concatenate(pt_label_test_blocks)

C_train = np.hstack([cont_train, pemb_train, oh_train]).astype(np.float32)
C_test  = np.hstack([cont_test,  pemb_test,  oh_test ]).astype(np.float32)
del cont_train, cont_test, pemb_train, pemb_test, oh_train, oh_test

X_train = X_train_kept
X_test  = X_test_kept
del X_train_kept, X_test_kept

print(f'  C_train {C_train.shape}    X_train {X_train.shape}')
print(f'  C_test  {C_test.shape}    X_test  {X_test.shape}')


pop_model = CorrelationNetwork()
pop_model.fit(X_train)
print(f'  Population Train MSE: {pop_model.measure_mses(X_train).mean():.4f}')
print(f'  Population Test  MSE: {pop_model.measure_mses(X_test).mean():.4f}')


# =============================================================================
# STEP 6: Train ContextualizedCorrelation
# =============================================================================

print('\n' + '=' * 78)
print('STEP 6: Training ContextualizedCorrelation')
print('=' * 78)

train_idx, val_idx = train_test_split(
    np.arange(len(X_train)),
    test_size=VAL_FRAC,
    stratify=pt_label_train,
    random_state=RANDOM_STATE,
)
val_pt_counts = pd.Series(pt_label_train[val_idx]).map({i: pt for pt, i in PT_INDEX.items()}).value_counts().to_dict()
print(f'  Stratified val split per pert_type: {val_pt_counts}')

rng_bal_idx = np.random.default_rng(RANDOM_STATE)
train_pt = pt_label_train[train_idx]
balanced_idx = []
for i, pt in enumerate(ALL_TYPES):
    pt_idxs = train_idx[train_pt == i]
    if len(pt_idxs) >= BALANCE_PER_TYPE:
        pick = rng_bal_idx.choice(pt_idxs, size=BALANCE_PER_TYPE, replace=False)
    else:
        pick = rng_bal_idx.choice(pt_idxs, size=BALANCE_PER_TYPE, replace=True)
    balanced_idx.append(pick)
    print(f'  {pt}: {len(pt_idxs):,} unique -> {BALANCE_PER_TYPE:,} sampled')
train_idx = np.concatenate(balanced_idx)
rng_bal_idx.shuffle(train_idx)
print(f'  Balanced train -> {len(train_idx):,} rows total')

datamodule = CorrelationDataModule(
    C_train=C_train[train_idx], X_train=X_train[train_idx],
    C_val=C_train[val_idx],     X_val=X_train[val_idx],
    C_test=C_test,              X_test=X_test,
    C_predict=C_test,           X_predict=X_test,
    batch_size=BATCH_SIZE,
)

model = ContextualizedCorrelation(
    context_dim=C_train.shape[1],
    x_dim=X_train.shape[1],
    encoder_type='mlp',
    encoder_kwargs=ENCODER_KWARGS,
    num_archetypes=NUM_ARCHETYPES,
    learning_rate=LEARNING_RATE,
)

ckpt_cb = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model')
trainer = Trainer(
    default_root_dir=str(OUTPUT_DIR),
    max_epochs=MAX_EPOCHS,
    accelerator='auto', devices='auto',
    callbacks=[ckpt_cb],
    deterministic=True,
)
trainer.fit(model, datamodule=datamodule)
print(f'  best_model_path: {ckpt_cb.best_model_path}')


# =============================================================================
# STEP 7: Per-pert-type inference + MSE; save outputs
# =============================================================================

print('\n' + '=' * 78)
print('STEP 7: Per-pert-type inference + MSE; save outputs for trt_cp + trt_sh')
print('=' * 78)

ckpt = torch.load(ckpt_cb.best_model_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()
print(f'  inference device: {device}')


def slice_offsets(offsets, pt):
    for p, lo, hi in offsets:
        if p == pt:
            return lo, hi
    raise KeyError(pt)


n_x = X_train.shape[1]
total_save_rows = 0
for pt in SAVE_TYPES:
    k_tr, k_te = keep_per_type[pt]
    total_save_rows += int(k_tr.sum()) + int(k_te.sum())

corr_path = OUTPUT_DIR / 'full_dataset_correlations.npy'
mus_path  = OUTPUT_DIR / 'full_dataset_mus.npy'
corrs_mm = np.lib.format.open_memmap(str(corr_path), mode='w+', dtype='float32',
                                      shape=(total_save_rows, n_x, n_x))
mus_mm   = np.lib.format.open_memmap(str(mus_path),  mode='w+', dtype='float32',
                                      shape=(total_save_rows, n_x, n_x))


def infer_and_mse(C_np, X_np, save_offset=None, batch=2048):
    n = len(C_np)
    if n == 0:
        return np.array([], dtype=np.float64)
    mses = np.empty(n, dtype=np.float64)
    Ct = torch.tensor(C_np.astype(np.float32))
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            out = model.predict_step({'contexts': Ct[s:e].to(device)}, 0)
            betas = out['betas'].cpu().numpy()
            mus   = out['mus'].cpu().numpy()
            x     = X_np[s:e]
            resid = x[:, :, None] - betas * x[:, None, :] - mus
            chunk = np.sum(resid ** 2, axis=(1, 2)) / (x.shape[-1] ** 2)
            chunk[np.isnan(betas).any(axis=(1, 2))] = np.nan
            mses[s:e] = chunk
            if save_offset is not None:
                corrs_mm[save_offset + s:save_offset + e] = out['correlations'].cpu().numpy()
                mus_mm  [save_offset + s:save_offset + e] = mus
    return mses


mse_per = {}
save_offset = 0
save_meta_rows = []

for pt in ALL_TYPES:
    print(f'\n  --- {pt} ---')
    k_tr, k_te = keep_per_type[pt]
    lo_tr, hi_tr = slice_offsets(train_offsets, pt)
    lo_te, hi_te = slice_offsets(test_offsets,  pt)
    C_tr = C_train[lo_tr:hi_tr]; X_tr = X_train[lo_tr:hi_tr]
    C_te = C_test [lo_te:hi_te]; X_te = X_test [lo_te:hi_te]

    if pt in SAVE_TYPES:
        df_tr = per_type_data[pt]['df_train'].loc[k_tr].reset_index(drop=True)
        meta = df_tr[['cell_id', 'inst_id', 'pert_id', 'pert_time', 'pert_dose']].copy()
        meta['pert_type'] = pt; meta['split'] = 'train'
        meta['canonical_smiles'] = df_tr['canonical_smiles'] if 'canonical_smiles' in df_tr.columns else np.nan
        meta['ensembl_id']       = df_tr['ensembl_id']       if 'ensembl_id'       in df_tr.columns else np.nan
        save_meta_rows.append(meta)
        mse_tr = infer_and_mse(C_tr, X_tr, save_offset=save_offset)
        save_offset += len(C_tr)

        df_te = per_type_data[pt]['df_test'].loc[k_te].reset_index(drop=True)
        meta = df_te[['cell_id', 'inst_id', 'pert_id', 'pert_time', 'pert_dose']].copy()
        meta['pert_type'] = pt; meta['split'] = 'test'
        meta['canonical_smiles'] = df_te['canonical_smiles'] if 'canonical_smiles' in df_te.columns else np.nan
        meta['ensembl_id']       = df_te['ensembl_id']       if 'ensembl_id'       in df_te.columns else np.nan
        save_meta_rows.append(meta)
        mse_te = infer_and_mse(C_te, X_te, save_offset=save_offset)
        save_offset += len(C_te)
    else:
        mse_tr = infer_and_mse(C_tr, X_tr)
        mse_te = infer_and_mse(C_te, X_te)

    mse_per[(pt, 'train')] = mse_tr
    mse_per[(pt, 'test')]  = mse_te
    print(f'    train n={len(mse_tr):>7,}  MSE={np.nanmean(mse_tr):.4f}')
    print(f'    test  n={len(mse_te):>7,}  MSE={np.nanmean(mse_te):.4f}')

corrs_mm.flush(); mus_mm.flush()

meta_df = pd.concat(save_meta_rows, ignore_index=True)
meta_df['mse'] = np.concatenate([mse_per[(pt, split)]
                                  for pt in SAVE_TYPES
                                  for split in ('train', 'test')])
meta_df.to_csv(OUTPUT_DIR / 'full_dataset_predictions.csv', index=False)
print(f'\n  saved full_dataset_correlations.npy, full_dataset_mus.npy  shape={(total_save_rows, n_x, n_x)}')
print(f'  saved full_dataset_predictions.csv  rows={len(meta_df)}')


# =============================================================================
# Per-pert-type MSE summary
# =============================================================================

print('\n' + '=' * 78)
print('PER-PERTURBATION-TYPE MSE SUMMARY (Contextualized model)')
print('=' * 78)
print(f"{'pert_type':<10}  {'n_train':>10}  {'n_test':>10}  {'n_full':>10}  "
      f"{'train_mse':>10}  {'test_mse':>10}  {'full_mse':>10}")
print('-' * 78)
summary_rows = []
for pt in ALL_TYPES:
    tr = mse_per[(pt, 'train')]
    te = mse_per[(pt, 'test')]
    full = np.concatenate([tr, te])
    row = {
        'pert_type': pt,
        'n_train': len(tr), 'n_test': len(te), 'n_full': len(full),
        'train_mse': float(np.nanmean(tr)) if len(tr) else float('nan'),
        'test_mse':  float(np.nanmean(te)) if len(te) else float('nan'),
        'full_mse':  float(np.nanmean(full)) if len(full) else float('nan'),
    }
    summary_rows.append(row)
    print(f"{pt:<10}  {row['n_train']:>10,}  {row['n_test']:>10,}  {row['n_full']:>10,}  "
          f"{row['train_mse']:>10.4f}  {row['test_mse']:>10.4f}  {row['full_mse']:>10.4f}")

all_tr = np.concatenate([mse_per[(pt, 'train')] for pt in ALL_TYPES])
all_te = np.concatenate([mse_per[(pt, 'test')]  for pt in ALL_TYPES])
print('-' * 78)
print(f"{'JOINT':<10}  {len(all_tr):>10,}  {len(all_te):>10,}  {len(all_tr)+len(all_te):>10,}  "
      f"{np.nanmean(all_tr):>10.4f}  {np.nanmean(all_te):>10.4f}  "
      f"{np.nanmean(np.concatenate([all_tr, all_te])):>10.4f}")

pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / 'per_pert_type_mse.csv', index=False)
print(f'\n  saved per_pert_type_mse.csv')
print('\nDone.')
