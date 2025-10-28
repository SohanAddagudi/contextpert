import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
DISEASE_DRUG_TRIPLES = DATA_DIR / "opentargets" / "disease_drug_triples_csv" / "disease_drug_triples_lincs.csv"
TRT_CP_PATH = DATA_DIR / "trt_cp_smiles_qc.csv"

OUTDIR = Path("figs") / "disease_similarity"

def parse_args():
    p = argparse.ArgumentParser(description="Disease similarity clustermaps")
    p.add_argument("--min_samples", type=int, default=3,
                   help="Minimum number of drugs per disease to include.")
    p.add_argument("--max_diseases", type=int, default=None,
                   help="Max number of diseases to plot (None for all).")
    p.add_argument("--linkage", type=str, default="average",
                   choices=["average","single","complete","ward"],
                   help="Linkage method for hierarchical clustering.")
    p.add_argument("--pca_dim", type=int, default=50,
                   help="PCA dimensions for metagenes.")
    p.add_argument("--morgan_radius", type=int, default=2,
                   help="Morgan fingerprint radius.")
    p.add_argument("--morgan_bits", type=int, default=2048,
                   help="Morgan fingerprint bit length.")
    p.add_argument("--with_colorbar", action="store_true",
                   help="Show colorbar for distance scale.")
    return p.parse_args()


def cdist_fast(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32, order="C")
    xy = mat @ mat.T
    norms = (mat * mat).sum(1)
    d2 = np.add.outer(norms, norms) - 2 * xy
    np.fill_diagonal(d2, 0.0)
    d2[d2 < 0] = 0.0
    return squareform(np.sqrt(d2, dtype=np.float32))


def palette(values, palette_name="tab20"):
    vals = list(values)
    uniq = pd.unique(pd.Series(vals))
    pal = sns.color_palette(palette_name, n_colors=max(20, len(uniq)))
    lut = {u: pal[i % len(pal)] for i, u in enumerate(uniq)}
    return [lut[v] for v in vals]


def title_case_drug(s: str) -> str:
    if s is None or str(s).strip() == "":
        return ""
    s = str(s).strip()
    if s.isupper() or s.islower():
        s = s.lower().title()
    ACR = {"DNA","RNA","EGFR","HER2","MEK","JAK","BRAF","HDAC","VEGF","PI3K","PD1","PD-L1","CDK"}
    import re
    toks = re.split(r"([-/\s])", s)
    toks = [t.upper() if t.upper() in ACR else t for t in toks]
    return "".join(toks)


def build_common_sample_structure(drug_disease_df, trt_cp_df):
    trt_cp_with_disease = trt_cp_df.merge(
        drug_disease_df[['smiles', 'diseaseName']],
        left_on='canonical_smiles',
        right_on='smiles',
        how='inner'
    )
    return trt_cp_with_disease


def build_expression_representation(common_samples):
    metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                     'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                     'pct_self_rank_q25', 'canonical_smiles', 'inchi_key', 'smiles', 'diseaseName']
    gene_cols = [col for col in common_samples.columns if col not in metadata_cols]
    return common_samples[gene_cols].values


def build_metagenes_representation(common_samples, trt_cp_df, n_metagenes=50):
    metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                     'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                     'pct_self_rank_q25', 'canonical_smiles', 'inchi_key', 'smiles', 'diseaseName']
    gene_cols = [col for col in common_samples.columns if col not in metadata_cols]
    
    # Learn metagenes from all trt_cp samples
    trt_cp_gene_cols = [col for col in trt_cp_df.columns 
                        if col not in ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                                       'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                                       'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']]
    
    X_all = trt_cp_df[trt_cp_gene_cols].values
    scaler = StandardScaler()
    pca = PCA(n_components=n_metagenes, random_state=42)
    pca.fit(scaler.fit_transform(X_all))
    
    X_metagenes = pca.transform(scaler.transform(common_samples[gene_cols].values))
    return X_metagenes


def build_morgan_representation(common_samples, radius=2, n_bits=2048):
    from rdkit.Chem import rdFingerprintGenerator
    
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    # Compute fingerprints for unique drugs
    smiles_to_fp = {}
    for smiles in common_samples['canonical_smiles'].unique():
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            fp_array = np.zeros(n_bits, dtype=np.float32)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            smiles_to_fp[smiles] = fp_array
    
    # Assign to all atomic samples
    X = np.array([smiles_to_fp[smiles] for smiles in common_samples['canonical_smiles'].values])
    return X


def generate_clustermap(X, tbl, title, out_prefix, linkage_method="average", with_colorbar=False):
    """Generate clustermap with three annotation bands (Disease, Drug, Cell type)"""
    n = len(tbl)
    
    # Compute distances
    X_clean = np.where(np.isfinite(X), X, 0.0)
    dist_cond = cdist_fast(X_clean)
    Z = hierarchy.linkage(dist_cond, method=linkage_method)
    D = squareform(dist_cond)
    
    # Create distance matrix and annotations
    idx_labels = [f"sample_{i}" for i in range(n)]
    df = pd.DataFrame(D, index=idx_labels, columns=idx_labels)
    
    row_colors_df = pd.DataFrame({
        'Disease': palette(tbl["disease"].tolist(), "tab20"),
        'Drug': palette(tbl["drug_disp"].tolist(), "tab20"),  
        'Cell type': palette(tbl["cell"].tolist(), "tab20"),
    }, index=idx_labels)
    
    # Plot
    g = sns.clustermap(
        df, row_linkage=Z, col_linkage=Z, cmap="vlag",
        xticklabels=False, yticklabels=False, figsize=(15, 15),
        row_colors=row_colors_df, col_colors=row_colors_df,
        cbar_kws={"label": "Euclidean distance"} if with_colorbar else None,
        cbar_pos=None if not with_colorbar else (0.03, 0.86, 0.02, 0.10),
        dendrogram_ratio=(0.12, 0.12), colors_ratio=(0.06, 0.06),
    )
    
    if g.ax_heatmap.collections:
        g.ax_heatmap.collections[0].set_rasterized(True)
    
    for ax in list(g.fig.axes):
        if ax not in {g.ax_heatmap, g.ax_row_dendrogram, g.ax_col_dendrogram,
                      g.ax_row_colors, g.ax_col_colors, getattr(g, "cax", None)}:
            ax.set_visible(False)
    
    # Save outputs
    png = out_prefix.with_suffix(".png")
    g.fig.patch.set_facecolor("white")
    g.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(g.fig)
    
    np.save(out_prefix.with_name(out_prefix.name.replace("clustermap_", "dist_")), D)
    
    leaves = hierarchy.leaves_list(Z)
    order_file = out_prefix.with_name(out_prefix.name.replace("clustermap_", "row_order_")).with_suffix(".txt")
    with open(order_file, "w", encoding="utf-8") as f:
        for i in leaves:
            f.write(f"{i}\n")
    
    print(f"  Saved: {png.name}")

def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    print("DISEASE SIMILARITY CLUSTERMAPS")
    print(f"Outputs → {OUTDIR.resolve()}\n")
    
    # Load data
    triples = pd.read_csv(DISEASE_DRUG_TRIPLES)
    trt_cp = pd.read_csv(TRT_CP_PATH)
    bad_smiles = ['-666', 'restricted']
    trt_cp = trt_cp[~trt_cp['canonical_smiles'].isin(bad_smiles)].copy()
    trt_cp = trt_cp[trt_cp['canonical_smiles'].notna()].copy()
    
    # Filter diseases
    disease_counts = triples.groupby('diseaseName')['smiles'].nunique().sort_values(ascending=False)
    disease_counts.to_csv(OUTDIR / "disease_counts.csv", header=["n_drugs"])
    
    keep_diseases = disease_counts[disease_counts >= args.min_samples].index.tolist()
    triples_filtered = triples[triples['diseaseName'].isin(keep_diseases)].copy()
    
    if args.max_diseases and args.max_diseases > 0:
        top_diseases = disease_counts.head(args.max_diseases).index.tolist()
        triples_filtered = triples_filtered[triples_filtered['diseaseName'].isin(top_diseases)].copy()
    
    # Build atomic sample structure
    common_samples = build_common_sample_structure(triples_filtered, trt_cp)
    
    tbl = pd.DataFrame({
        "disease": common_samples['diseaseName'].values,
        "drug_id": common_samples['pert_id'].values,
        "cell": common_samples['cell_id'].values,
        "smiles": common_samples['smiles'].values,
    })
    tbl["drug_disp"] = tbl["drug_id"].astype(str).map(title_case_drug)
    
    # Pre-sort by cell type
    sort_order = tbl.sort_values('cell').index
    tbl = tbl.loc[sort_order].reset_index(drop=True)
    common_samples = common_samples.loc[sort_order].reset_index(drop=True)
    
    print(f"Clustering {len(tbl):,} atomic samples:")
    print(f"  {tbl['disease'].nunique()} diseases, {tbl['drug_id'].nunique()} drugs, {tbl['cell'].nunique()} cells\n")
    
    # Build and plot representations
    sns.set(context="notebook", style="white")
    
    representations = {
        'expression': lambda: build_expression_representation(common_samples),
        'metagenes': lambda: build_metagenes_representation(common_samples, trt_cp, n_metagenes=args.pca_dim),
        'morgan': lambda: build_morgan_representation(common_samples, radius=args.morgan_radius, n_bits=args.morgan_bits),
    }
    
    for rep_name, rep_fn in representations.items():
        print(f"Processing {rep_name}...")
        try:
            X = rep_fn()
            if X.shape[0] != len(tbl):
                print(f"Shape mismatch")
                continue
            
            generate_clustermap(X, tbl, f"Disease Similarity – {rep_name.capitalize()}",
                              OUTDIR / f"clustermap_{rep_name}", args.linkage, args.with_colorbar)
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nDone. Outputs in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
