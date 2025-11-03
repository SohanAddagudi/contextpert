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
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import squareform
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

DATA_DIR = Path(os.environ["CONTEXTPERT_DATA_DIR"])
DISEASE_DRUG_TRIPLES = DATA_DIR / "opentargets" / "disease_drug_triples_csv" / "disease_drug_triples_lincs.csv"
TRT_CP_PATH = DATA_DIR / "trt_cp_smiles_qc.csv"
OUTDIR = Path("figs") / "disease_similarity"

def parse_args():
    p = argparse.ArgumentParser(description="Fig 1: Separate cell type clustermaps")
    p.add_argument("--representation", type=str, default="metagenes",
                   choices=["expression", "metagenes", "morgan", "network"],
                   help="Drug representation to use")
    p.add_argument("--cell1", type=str, default=None, help="First cell type (auto if None)")
    p.add_argument("--cell2", type=str, default=None, help="Second cell type (auto if None)")
    p.add_argument("--min_overlap_drugs", type=int, default=20, help="Min overlapping drugs between cells")
    p.add_argument("--min_samples", type=int, default=3, help="Min drugs per disease")
    p.add_argument("--max_diseases", type=int, default=None, help="Cap number of diseases")
    p.add_argument("--pca_dim", type=int, default=50, help="PCA dimensions for metagenes")
    p.add_argument("--morgan_radius", type=int, default=2, help="Morgan fingerprint radius")
    p.add_argument("--morgan_bits", type=int, default=2048, help="Morgan fingerprint bits")
    p.add_argument("--linkage", type=str, default="average", 
                   choices=["average","single","complete","ward"])
    p.add_argument("--network_results_dir", type=str, default=None,
                   help="Directory containing network prediction outputs (required for representation=network)")
    return p.parse_args()

def cdist_fast(mat):
    mat = np.asarray(mat, dtype=np.float32, order="C")
    xy = mat @ mat.T
    norms = (mat * mat).sum(1)
    d2 = np.add.outer(norms, norms) - 2 * xy
    np.fill_diagonal(d2, 0.0)
    d2[d2 < 0] = 0.0
    return squareform(np.sqrt(d2, dtype=np.float32))


def build_common_structure(drug_disease_df, trt_cp_df):
    """Merge disease labels with all trt_cp samples (atomic level)"""
    return trt_cp_df.merge(
        drug_disease_df[['smiles', 'diseaseName']],
        left_on='canonical_smiles',
        right_on='smiles',
        how='inner'
    )


def find_best_cell_pair(trt_cp_with_disease):
    """Find two cell types with maximum overlapping drugs (with disease labels)"""
    # Get drugs per cell
    cell_drugs = trt_cp_with_disease.groupby('cell_id')['pert_id'].apply(set)
    
    # Find pair with max overlap
    best_overlap = 0
    best_pair = None
    
    cells = list(cell_drugs.index)
    for i in range(len(cells)):
        for j in range(i+1, len(cells)):
            overlap = len(cell_drugs[cells[i]] & cell_drugs[cells[j]])
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (cells[i], cells[j])
    
    return best_pair, best_overlap


def build_expression_per_drug(samples_one_cell):
    """Average expression per drug (one sample per drug)"""
    metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                     'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                     'pct_self_rank_q25', 'canonical_smiles', 'inchi_key', 'smiles', 'diseaseName']
    gene_cols = [col for col in samples_one_cell.columns if col not in metadata_cols]
    
    # Aggregate by drug (pert_id)
    drug_data = samples_one_cell.groupby('pert_id').agg({
        **{col: 'mean' for col in gene_cols},
        'diseaseName': 'first',
        'canonical_smiles': 'first'
    }).reset_index()
    
    return drug_data[gene_cols].values, drug_data['diseaseName'].values, drug_data['pert_id'].values


def build_metagenes_per_drug(samples_one_cell, trt_cp_full, n_metagenes=50):
    """Metagenes per drug (averaged, then PCA transformed)"""
    metadata_cols = ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                     'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                     'pct_self_rank_q25', 'canonical_smiles', 'inchi_key', 'smiles', 'diseaseName']
    gene_cols = [col for col in samples_one_cell.columns if col not in metadata_cols]
    
    # Learn PCA from all trt_cp
    trt_cp_gene_cols = [col for col in trt_cp_full.columns 
                        if col not in ['inst_id', 'cell_id', 'pert_id', 'pert_type', 'pert_dose',
                                       'pert_dose_unit', 'pert_time', 'sig_id', 'distil_cc_q75',
                                       'pct_self_rank_q25', 'canonical_smiles', 'inchi_key']]
    scaler = StandardScaler()
    pca = PCA(n_components=n_metagenes, random_state=42)
    pca.fit(scaler.fit_transform(trt_cp_full[trt_cp_gene_cols].values))
    
    # Aggregate by drug first
    drug_data = samples_one_cell.groupby('pert_id').agg({
        **{col: 'mean' for col in gene_cols},
        'diseaseName': 'first',
        'canonical_smiles': 'first'
    }).reset_index()
    
    # Transform to metagene space
    X_metagenes = pca.transform(scaler.transform(drug_data[gene_cols].values))
    
    return X_metagenes, drug_data['diseaseName'].values, drug_data['pert_id'].values


def build_morgan_per_drug(samples_one_cell, radius=2, n_bits=2048):
    """Morgan fingerprints per drug"""
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    # Get unique drugs
    drug_data = samples_one_cell.groupby('pert_id').agg({
        'canonical_smiles': 'first',
        'diseaseName': 'first'
    }).reset_index()
    
    # Compute fingerprints
    fps = []
    valid_indices = []
    for idx, row in drug_data.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol is not None:
            fp = morgan_gen.GetFingerprint(mol)
            fp_array = np.zeros(n_bits, dtype=np.float32)
            for i in range(n_bits):
                fp_array[i] = fp[i]
            fps.append(fp_array)
            valid_indices.append(idx)
    
    drug_data = drug_data.iloc[valid_indices].reset_index(drop=True)
    X = np.array(fps, dtype=np.float32)
    
    return X, drug_data['diseaseName'].values, drug_data['pert_id'].values


def load_network_prediction_lookup(results_dir: str | Path) -> dict[str, np.ndarray]:
    """Load network correlation vectors indexed by inst_id."""
    results_path = Path(results_dir)
    predictions_path = results_path / "full_dataset_predictions.csv"
    correlations_path = results_path / "full_dataset_correlations.npy"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Network predictions CSV not found at {predictions_path}")
    if not correlations_path.exists():
        raise FileNotFoundError(f"Network correlation matrix not found at {correlations_path}")

    predictions_df = pd.read_csv(predictions_path)
    correlations = np.load(correlations_path, allow_pickle=False)

    if correlations.ndim == 3:
        p = correlations.shape[1]
        iu = np.triu_indices(p, k=1)
        correlations = correlations[:, iu[0], iu[1]]
    elif correlations.ndim != 2:
        raise ValueError(f"Unexpected correlation tensor shape: {correlations.shape}")

    if len(predictions_df) != correlations.shape[0]:
        raise ValueError(
            "Correlation array length does not match predictions CSV "
            f"({correlations.shape[0]} vs {len(predictions_df)})"
        )

    correlations = correlations.astype(np.float32, copy=False)
    if not np.isfinite(correlations).all():
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)

    inst_ids = predictions_df['inst_id'].astype(str).tolist()
    lookup: dict[str, list[np.ndarray]] = {}
    for inst_id, vector in zip(inst_ids, correlations):
        if inst_id not in lookup:
            lookup[inst_id] = [vector]
        else:
            lookup[inst_id].append(vector)

    averaged_lookup = {
        inst_id: np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        for inst_id, vectors in lookup.items()
    }

    return averaged_lookup


def build_network_prediction_per_drug(samples_one_cell: pd.DataFrame,
                                      network_lookup: dict[str, np.ndarray]):
    """Aggregate network correlation predictions per drug for a single cell."""
    if samples_one_cell.empty:
        return (np.empty((0, 0), dtype=np.float32),
                np.array([], dtype=object),
                np.array([], dtype=object))

    df = samples_one_cell.copy().reset_index(drop=True)
    df['inst_key'] = df['inst_id'].astype(str)
    available_mask = df['inst_key'].isin(network_lookup)

    if not available_mask.any():
        cell_name = df['cell_id'].iloc[0] if 'cell_id' in df.columns else 'unknown'
        raise ValueError(f"No network predictions found for cell {cell_name}.")

    df = df.loc[available_mask].reset_index(drop=True)

    feature_matrix = np.stack([network_lookup[key] for key in df['inst_key']], axis=0).astype(np.float32)
    if not np.isfinite(feature_matrix).all():
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    vectors, diseases, drugs = [], [], []
    for pert_id, group in df.groupby('pert_id', sort=False):
        indices = group.index.to_numpy()
        vectors.append(feature_matrix[indices].mean(axis=0))
        diseases.append(group['diseaseName'].iloc[0])
        drugs.append(pert_id)

    return (np.vstack(vectors).astype(np.float32),
            np.array(diseases, dtype=object),
            np.array(drugs, dtype=object))


def align_to_drug_order(X: np.ndarray, diseases: np.ndarray, drugs: np.ndarray, ordered_drugs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder arrays to match the provided ordered_drugs list."""
    index_map = {drug: idx for idx, drug in enumerate(drugs)}
    missing = [drug for drug in ordered_drugs if drug not in index_map]
    if missing:
        raise ValueError(f"Missing expected drugs during alignment: {missing[:5]}")
    positions = [index_map[drug] for drug in ordered_drugs]
    return (X[positions], diseases[positions], np.array(ordered_drugs, dtype=object))

def _build_block_midpoints(diseases_ordered):
    """Return dict {disease: midpoint_row_index_in_heatmap}"""
    mids = {}
    i = 0
    n = len(diseases_ordered)
    while i < n:
        d = diseases_ordered[i]
        j = i
        while j < n and diseases_ordered[j] == d:
            j += 1
        mids[d] = (i + j - 1) / 2.0
        i = j
    return mids


def _to_tree_and_draw_aligned(ax, Z, label_order, label_to_y, orientation='left', line_kwargs=None):
    """
    Draw a SciPy linkage as a dendrogram whose leaves are aligned to given y-positions.
    orientation: 'left' or 'right' (tree grows away from the heatmap).
    """
    if line_kwargs is None:
        line_kwargs = dict(color='0.2', linewidth=1.0)
    
    root, _ = to_tree(Z, rd=True)
    
    # Recursively place leaves to requested y's; x is cumulative cluster height.
    def assign(node):
        if node.is_leaf():
            y = float(label_to_y[label_order[node.id]])
            x = 0.0
            return y, x
        yl, xl = assign(node.left)
        yr, xr = assign(node.right)
        y_here = 0.5 * (yl + yr)
        x_here = max(xl, xr) + max(node.dist, 1e-9)
        
        # Draw: two horizontals from children to join, then vertical trunk
        if orientation == 'left':
            ax.plot([xl, x_here], [yl, yl], **line_kwargs)
            ax.plot([xr, x_here], [yr, yr], **line_kwargs)
            ax.plot([x_here, x_here], [yl, yr], **line_kwargs)
        elif orientation == 'right':
            # Mirror: tree grows right, trunk at x=0
            ax.plot([0 - xl, 0 - x_here], [yl, yl], **line_kwargs)
            ax.plot([0 - xr, 0 - x_here], [yr, yr], **line_kwargs)
            ax.plot([0 - x_here, 0 - x_here], [yl, yr], **line_kwargs)
        else:
            raise ValueError("orientation must be 'left' or 'right'")
        return y_here, x_here
    
    assign(root)
    ax.invert_yaxis()
    if orientation == 'right':
        ax.invert_xaxis()  # Trunk flush against heatmap
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


# Plotting
def plot_heatmap_with_disease_dendrogram(X, diseases, drugs, cell_name, disease_palette, out_prefix, linkage_method="average"):
    """
    Plot drug-level heatmap with custom layout:
    - Left: Distance colorbar
    - Center: Heatmap with top dendrogram only, disease color bar
    """
    n = len(diseases)
    if n < 2:
        return None
    
    # Compute distances and DRUG-level clustering
    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    dcond = cdist_fast(X_clean)
    Z = hierarchy.linkage(dcond, method=linkage_method)
    D = squareform(dcond)

    # Reorder by clustering (same for both axes - symmetric)
    dendro_data = hierarchy.dendrogram(Z, no_plot=True)
    leaves = np.array(dendro_data["leaves"], dtype=int)
    D_ordered = D[leaves][:, leaves]
    diseases_ordered = diseases[leaves]
    
    # Min-max scale to [0,1] globally (preserve symmetry)
    d_min = float(np.min(D_ordered))
    d_max = float(np.max(D_ordered))
    if d_max > d_min:
        D_scaled = (D_ordered - d_min) / (d_max - d_min)
    else:
        D_scaled = np.zeros_like(D_ordered, dtype=np.float32)
    D_scaled = D_scaled.astype(np.float32, copy=False)
    
    # Create figure layout with proper alignment
    # Distance colorbar should align with heatmap only, not dendro+colors
    fig = plt.figure(figsize=(11, 9))
    
    # Main gridspec: 3 rows (dendro, colors, heatmap), 2 cols (empty+cbar, main)
    gs = fig.add_gridspec(nrows=3, ncols=2, 
                         width_ratios=[0.4, 10],
                         height_ratios=[1.5, 0.2, 8],
                         wspace=0.05, hspace=0.02)
    
    # Top row: dendrogram (spans right column only)
    ax_dendro = fig.add_subplot(gs[0, 1])
    
    # Middle row: disease color bar (spans right column only)  
    ax_colors = fig.add_subplot(gs[1, 1])
    
    # Bottom row: colorbar (left) and heatmap (right)
    ax_cbar = fig.add_subplot(gs[2, 0])
    ax_heat = fig.add_subplot(gs[2, 1])
    
    # Set common extent for all elements
    n_items = len(diseases_ordered)
    extent = (-0.5, n_items - 0.5, -0.5, n_items - 0.5)  # (left, right, bottom, top)
    xlim = (-0.5, n_items - 0.5)
    
    # Draw dendrogram (aligned with heatmap)
    for xs, ys in zip(dendro_data["icoord"], dendro_data["dcoord"]):
        xs_scaled = (np.asarray(xs) - 5.0) / 10.0
        ax_dendro.plot(xs_scaled, ys, color="0.2", linewidth=1.0)

    ax_dendro.set_xlim(xlim)
    max_height = max((max(segment) for segment in dendro_data["dcoord"]), default=1.0)
    ax_dendro.set_ylim(0, max_height * 1.02)
    ax_dendro.set_aspect('auto')  # Let it stretch to match gridspec
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.set_xlabel('')
    ax_dendro.set_ylabel('')
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    ax_dendro.margins(x=0)
    
    # Draw disease color bar (aligned with heatmap)
    disease_colors_ordered = [disease_palette[d] for d in diseases_ordered]
    color_array = np.array(disease_colors_ordered).reshape(1, -1, 3)
    ax_colors.imshow(color_array, aspect='auto', interpolation='nearest', extent=xlim + (0, 1))
    ax_colors.set_xlim(xlim)
    ax_colors.set_xticks([])
    ax_colors.set_yticks([])
    ax_colors.set_frame_on(False)
    
    # Draw heatmap (square aspect ratio)
    im = ax_heat.imshow(D_scaled, cmap="coolwarm", aspect='equal', interpolation='nearest', 
                       vmin=0, vmax=1, extent=extent)
    ax_heat.set_xlim(xlim)
    ax_heat.set_ylim(-0.5, n_items - 0.5)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel('')
    ax_heat.set_ylabel('')
    # No title - cell type is implicit
    
    # Left colorbar for distances
    cb = plt.colorbar(im, cax=ax_cbar)
    ax_cbar.yaxis.set_label_position("left")
    ax_cbar.yaxis.tick_left()

    # Match subplot geometry after initial draw
    fig.canvas.draw()
    heat_box = ax_heat.get_position()
    dendro_box = ax_dendro.get_position()
    colors_box = ax_colors.get_position()
    cbar_box = ax_cbar.get_position()

    # Pull distance colorbar closer to heatmap, keep same width & height as heatmap
    pad = 0.01
    ax_cbar.set_position([
        max(heat_box.x0 - pad - cbar_box.width, 0.02),
        heat_box.y0,
        cbar_box.width,
        heat_box.height
    ])

    # Place disease color strip slightly above heatmap; height = single sample width
    sample_size = heat_box.width / max(n_items, 1)
    gap = sample_size * 0.2
    colors_y0 = min(heat_box.y1 + gap, 1.0 - sample_size - 0.01)
    ax_colors.set_position([
        heat_box.x0,
        colors_y0,
        heat_box.width,
        sample_size
    ])

    # Extend dendrogram so it touches the disease strip (no gap)
    colors_box_new = ax_colors.get_position()
    dendro_top = dendro_box.y0 + dendro_box.height
    dendro_bottom = colors_box_new.y1
    if dendro_top <= dendro_bottom:
        dendro_top = dendro_bottom + 0.02
    ax_dendro.set_position([
        heat_box.x0,
        dendro_bottom,
        heat_box.width,
        dendro_top - dendro_bottom
    ])
    ax_dendro.set_xlim(xlim)
    
    # Save
    png = out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    triples = pd.read_csv(DISEASE_DRUG_TRIPLES)
    trt_cp = pd.read_csv(TRT_CP_PATH)
    bad_smiles = ['-666', 'restricted']
    trt_cp = trt_cp[~trt_cp['canonical_smiles'].isin(bad_smiles)].copy()
    trt_cp = trt_cp[trt_cp['canonical_smiles'].notna()].copy()
    
    # Filter diseases
    disease_counts = triples.groupby('diseaseName')['smiles'].nunique().sort_values(ascending=False)
    keep_diseases = disease_counts[disease_counts >= args.min_samples].index.tolist()
    triples_filtered = triples[triples['diseaseName'].isin(keep_diseases)].copy()
    
    if args.max_diseases and args.max_diseases > 0:
        top_diseases = disease_counts.head(args.max_diseases).index.tolist()
        triples_filtered = triples_filtered[triples_filtered['diseaseName'].isin(top_diseases)].copy()
    
    # Build atomic samples with disease labels
    common = build_common_structure(triples_filtered, trt_cp)
    
    # Find best cell pair or use specified
    if args.cell1 and args.cell2:
        cell1, cell2 = args.cell1, args.cell2
        overlap_drugs = (set(common[common['cell_id']==cell1]['pert_id'].unique()) & 
                        set(common[common['cell_id']==cell2]['pert_id'].unique()))
        n_overlap = len(overlap_drugs)
    else:
        (cell1, cell2), n_overlap = find_best_cell_pair(common)
        overlap_drugs = (set(common[common['cell_id']==cell1]['pert_id'].unique()) & 
                        set(common[common['cell_id']==cell2]['pert_id'].unique()))
    
    if n_overlap < args.min_overlap_drugs:
        raise ValueError(f"Only {n_overlap} overlapping drugs (need ≥{args.min_overlap_drugs})")
    
    # Filter to overlapping drugs only
    samples_cell1 = common[(common['cell_id'] == cell1) & (common['pert_id'].isin(overlap_drugs))].copy()
    samples_cell2 = common[(common['cell_id'] == cell2) & (common['pert_id'].isin(overlap_drugs))].copy()
    
    # Build representations (aggregated by drug)

    network_lookup = None
    if args.representation == "network":
        if not args.network_results_dir:
            raise ValueError("--network_results_dir must be provided when representation='network'.")
        network_lookup = load_network_prediction_lookup(args.network_results_dir)
    
    if args.representation == "expression":
        X1, diseases1, drugs1 = build_expression_per_drug(samples_cell1)
        X2, diseases2, drugs2 = build_expression_per_drug(samples_cell2)
    elif args.representation == "metagenes":
        X1, diseases1, drugs1 = build_metagenes_per_drug(samples_cell1, trt_cp, n_metagenes=args.pca_dim)
        X2, diseases2, drugs2 = build_metagenes_per_drug(samples_cell2, trt_cp, n_metagenes=args.pca_dim)
    elif args.representation == "morgan":
        X1, diseases1, drugs1 = build_morgan_per_drug(samples_cell1, radius=args.morgan_radius, n_bits=args.morgan_bits)
        X2, diseases2, drugs2 = build_morgan_per_drug(samples_cell2, radius=args.morgan_radius, n_bits=args.morgan_bits)
    elif args.representation == "network":
        X1, diseases1, drugs1 = build_network_prediction_per_drug(samples_cell1, network_lookup)
        X2, diseases2, drugs2 = build_network_prediction_per_drug(samples_cell2, network_lookup)
    else:
        raise ValueError(f"Unknown representation: {args.representation}")
    
    # Align both cells to the common set of drugs (preserve order from cell1)
    drugs1_list = drugs1.tolist() if isinstance(drugs1, np.ndarray) else list(drugs1)
    drugs2_set = set(drugs2.tolist() if isinstance(drugs2, np.ndarray) else list(drugs2))
    ordered_common_drugs = [drug for drug in drugs1_list if drug in drugs2_set]
    if not ordered_common_drugs:
        raise ValueError("No common drugs remain after building representations; cannot proceed.")

    X1, diseases1, drugs1 = align_to_drug_order(X1, diseases1, drugs1, ordered_common_drugs)
    X2, diseases2, drugs2 = align_to_drug_order(X2, diseases2, drugs2, ordered_common_drugs)
    
    # Create CONSISTENT disease palette across both plots
    all_diseases = sorted(set(diseases1) | set(diseases2))
    disease_pal = sns.color_palette("tab20", n_colors=max(20, len(all_diseases)))
    disease_palette = {d: disease_pal[i % len(disease_pal)] for i, d in enumerate(all_diseases)}
    
    plot_heatmap_with_disease_dendrogram(
        X1, diseases1, drugs1, cell1, disease_palette,
        OUTDIR / f"fig1_{args.representation}_{cell1}",
        linkage_method=args.linkage
    )
    
    plot_heatmap_with_disease_dendrogram(
        X2, diseases2, drugs2, cell2, disease_palette,
        OUTDIR / f"fig1_{args.representation}_{cell2}",
        linkage_method=args.linkage
    )
    
    # Save disease legend separately
    fig_legend, ax_legend = plt.subplots(figsize=(4, max(3, len(all_diseases) * 0.25)))
    ax_legend.axis('off')
    handles = [plt.Line2D([0], [0], marker='s', color='w', 
                         markerfacecolor=disease_palette[d], markersize=10, label=d)
              for d in all_diseases]
    ax_legend.legend(handles=handles, loc='center', frameon=False, fontsize=10)
    fig_legend.savefig(OUTDIR / f"fig1_{args.representation}_legend.png", 
                      dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_legend)
    
if __name__ == "__main__":
    main()
