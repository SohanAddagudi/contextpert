import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from collections import defaultdict
import warnings

from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


def evaluate_drug_target_mapping(drug_repr_df, target_repr_df, target_pairs_df, k_list=[1, 5, 10, 50]):
    """Evaluates drug and target representations for drug-target interaction prediction

    Computes bidirectional retrieval metrics (drug->target and target->drug) using k-nearest
    neighbors, as well as graph-based metrics (AUROC, AUPRC) by thresholding distances to
    create predicted bipartite graphs.

    Args:
        drug_repr_df: DataFrame with 'smiles' column + representation columns
        target_repr_df: DataFrame with 'targetId' column + representation columns
        target_pairs_df: DataFrame with 'smiles' and 'targetId' columns (ground truth positives)
        k_list: List of k values for computing retrieval metrics

    Returns:
        dict: Metrics including precision@k, recall@k, MRR@k (both directions) and AUROC/AUPRC
    """
    # Canonicalize SMILES
    drug_repr_df = drug_repr_df.copy()
    target_pairs_df = target_pairs_df.copy()
    drug_repr_df['smiles'] = drug_repr_df['smiles'].apply(canonicalize_smiles)
    target_pairs_df['smiles'] = target_pairs_df['smiles'].apply(canonicalize_smiles)

    # Validate that all drugs and targets in ground truth are in representations
    missing_drugs = target_pairs_df[~target_pairs_df['smiles'].isin(drug_repr_df['smiles'])]
    missing_targets = target_pairs_df[~target_pairs_df['targetId'].isin(target_repr_df['targetId'])]

    if not missing_drugs.empty:
        warnings.warn(f"{len(missing_drugs)} drug-target pairs have drugs missing from drug_repr_df")
        target_pairs_df = target_pairs_df[target_pairs_df['smiles'].isin(drug_repr_df['smiles'])]

    if not missing_targets.empty:
        warnings.warn(f"{len(missing_targets)} drug-target pairs have targets missing from target_repr_df")
        target_pairs_df = target_pairs_df[target_pairs_df['targetId'].isin(target_repr_df['targetId'])]

    assert missing_drugs.empty, f"Failure: {len(missing_drugs)} SMILES in target_pairs_df are missing from drug_repr_df: {missing_smiles['smiles'].tolist()}"
    assert missing_targets.empty, f"Failure: {len(missing_targets)} genes in target_pairs_df are missing from target_repr_df: {missing_targets['targetId'].tolist()}"

    if len(target_pairs_df) == 0:
        raise ValueError("No valid drug-target pairs after filtering. Cannot evaluate.")

    print(f"Evaluating {len(target_pairs_df)} drug-target pairs")
    print(f"  Unique drugs: {target_pairs_df['smiles'].nunique()}")
    print(f"  Unique targets: {target_pairs_df['targetId'].nunique()}")

    # Get representation columns
    drug_repr_cols = [col for col in drug_repr_df.columns if col != 'smiles']
    target_repr_cols = [col for col in target_repr_df.columns if col != 'targetId']

    # Build ground truth mapping: drug -> set of targets, target -> set of drugs
    drug_to_targets = defaultdict(set)
    target_to_drugs = defaultdict(set)

    for _, row in target_pairs_df.iterrows():
        drug_to_targets[row['smiles']].add(row['targetId'])
        target_to_drugs[row['targetId']].add(row['smiles'])

    # Extract representation matrices
    drug_smiles_list = drug_repr_df['smiles'].tolist()
    target_id_list = target_repr_df['targetId'].tolist()

    X_drugs = drug_repr_df[drug_repr_cols].values.astype(np.float32)
    X_targets = target_repr_df[target_repr_cols].values.astype(np.float32)

    # Create index mappings
    drug_to_idx = {smiles: idx for idx, smiles in enumerate(drug_smiles_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_id_list)}

    print(f"\nRepresentation dimensions:")
    print(f"  Drugs: {X_drugs.shape}")
    print(f"  Targets: {X_targets.shape}")

    # ========================================================================
    # Part 1: Drug -> Target Retrieval Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("COMPUTING DRUG -> TARGET RETRIEVAL METRICS")
    print("="*80)

    drug_target_results = _compute_retrieval_metrics(
        query_entities=list(drug_to_targets.keys()),
        query_repr=X_drugs,
        query_to_idx=drug_to_idx,
        gallery_repr=X_targets,
        gallery_entities=target_id_list,
        ground_truth_map=drug_to_targets,
        k_list=k_list,
        direction="drug_to_target"
    )

    # ========================================================================
    # Part 2: Target -> Drug Retrieval Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("COMPUTING TARGET -> DRUG RETRIEVAL METRICS")
    print("="*80)

    target_drug_results = _compute_retrieval_metrics(
        query_entities=list(target_to_drugs.keys()),
        query_repr=X_targets,
        query_to_idx=target_to_idx,
        gallery_repr=X_drugs,
        gallery_entities=drug_smiles_list,
        ground_truth_map=target_to_drugs,
        k_list=k_list,
        direction="target_to_drug"
    )

    # ========================================================================
    # Part 3: Graph-Based Metrics (AUROC, AUPRC)
    # ========================================================================
    print("\n" + "="*80)
    print("COMPUTING GRAPH-BASED METRICS")
    print("="*80)

    graph_results = _compute_graph_metrics(
        X_drugs=X_drugs,
        X_targets=X_targets,
        drug_smiles_list=drug_smiles_list,
        target_id_list=target_id_list,
        target_pairs_df=target_pairs_df
    )

    # Combine all results
    results = {**drug_target_results, **target_drug_results, **graph_results}

    return results


def _compute_retrieval_metrics(query_entities, query_repr, query_to_idx,
                               gallery_repr, gallery_entities, ground_truth_map,
                               k_list, direction):
    """Helper function to compute retrieval metrics for one direction"""

    results = {}
    all_precisions = {k: [] for k in k_list}
    all_recalls = {k: [] for k in k_list}
    all_mrrs = {k: [] for k in k_list}

    # Only evaluate queries that have ground truth positives
    valid_queries = [q for q in query_entities if q in ground_truth_map and len(ground_truth_map[q]) > 0]

    if len(valid_queries) == 0:
        print(f"No valid queries for {direction}")
        return {f'{direction}_queries': 0}

    print(f"Evaluating {len(valid_queries)} queries")

    # Build nearest neighbor index
    max_k = min(max(k_list), len(gallery_entities))
    nn = NearestNeighbors(n_neighbors=max_k, metric='euclidean', algorithm='brute')
    nn.fit(gallery_repr)

    # Query each entity
    for query in valid_queries:
        if query not in query_to_idx:
            continue

        query_idx = query_to_idx[query]
        query_vec = query_repr[query_idx:query_idx+1]

        # Find k-nearest neighbors
        distances, indices = nn.kneighbors(query_vec)
        neighbor_entities = [gallery_entities[idx] for idx in indices[0]]

        # Get ground truth positives for this query
        true_positives = ground_truth_map[query]

        # Compute metrics for each k
        for k in k_list:
            k_eff = min(k, len(neighbor_entities))
            top_k = neighbor_entities[:k_eff]

            # Count true positives in top-k
            tp_in_top_k = sum(1 for entity in top_k if entity in true_positives)

            # Precision@k: fraction of top-k that are true positives
            precision = tp_in_top_k / k_eff if k_eff > 0 else 0.0
            all_precisions[k].append(precision)

            # Recall@k (Hits@k): did we find at least one true positive?
            recall = 1.0 if tp_in_top_k > 0 else 0.0
            all_recalls[k].append(recall)

            # MRR@k: reciprocal rank of first true positive
            first_match_rank = None
            for rank, entity in enumerate(top_k, start=1):
                if entity in true_positives:
                    first_match_rank = rank
                    break

            mrr = 1.0 / first_match_rank if first_match_rank else 0.0
            all_mrrs[k].append(mrr)

    # Aggregate metrics
    for k in k_list:
        results[f'{direction}_precision@{k}'] = float(np.mean(all_precisions[k])) if all_precisions[k] else 0.0
        results[f'{direction}_recall@{k}'] = float(np.mean(all_recalls[k])) if all_recalls[k] else 0.0
        results[f'{direction}_mrr@{k}'] = float(np.mean(all_mrrs[k])) if all_mrrs[k] else 0.0

    results[f'{direction}_queries'] = len(valid_queries)

    return results


def _compute_graph_metrics(X_drugs, X_targets, drug_smiles_list, target_id_list, target_pairs_df):
    """Helper function to compute graph-based metrics (AUROC, AUPRC)"""

    results = {}

    # Compute all pairwise distances between drugs and targets
    print("Computing pairwise distance matrix...")
    n_drugs = len(drug_smiles_list)
    n_targets = len(target_id_list)

    # Compute distances in batches to avoid memory issues
    distances = np.zeros((n_drugs, n_targets), dtype=np.float32)
    batch_size = 1000

    for i in range(0, n_drugs, batch_size):
        end_i = min(i + batch_size, n_drugs)
        batch_drugs = X_drugs[i:end_i]

        # Compute distances: ||drug - target||_2
        for j in range(0, n_targets, batch_size):
            end_j = min(j + batch_size, n_targets)
            batch_targets = X_targets[j:end_j]

            # Broadcasting: (batch_drugs, 1, dim) - (1, batch_targets, dim)
            diff = batch_drugs[:, np.newaxis, :] - batch_targets[np.newaxis, :, :]
            batch_distances = np.linalg.norm(diff, axis=2)
            distances[i:end_i, j:end_j] = batch_distances

    print(f"Distance matrix shape: {distances.shape}")

    # Create ground truth adjacency matrix
    print("Building ground truth adjacency matrix...")
    y_true = np.zeros((n_drugs, n_targets), dtype=np.int32)

    drug_to_idx = {smiles: idx for idx, smiles in enumerate(drug_smiles_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_id_list)}

    for _, row in target_pairs_df.iterrows():
        drug_idx = drug_to_idx.get(row['smiles'])
        target_idx = target_to_idx.get(row['targetId'])

        if drug_idx is not None and target_idx is not None:
            y_true[drug_idx, target_idx] = 1

    n_positives = y_true.sum()
    n_total = y_true.size
    print(f"Ground truth: {n_positives} positives out of {n_total} pairs ({n_positives/n_total*100:.3f}%)")

    # Flatten for sklearn metrics (use negative distances as scores - closer = higher score)
    y_true_flat = y_true.flatten()
    y_scores = -distances.flatten()  # Negative because smaller distance = more likely positive

    # Compute AUROC and AUPRC
    print("Computing AUROC and AUPRC...")
    auroc = roc_auc_score(y_true_flat, y_scores)
    auprc = average_precision_score(y_true_flat, y_scores)

    results['auroc'] = float(auroc)
    results['auprc'] = float(auprc)
    results['n_positives'] = int(n_positives)
    results['n_total_pairs'] = int(n_total)

    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")

    return results


def submit_drug_target_mapping(drug_repr_df, target_repr_df, k_list=[1, 5, 10, 50], mode='lincs'):
    """Submits drug and target representations for evaluation against OpenTargets drug-target pairs

    Args:
        drug_repr_df: DataFrame with 'smiles' column + representation columns
        target_repr_df: DataFrame with 'targetId' column + representation columns
        k_list: List of k values for computing retrieval metrics
        mode: Evaluation mode - 'lincs' uses LINCS-filtered subset (default), 'full' uses all OpenTargets pairs

    Returns:
        dict: Evaluation metrics
    """
    # Load drug-target pairs ground truth
    if mode == 'lincs':
        ref_filename = 'drug_target_pairs_lincs.csv'
    elif mode == 'full':
        ref_filename = 'drug_target_pairs.csv'
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'lincs' or 'full'")

    pairs_path = os.path.join(DATA_DIR, 'opentargets/drug_target_pairs_csv', ref_filename)
    target_pairs_df = pd.read_csv(pairs_path)

    # Keep only relevant columns
    target_pairs_df = target_pairs_df[['smiles', 'targetId']].drop_duplicates()

    print(f"\nGround truth data (mode={mode}): {len(target_pairs_df)} drug-target pairs")
    print(f"  Drugs: {target_pairs_df['smiles'].nunique()}")
    print(f"  Targets: {target_pairs_df['targetId'].nunique()}")

    # Check overlap
    pred_drugs = set(drug_repr_df['smiles'].values)
    pred_targets = set(target_repr_df['targetId'].values)
    ref_drugs = set(target_pairs_df['smiles'].values)
    ref_targets = set(target_pairs_df['targetId'].values)

    drug_overlap = pred_drugs & ref_drugs
    target_overlap = pred_targets & ref_targets

    print(f"\nPrediction data:")
    print(f"  Drugs: {len(pred_drugs)} (overlap: {len(drug_overlap)}, {len(drug_overlap)/len(pred_drugs)*100:.1f}%)")
    print(f"  Targets: {len(pred_targets)} (overlap: {len(target_overlap)}, {len(target_overlap)/len(pred_targets)*100:.1f}%)")

    if len(drug_overlap) == 0:
        raise ValueError("No overlap between prediction drugs and reference drugs. Cannot evaluate.")

    if len(target_overlap) == 0:
        raise ValueError("No overlap between prediction targets and reference targets. Cannot evaluate.")

    # Filter to only drugs/targets in reference
    drug_repr_df_filtered = drug_repr_df[drug_repr_df['smiles'].isin(ref_drugs)].copy()
    target_repr_df_filtered = target_repr_df[target_repr_df['targetId'].isin(ref_targets)].copy()

    # Run evaluation
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    results = evaluate_drug_target_mapping(
        drug_repr_df_filtered,
        target_repr_df_filtered,
        target_pairs_df,
        k_list=k_list
    )

    # Print results
    print("\n" + "="*80)
    print("FINAL METRICS")
    print("="*80)

    print(f"\nDrug -> Target Retrieval ({results.get('drug_to_target_queries', 0)} queries):")
    for k in k_list:
        print(f"  k={k}:")
        print(f"    Precision@{k}: {results.get(f'drug_to_target_precision@{k}', 0):.4f}")
        print(f"    Recall@{k}:    {results.get(f'drug_to_target_recall@{k}', 0):.4f}")
        print(f"    MRR@{k}:       {results.get(f'drug_to_target_mrr@{k}', 0):.4f}")

    print(f"\nTarget -> Drug Retrieval ({results.get('target_to_drug_queries', 0)} queries):")
    for k in k_list:
        print(f"  k={k}:")
        print(f"    Precision@{k}: {results.get(f'target_to_drug_precision@{k}', 0):.4f}")
        print(f"    Recall@{k}:    {results.get(f'target_to_drug_recall@{k}', 0):.4f}")
        print(f"    MRR@{k}:       {results.get(f'target_to_drug_mrr@{k}', 0):.4f}")

    print(f"\nGraph-Based Metrics:")
    print(f"  AUROC: {results.get('auroc', 0):.4f}")
    print(f"  AUPRC: {results.get('auprc', 0):.4f}")
    print(f"  Positives: {results.get('n_positives', 0):,} / {results.get('n_total_pairs', 0):,}")

    print("\n" + "="*80)

    return results
