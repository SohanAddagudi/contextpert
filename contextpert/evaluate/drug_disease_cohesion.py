import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings

from contextpert.utils import canonicalize_smiles

DATA_DIR = os.environ['CONTEXTPERT_DATA_DIR']


def evaluate_drug_disease_cohesion(pred_df, target_df, k_list=[1, 5, 10, 50]):
    """Evaluates a small molecule representation in terms of how well it captures disease labels

    Uses leave-one-target-signature-out cross-validation: for each drug, hold out all samples with the
    same target signature (combination of all targets a drug hits), then compute top-k accuracy by
    checking if k-nearest neighbors share the same disease label.

    Args:
        pred_df: A dataframe with one 'smiles' column, remaining columns are the representation
        target_df: A dataframe with columns 'smiles', 'targets', and 'diseaseId'.
        k_list: List of k values for computing precision@k, hits@k, and MRR@k

    Returns:
        dict: Metrics including precision@k (micro/macro), hits@k, and MRR@k for each k
    """
    # Canonicalize SMILES before merging
    pred_df = pred_df.copy()
    target_df = target_df.copy()
    pred_df['smiles'] = pred_df['smiles'].apply(canonicalize_smiles)
    target_df['smiles'] = target_df['smiles'].apply(canonicalize_smiles)

    # Ensure all smiles from target_df are in pred_df.
    missing_smiles = target_df[~target_df['smiles'].isin(pred_df['smiles'])]
    assert missing_smiles.empty, f"Failure: {len(missing_smiles)} SMILES in target_df are missing from pred_df: {missing_smiles['smiles'].tolist()}"

    # Get representation columns (all columns except 'smiles')
    repr_cols = [col for col in pred_df.columns if col != 'smiles']

    # Merge pred_df with target_df to get disease and target signature labels
    merged = pred_df.merge(target_df, on='smiles', how='inner')

    if len(merged) == 0:
        raise ValueError("No matching SMILES between pred_df and target_df")

    print(f"Evaluating {len(merged)} drug indications. {merged['smiles'].nunique()} unique drugs, {merged['targets'].nunique()} unique target signatures, {merged['diseaseId'].nunique()} unique diseases")

    # Extract representation matrix
    X = merged[repr_cols].values.astype(np.float32)
    smiles_arr = merged['smiles'].values
    target_sig_arr = merged['targets'].values
    disease_arr = merged['diseaseId'].values

    # Store results for each query
    all_hits = {k: [] for k in k_list}
    all_precisions = {k: [] for k in k_list}
    all_reciprocal_ranks = {k: [] for k in k_list}
    per_disease_results = defaultdict(lambda: {k: [] for k in k_list})

    # Leave-one-target-signature-out evaluation
    unique_target_sigs = np.unique(target_sig_arr)

    for target_sig in unique_target_sigs:
        # Get all drugs with this target signature
        target_mask = target_sig_arr == target_sig
        target_indices = np.where(target_mask)[0]

        if len(target_indices) == 0:
            continue

        # Hold out drugs with this target signature as queries
        gallery_mask = ~target_mask
        gallery_indices = np.where(gallery_mask)[0]

        if len(gallery_indices) < max(k_list):
            # Not enough gallery samples for evaluation
            continue

        X_query = X[target_indices]
        X_gallery = X[gallery_indices]
        y_query = disease_arr[target_indices]
        y_gallery = disease_arr[gallery_indices]

        # Find k-nearest neighbors for each query
        max_k = min(max(k_list), len(X_gallery))
        nn = NearestNeighbors(n_neighbors=max_k, metric='euclidean', algorithm='brute')
        nn.fit(X_gallery)
        distances, indices = nn.kneighbors(X_query)

        # Evaluate each query
        for i, query_idx in enumerate(target_indices):
            query_disease = y_query[i]
            neighbor_diseases = y_gallery[indices[i]]

            # Compute metrics for each k
            for k in k_list:
                k_eff = min(k, len(neighbor_diseases))
                top_k_diseases = neighbor_diseases[:k_eff]

                # Hits@k: did we find the disease in top k?
                hit = int(query_disease in top_k_diseases)
                all_hits[k].append(hit)
                per_disease_results[query_disease][k].append(hit)

                # Precision@k: fraction of top-k that match query disease
                precision = np.mean(top_k_diseases == query_disease)
                all_precisions[k].append(precision)

                # MRR@k: reciprocal rank of first correct match
                matches = np.where(top_k_diseases == query_disease)[0]
                if len(matches) > 0:
                    rank = matches[0] + 1  # 1-indexed
                    rr = 1.0 / rank
                else:
                    rr = 0.0
                all_reciprocal_ranks[k].append(rr)

    # Aggregate metrics
    results = {}

    for k in k_list:
        if len(all_hits[k]) == 0:
            results[f'hits@{k}'] = 0.0
            results[f'precision@{k}_micro'] = 0.0
            results[f'precision@{k}_macro'] = 0.0
            results[f'mrr@{k}'] = 0.0
        else:
            # Micro-averaged metrics (average over all queries)
            results[f'hits@{k}'] = float(np.mean(all_hits[k]))
            results[f'precision@{k}_micro'] = float(np.mean(all_precisions[k]))
            results[f'mrr@{k}'] = float(np.mean(all_reciprocal_ranks[k]))

            # Macro-averaged metrics (average over diseases)
            disease_avgs = []
            for disease, hits in per_disease_results.items():
                if len(hits[k]) > 0:
                    disease_avgs.append(np.mean(hits[k]))
            results[f'precision@{k}_macro'] = float(np.mean(disease_avgs)) if disease_avgs else 0.0

    results['n_queries'] = len(all_hits[k_list[0]])
    results['n_unique_diseases'] = len(per_disease_results)

    return results

def submit_drug_disease_cohesion(pred_df, mode='lincs'):
    """Submits predictions against the OpenTargets disease-drug triples dataset

    Args:
        pred_df: A dataframe with one 'smiles' column, remaining columns are the representation
        mode: Evaluation mode - 'full' uses all OpenTargets drugs, 'lincs' filters to LINCS overlap

    Returns:
        dict: Evaluation metrics
    """
    # Load the disease-drug triples with target signatures
    # Only includes diseases with 2+ unique target signatures for valid evaluation
    if mode == 'lincs':
        ref_filename = 'disease_drug_triples_lincs.csv'
    elif mode == 'full':
        ref_filename = 'disease_drug_triples.csv'
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full' or 'lincs'")

    ref_path = os.path.join(DATA_DIR, 'opentargets/disease_drug_triples_csv', ref_filename)
    disease_drug_df = pd.read_csv(ref_path)

    # Prepare target_df format expected by evaluate function
    # Columns: smiles, targets, diseaseId
    target_df = disease_drug_df[['smiles', 'targets', 'diseaseId']].drop_duplicates()

    print(f"\nReference data (mode={mode}): {len(target_df)} unique drug-disease associations")
    print(f"  Diseases: {target_df['diseaseId'].nunique()}")
    print(f"  Drugs: {target_df['smiles'].nunique()}")
    print(f"  Target signatures: {target_df['targets'].nunique()}")

    # Check overlap
    pred_drugs = set(pred_df['smiles'].values)
    ref_drugs = set(target_df['smiles'].values)
    overlap = pred_drugs & ref_drugs

    print(f"\nPrediction data: {len(pred_drugs)} unique drugs")
    print(f"Overlap with reference: {len(overlap)} drugs ({len(overlap)/len(pred_drugs)*100:.1f}%)")

    if len(overlap) == 0:
        raise ValueError("No overlap between prediction SMILES and reference SMILES. Cannot evaluate.")

    # Filter pred_df to only drugs in reference
    pred_df_filtered = pred_df[pred_df['smiles'].isin(ref_drugs)].copy()

    if len(pred_df_filtered) < 10:
        warnings.warn(f"Only {len(pred_df_filtered)} drugs available for evaluation. Results may not be reliable.")

    # Run evaluation
    print("\n" + "="*80)
    print("EVALUATION RESULTS (Leave-One-Target-Signature-Out)")
    print("="*80)

    k_list = [1, 5, 10, 50]
    results = evaluate_drug_disease_cohesion(pred_df_filtered, target_df, k_list=k_list)

    # Print results
    print(f"\nEvaluated {results['n_queries']} queries across {results['n_unique_diseases']} diseases")
    print("\nMetrics:")
    for k in k_list:
        print(f"\n  k={k}:")
        print(f"    Hits@{k}:         {results[f'hits@{k}']:.4f}")
        print(f"    Precision@{k} (micro): {results[f'precision@{k}_micro']:.4f}")
        print(f"    Precision@{k} (macro): {results[f'precision@{k}_macro']:.4f}")
        print(f"    MRR@{k}:          {results[f'mrr@{k}']:.4f}")

    print("\n" + "="*80)

    return results
