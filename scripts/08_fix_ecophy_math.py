#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.tree import nj
from skbio.stats.ordination import pcoa

BASE_DIR = os.path.expanduser("~/ecophy_project")
EXPORT_DIR = f"{BASE_DIR}/results/local_bundle_v3"

def compute_unsupervised_eco_weights(Z_c, RM_c, gids, kmer_info):
    """
    Computes two strictly unsupervised (label-free) quantities for each of the 136 canonical 4-mers:
    1. R-M Offset Matrix: shifts R-M(+) and R-M(-) contigs toward the shared community baseline (not zero!).
    2. Ecological & Intra-Genomic Instability Score (E_j): combines intra-genome spatial variance across
       50kb contigs with phage avoidance and R-M polymorphism.
    """
    n_samples, n_features = Z_c.shape
    s_phage = kmer_info["phage_avoidance_S"].values
    is_pal = kmer_info["is_strict_palindrome"].values

    # 1. Unsupervised Intra-Genome Contig Variance Ratio (Var_within_genome / Var_total)
    total_var = np.var(Z_c, axis=0) + 1e-8
    unique_gids = np.unique(gids)
    within_vars = []
    for g in unique_gids:
        sub = Z_c[gids == g]
        if len(sub) > 1:
            within_vars.append(np.var(sub, axis=0))
    mean_within_var = np.mean(within_vars, axis=0)
    spatial_instability = mean_within_var / total_var  # High = noisy/mobile across chromosome

    # 2. Unsupervised R-M Offset & R-M Variance Coupling
    rm_offsets = np.zeros_like(Z_c)
    rm_coupling = np.zeros(n_features)

    for j in range(n_features):
        col = Z_c[:, j]
        mu_all = np.mean(col)
        rm_col = RM_c[:, j]
        n_pos = np.sum(rm_col == 1)
        n_neg = np.sum(rm_col == 0)

        if is_pal[j] and n_pos >= 5 and n_neg >= 5:
            mu_pos = np.mean(col[rm_col == 1])
            mu_neg = np.mean(col[rm_col == 0])
            # Store each sample's R-M group deviation from the community mean
            rm_offsets[rm_col == 1, j] = mu_pos - mu_all
            rm_offsets[rm_col == 0, j] = mu_neg - mu_all
            rm_coupling[j] = abs(mu_pos - mu_neg) / (np.std(col) + 1e-8)

    # 3. Combined Unsupervised Ecological Disturbance Score E_j
    # Normalizes phage avoidance, R-M coupling, and spatial contig instability
    s_norm = s_phage / (np.max(s_phage) + 1e-8)
    rm_norm = rm_coupling / (np.max(rm_coupling) + 1e-8)
    instab_norm = spatial_instability / (np.max(spatial_instability) + 1e-8)

    # Motifs with high spatial instability AND phage/R-M pressure get the highest demotion score
    E_score = instab_norm * (1.0 + 1.5 * s_norm + 1.5 * rm_norm)
    # Preserve the top 50% most stable housekeeping 4-mers completely untouched (E_score = 0)
    median_cutoff = np.median(E_score)
    E_score = np.where(E_score > median_cutoff, E_score - median_cutoff, 0.0)
    E_score = E_score / (np.max(E_score) + 1e-8)

    return rm_offsets, E_score, spatial_instability, rm_coupling

def apply_centered_eco_transform(Z_mat, rm_offsets, E_score, lam):
    """
    1. Removes the R-M group offset toward the community mean (not 0.0!).
    2. Standardizes features around the community mean.
    3. Weights features by w_j = exp(-lam * E_score).
    """
    alpha = np.tanh(lam * 0.8)
    Z_deconf = Z_mat - alpha * rm_offsets
    Z_centered = (Z_deconf - np.mean(Z_deconf, axis=0)) / (np.std(Z_deconf, axis=0) + 1e-8)
    weights = np.exp(-lam * E_score)
    return Z_centered * np.sqrt(weights), weights

def evaluate_cross_species_and_genome(dist_mat, genera, species, gids, k=5):
    """
    Evaluates:
    1. Cross-Species Genus k-NN Accuracy (masks same-species clones so classification tests true Genus phylogeny).
    2. Cross-Genome Genus k-NN Accuracy (masks only same-assembly contigs).
    3. Inter- / Intra-Genus Separation Ratio.
    """
    n = len(genera)
    correct_cross_sp = 0
    valid_cross_sp = 0
    correct_cross_g = 0
    intra, inter = [], []

    for i in range(n):
        # A. Cross-Genome evaluation
        d_g = dist_mat[i].copy()
        for j in range(n):
            if gids[i] == gids[j]:
                d_g[j] = np.inf
            else:
                (intra if genera[i] == genera[j] else inter).append(dist_mat[i, j])
        votes_g = [genera[j] for j in np.argsort(d_g)[:k]]
        if max(set(votes_g), key=votes_g.count) == genera[i]:
            correct_cross_g += 1

        # B. Cross-Species evaluation (masks out identical species clones)
        d_sp = dist_mat[i].copy()
        same_genus_other_sp = 0
        for j in range(n):
            if species[i] == species[j]:
                d_sp[j] = np.inf
            elif genera[i] == genera[j]:
                same_genus_other_sp += 1

        if same_genus_other_sp >= k:
            valid_cross_sp += 1
            votes_sp = [genera[j] for j in np.argsort(d_sp)[:k]]
            if max(set(votes_sp), key=votes_sp.count) == genera[i]:
                correct_cross_sp += 1

    acc_g = (correct_cross_g / n) * 100.0
    acc_sp = (correct_cross_sp / max(1, valid_cross_sp)) * 100.0
    ratio = float(np.mean(inter) / np.mean(intra))
    return acc_sp, acc_g, ratio

def main():
    print("[1/3] Loading local_bundle_v3 tables...")
    df_c = pd.read_csv(f"{EXPORT_DIR}/contig_metadata.tsv", sep="\t")
    Z_c = pd.read_csv(f"{EXPORT_DIR}/contig_markov_z.tsv", sep="\t", index_col=0).values
    RM_c = pd.read_csv(f"{EXPORT_DIR}/contig_rm_mask.tsv", sep="\t", index_col=0).values
    kmer_info = pd.read_csv(f"{EXPORT_DIR}/kmer_phage_info.tsv", sep="\t")

    # Extract clean Species name (e.g. "B. fragilis") from display_label
    df_c["species"] = df_c["display_label"].apply(lambda s: s.split(" (")[0])
    genera = df_c["genus"].tolist()
    species = df_c["species"].tolist()
    gids = df_c["genome_id"].values

    print("[2/3] Computing Unsupervised R-M Offsets & Spatial Instability Scores...")
    rm_offsets, E_score, spatial_instab, rm_coupling = compute_unsupervised_eco_weights(
        Z_c, RM_c, gids, kmer_info
    )

    print("[3/3] Sweeping Lambda (0.00 to 3.00)...")
    lambdas = np.linspace(0.0, 3.0, 61)
    records = []

    for lam in lambdas:
        Z_trans, _ = apply_centered_eco_transform(Z_c, rm_offsets, E_score, lam)
        D = squareform(pdist(Z_trans, metric="euclidean"))
        acc_sp, acc_g, ratio = evaluate_cross_species_and_genome(D, genera, species, gids, k=5)
        records.append({
            "lambda": round(float(lam), 3),
            "cross_species_knn_acc": round(acc_sp, 3),
            "cross_genome_knn_acc": round(acc_g, 3),
            "clade_separation_ratio": round(ratio, 5)
        })

    sweep_df = pd.DataFrame(records)
    sweep_df.to_csv(f"{EXPORT_DIR}/lambda_sweep_results.tsv", sep="\t", index=False)

    base = sweep_df.iloc[0]
    # Pick optimal lambda maximizing Cross-Species Accuracy, breaking ties by Clade Separation Ratio
    best_row = sweep_df.sort_values(
        by=["cross_species_knn_acc", "clade_separation_ratio"], ascending=[False, False]
    ).iloc[0]
    best_lam = float(best_row["lambda"])

    print("==========================================================================")
    print(f"  UNWEIGHTED BASELINE (lambda = 0.00):")
    print(f"    - Cross-Species 5-NN Genus Accuracy: {base['cross_species_knn_acc']:.2f}%")
    print(f"    - Cross-Genome  5-NN Genus Accuracy: {base['cross_genome_knn_acc']:.2f}%")
    print(f"    - Inter/Intra-Genus Distance Ratio:  {base['clade_separation_ratio']:.4f}")
    print("--------------------------------------------------------------------------")
    print(f"  OPTIMAL ECO-WEIGHTED (lambda = {best_lam:.2f}):")
    print(f"    - Cross-Species 5-NN Genus Accuracy: {best_row['cross_species_knn_acc']:.2f}% "
          f"(+{best_row['cross_species_knn_acc'] - base['cross_species_knn_acc']:.2f}%)")
    print(f"    - Cross-Genome  5-NN Genus Accuracy: {best_row['cross_genome_knn_acc']:.2f}% "
          f"(+{best_row['cross_genome_knn_acc'] - base['cross_genome_knn_acc']:.2f}%)")
    print(f"    - Inter/Intra-Genus Distance Ratio:  {best_row['clade_separation_ratio']:.4f} "
          f"(+{((best_row['clade_separation_ratio']/base['clade_separation_ratio'])-1)*100:.2f}%)")
    print("==========================================================================")

    # Save updated weights and PCoA coordinates at best_lam
    Z_unw, _ = apply_centered_eco_transform(Z_c, rm_offsets, E_score, 0.0)
    Z_opt, w_opt = apply_centered_eco_transform(Z_c, rm_offsets, E_score, best_lam)

    kmer_info["spatial_instability"] = np.round(spatial_instab, 4)
    kmer_info["rm_coupling"] = np.round(rm_coupling, 4)
    kmer_info["eco_disturbance_E"] = np.round(E_score, 4)
    kmer_info["optimal_weight"] = np.round(w_opt, 4)
    kmer_info.to_csv(f"{EXPORT_DIR}/kmer_phage_info.tsv", sep="\t", index=False)

    D_unw = squareform(pdist(Z_unw, metric="euclidean"))
    D_opt = squareform(pdist(Z_opt, metric="euclidean"))
    ids = df_c["sample_id"].tolist()

    pcoa_unw = pcoa(DistanceMatrix(D_unw, ids))
    pcoa_opt = pcoa(DistanceMatrix(D_opt, ids))

    df_c["PC1_unw"] = pcoa_unw.samples["PC1"].values
    df_c["PC2_unw"] = pcoa_unw.samples["PC2"].values
    df_c["PC1_eco"] = pcoa_opt.samples["PC1"].values
    df_c["PC2_eco"] = pcoa_opt.samples["PC2"].values
    df_c["PC1_unw_var"] = round(float(pcoa_unw.proportion_explained["PC1"] * 100), 2)
    df_c["PC2_unw_var"] = round(float(pcoa_unw.proportion_explained["PC2"] * 100), 2)
    df_c["PC1_eco_var"] = round(float(pcoa_opt.proportion_explained["PC1"] * 100), 2)
    df_c["PC2_eco_var"] = round(float(pcoa_opt.proportion_explained["PC2"] * 100), 2)
    df_c.to_csv(f"{EXPORT_DIR}/contig_pcoa_coordinates.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
