#!/usr/bin/env python3
import os
import glob
import itertools
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.tree import nj
from skbio.stats.ordination import pcoa
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/ecophy_project")
RESULTS_DIR = f"{BASE_DIR}/results"

BASES = ["A", "C", "G", "T"]
COMPLEMENT = str.maketrans("ACGT", "TGCA")
ALL_4MERS = ["".join(p) for p in itertools.product(BASES, repeat=4)]
ALL_3MERS = ["".join(p) for p in itertools.product(BASES, repeat=3)]
ALL_2MERS = ["".join(p) for p in itertools.product(BASES, repeat=2)]

PALINDROMES_4MER = [
    "AATT", "ACGT", "AGCT", "ATAT",
    "CATG", "CCGG", "CGCG", "CTAG",
    "GATC", "GCGC", "GGCC", "GTAC",
    "TATA", "TCGA", "TGCA", "TTAA"
]

def rev_comp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def count_symmetrized_kmers(fna_path: str):
    """Counts 2-mers, 3-mers, and 4-mers across both strands (forward + reverse complement)."""
    c2 = {k: 1.0 for k in ALL_2MERS}  # Laplace pseudocount = 1.0
    c3 = {k: 1.0 for k in ALL_3MERS}
    c4 = {k: 1.0 for k in ALL_4MERS}

    for record in SeqIO.parse(fna_path, "fasta"):
        seq_fwd = str(record.seq).upper()
        for seq in (seq_fwd, rev_comp(seq_fwd)):
            n = len(seq)
            for i in range(n - 3):
                w4 = seq[i:i+4]
                if w4 in c4:
                    c4[w4] += 1.0
                    c3[w4[:3]] += 1.0
                    c2[w4[:2]] += 1.0
    return c2, c3, c4

def compute_entropy_and_markov_z(fna_path: str):
    """Computes H4, conditional entropy h4 = H4 - H3, and 2nd-order Markov Z-scores."""
    c2, c3, c4 = count_symmetrized_kmers(fna_path)

    total_4 = sum(c4.values())
    total_3 = sum(c3.values())
    p4 = np.array([c4[k] / total_4 for k in ALL_4MERS])
    p3 = np.array([c3[k] / total_3 for k in ALL_3MERS])

    # Shannon entropies (in bits)
    H4 = -np.sum(p4 * np.log2(p4))
    H3 = -np.sum(p3 * np.log2(p3))
    h4_cond = H4 - H3

    # Second-order Markov Z-score (Karlin & Burge / Schbath estimator)
    z_scores = np.zeros(len(ALL_4MERS))
    for idx, w in enumerate(ALL_4MERS):
        w123 = w[:3]
        w234 = w[1:]
        w23 = w[1:3]
        w12 = w[:2]

        n_w = c4[w]
        n_123 = c3[w123]
        n_234 = c3[w234]
        n_23 = c2[w23]
        n_12 = c2[w12]

        exp_w = (n_123 * n_234) / n_23
        var_w = exp_w * max(1e-6, (1.0 - n_123 / n_12)) * max(1e-6, (1.0 - n_234 / n_23))
        z_scores[idx] = (n_w - exp_w) / np.sqrt(var_w)

    return H4, h4_cond, z_scores

def clade_separation_ratio(dist_matrix: np.ndarray, clades: list) -> float:
    """Computes ratio of mean inter-phylum distance to mean intra-phylum distance."""
    intra, inter = [], []
    n = len(clades)
    for i in range(n):
        for j in range(i + 1, n):
            if clades[i] == clades[j]:
                intra.append(dist_matrix[i, j])
            else:
                inter.append(dist_matrix[i, j])
    return float(np.mean(inter) / np.mean(intra))

def main():
    print("[1/5] Calculating Markov Z-scores for Phage Dataset...")
    phage_files = sorted(glob.glob(f"{BASE_DIR}/data/phages/fna/*.fna"))
    phage_z_list = []
    for pf in phage_files:
        _, _, z = compute_entropy_and_markov_z(pf)
        phage_z_list.append(z)
    phage_z_mat = np.vstack(phage_z_list)

    # Mean phage avoidance score S_phage(i) for depleted motifs (Z < 0)
    mean_phage_z = np.mean(phage_z_mat, axis=0)
    s_phage = {ALL_4MERS[i]: max(0.0, -mean_phage_z[i]) for i in range(len(ALL_4MERS))}

    print("      Top Phage-Avoided Palindromes (mean negative Markov Z-score):")
    pal_scores = sorted([(p, s_phage[p]) for p in PALINDROMES_4MER], key=lambda x: x[1], reverse=True)
    for p, score in pal_scores[:5]:
        print(f"        {p}: avoidance score = {score:.2f}")

    print("[2/5] Calculating Entropy and Markov Z-scores for Bacterial Genomes...")
    rm_df = pd.read_csv(f"{RESULTS_DIR}/rm_motif_presence.tsv", sep="\t")
    bact_records = []
    z_matrix = []

    for _, row in rm_df.iterrows():
        gid = row["genome_id"]
        clade = row["clade"]
        subdir = "entero" if clade == "Enterobacteriaceae" else "bacteroides"
        fna_path = f"{BASE_DIR}/data/{subdir}/fna/{gid}.fna"
        if not os.path.exists(fna_path):
            continue
        H4, h4, z = compute_entropy_and_markov_z(fna_path)
        z_matrix.append(z)
        bact_records.append({
            "genome_id": gid,
            "clade": clade,
            "H4_bits": H4,
            "h4_cond_bits": h4,
            "GATC_RM": row["GATC"],
            "total_RM_palindromes": sum(row[p] for p in PALINDROMES_4MER)
        })

    meta_df = pd.DataFrame(bact_records)
    meta_df.to_csv(f"{RESULTS_DIR}/bacterial_entropy_summary.tsv", sep="\t", index=False)
    Z_bact = np.vstack(z_matrix)
    n_genomes = Z_bact.shape[0]

    print("[3/5] Computing Unweighted vs. Eco-Weighted Distance Matrices...")
    # 1. Standard Unweighted Euclidean Distance on Markov Z-scores
    D_unweighted = squareform(pdist(Z_bact, metric="euclidean"))

    # 2. Eco-Weighted Distance Matrix
    # Demote 4-mers targeted by active R-M systems and avoided by phages
    lam = 0.85  # Ecological penalty hyperparameter
    rm_prevalence = {p: rm_df[p].mean() for p in PALINDROMES_4MER}

    weights_eco = np.ones(len(ALL_4MERS))
    for idx, kmer in enumerate(ALL_4MERS):
        if kmer in PALINDROMES_4MER and rm_prevalence[kmer] > 0:
            i_rm = 1.0
            weights_eco[idx] = np.exp(-lam * i_rm * s_phage[kmer])

    Z_weighted = Z_bact * np.sqrt(weights_eco)
    D_ecoweighted = squareform(pdist(Z_weighted, metric="euclidean"))

    # 3. Ablation Control (100 random permutations of non-palindromic weights)
    rng = np.random.default_rng(42)
    non_pal_indices = [i for i, k in enumerate(ALL_4MERS) if k not in PALINDROMES_4MER]
    active_demotions = [weights_eco[i] for i, k in enumerate(ALL_4MERS) if k in PALINDROMES_4MER and weights_eco[i] < 1.0]

    ablation_ratios = []
    for _ in range(100):
        w_rand = np.ones(len(ALL_4MERS))
        chosen = rng.choice(non_pal_indices, size=len(active_demotions), replace=False)
        for c_idx, val in zip(chosen, active_demotions):
            w_rand[c_idx] = val
        D_rand = squareform(pdist(Z_bact * np.sqrt(w_rand), metric="euclidean"))
        ablation_ratios.append(clade_separation_ratio(D_rand, meta_df["clade"].tolist()))

    ratio_unw = clade_separation_ratio(D_unweighted, meta_df["clade"].tolist())
    ratio_eco = clade_separation_ratio(D_ecoweighted, meta_df["clade"].tolist())
    p_val = np.mean([r >= ratio_eco for r in ablation_ratios])

    print(f"      Clade Separation Ratio (Unweighted):     {ratio_unw:.4f}")
    print(f"      Clade Separation Ratio (Eco-Weighted):   {ratio_eco:.4f}")
    print(f"      Random Ablation Mean +/- SD:             {np.mean(ablation_ratios):.4f} +/- {np.std(ablation_ratios):.4f} (p = {p_val:.3f})")

    print("[4/5] Building Neighbor-Joining Trees...")
    ids = meta_df["genome_id"].tolist()
    tree_unw = nj(DistanceMatrix(D_unweighted, ids))
    tree_eco = nj(DistanceMatrix(D_ecoweighted, ids))
    tree_unw.write(f"{RESULTS_DIR}/tree_unweighted.nwk")
    tree_eco.write(f"{RESULTS_DIR}/tree_ecoweighted.nwk")

    print("[5/5] Generating PCoA Comparison Plot...")
    pc_unw = pcoa(DistanceMatrix(D_unweighted, ids)).samples
    pc_eco = pcoa(DistanceMatrix(D_ecoweighted, ids)).samples

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, pc, title in zip(
        axes,
        [pc_unw, pc_eco],
        [f"Standard Unweighted 4-mer PCoA\n(Separation Ratio = {ratio_unw:.3f})",
         f"Eco-Weighted 4-mer PCoA\n(Separation Ratio = {ratio_eco:.3f}, p = {p_val:.3f})"]
    ):
        for clade, color in [("Enterobacteriaceae", "#1f77b4"), ("Bacteroidaceae", "#ff7f0e")]:
            for rm_flag, marker, label_suffix in [(1, "o", "R-M (+)"), (0, "^", "R-M (-)")]:
                sub = meta_df[(meta_df["clade"] == clade) & (meta_df["GATC_RM"] == rm_flag)]
                if sub.empty:
                    continue
                pts = pc.loc[sub["genome_id"]]
                ax.scatter(
                    pts["PC1"], pts["PC2"],
                    c=color, marker=marker, s=70, alpha=0.85,
                    edgecolors="black" if rm_flag == 1 else "none",
                    label=f"{clade} [{label_suffix}]"
                )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pcoa_comparison.png", dpi=300)
    print(f"Pipeline complete! Trees and figures saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
