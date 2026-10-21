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

BASE_DIR = os.path.expanduser("~/ecophy_project")
RESULTS_DIR = f"{BASE_DIR}/results"
EXPORT_DIR = f"{BASE_DIR}/results/local_bundle"
os.makedirs(EXPORT_DIR, exist_ok=True)

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

def get_species_label(fna_path: str, acc: str) -> str:
    """Extracts a clean Species + Strain label from the first FASTA header."""
    for record in SeqIO.parse(fna_path, "fasta"):
        desc = record.description
        parts = desc.split()
        if len(parts) >= 3:
            genus_sp = f"{parts[1][0]}. {parts[2]}"
            return f"{genus_sp} ({acc})"
        break
    return acc

def compute_markov_z(fna_path: str):
    c2 = {k: 1.0 for k in ALL_2MERS}
    c3 = {k: 1.0 for k in ALL_3MERS}
    c4 = {k: 1.0 for k in ALL_4MERS}

    for record in SeqIO.parse(fna_path, "fasta"):
        seq_fwd = str(record.seq).upper()
        for seq in (seq_fwd, rev_comp(seq_fwd)):
            for i in range(len(seq) - 3):
                w4 = seq[i:i+4]
                if w4 in c4:
                    c4[w4] += 1.0
                    c3[w4[:3]] += 1.0
                    c2[w4[:2]] += 1.0

    p4 = np.array([c4[k] / sum(c4.values()) for k in ALL_4MERS])
    p3 = np.array([c3[k] / sum(c3.values()) for k in ALL_3MERS])
    H4 = -np.sum(p4 * np.log2(p4))
    H3 = -np.sum(p3 * np.log2(p3))

    z_scores = np.zeros(len(ALL_4MERS))
    for idx, w in enumerate(ALL_4MERS):
        n_w, n_123, n_234 = c4[w], c3[w[:3]], c3[w[1:]]
        n_23, n_12 = c2[w[1:3]], c2[w[:2]]
        exp_w = (n_123 * n_234) / n_23
        var_w = exp_w * max(1e-6, (1.0 - n_123 / n_12)) * max(1e-6, (1.0 - n_234 / n_23))
        z_scores[idx] = (n_w - exp_w) / np.sqrt(var_w)

    return H4, (H4 - H3), z_scores

def main():
    print("[1/4] Exporting Phage Avoidance Profile...")
    phage_files = sorted(glob.glob(f"{BASE_DIR}/data/phages/fna/*.fna"))
    phage_z = np.vstack([compute_markov_z(pf)[2] for pf in phage_files])
    mean_phage_z = np.mean(phage_z, axis=0)
    sd_phage_z = np.std(phage_z, axis=0)
    s_phage = {ALL_4MERS[i]: max(0.0, -mean_phage_z[i]) for i in range(len(ALL_4MERS))}

    print("[2/4] Exporting Bacterial Z-Scores & Metadata...")
    rm_df = pd.read_csv(f"{RESULTS_DIR}/rm_motif_presence.tsv", sep="\t")
    meta_rows, z_rows = [], []

    for _, row in rm_df.iterrows():
        gid, clade = row["genome_id"], row["clade"]
        subdir = "entero" if clade == "Enterobacteriaceae" else "bacteroides"
        fna_path = f"{BASE_DIR}/data/{subdir}/fna/{gid}.fna"
        if not os.path.exists(fna_path):
            continue

        label = get_species_label(fna_path, gid)
        H4, h4, z = compute_markov_z(fna_path)
        z_rows.append(pd.Series(z, index=ALL_4MERS, name=gid))

        meta_entry = {
            "genome_id": gid,
            "display_label": label,
            "clade": clade,
            "H4_bits": round(H4, 5),
            "h4_cond_bits": round(h4, 5),
            "GATC_RM": int(row["GATC"]),
            "CCGG_RM": int(row["CCGG"]),
            "CTAG_RM": int(row["CTAG"]),
            "total_RM_motifs": int(sum(row[p] for p in PALINDROMES_4MER))
        }
        meta_rows.append(meta_entry)

    meta_df = pd.DataFrame(meta_rows)
    z_df = pd.DataFrame(z_rows)
    meta_df.to_csv(f"{EXPORT_DIR}/metadata_and_entropy.tsv", sep="\t", index=False)
    z_df.to_csv(f"{EXPORT_DIR}/bacterial_markov_z_scores.tsv", sep="\t")

    print("[3/4] Exporting Motif Weights & Distance Matrices...")
    lam = 0.85
    weights_eco = np.ones(len(ALL_4MERS))
    kmer_table = []

    for idx, kmer in enumerate(ALL_4MERS):
        is_pal = kmer in PALINDROMES_4MER
        rm_prev = float(rm_df[kmer].mean()) if is_pal else 0.0
        if is_pal and rm_prev > 0:
            weights_eco[idx] = np.exp(-lam * 1.0 * s_phage[kmer])
        kmer_table.append({
            "kmer": kmer,
            "is_palindrome": is_pal,
            "mean_phage_Z": round(float(mean_phage_z[idx]), 4),
            "sd_phage_Z": round(float(sd_phage_z[idx]), 4),
            "phage_avoidance_S": round(float(s_phage[kmer]), 4),
            "bacterial_RM_prevalence": round(rm_prev, 4),
            "eco_weight": round(float(weights_eco[idx]), 4)
        })

    pd.DataFrame(kmer_table).to_csv(f"{EXPORT_DIR}/kmer_weights_and_phage_avoidance.tsv", sep="\t", index=False)

    Z_mat = z_df.values
    ids = meta_df["genome_id"].tolist()
    D_unw = squareform(pdist(Z_mat, metric="euclidean"))
    D_eco = squareform(pdist(Z_mat * np.sqrt(weights_eco), metric="euclidean"))

    pd.DataFrame(D_unw, index=ids, columns=ids).to_csv(f"{EXPORT_DIR}/dist_unweighted.tsv", sep="\t")
    pd.DataFrame(D_eco, index=ids, columns=ids).to_csv(f"{EXPORT_DIR}/dist_ecoweighted.tsv", sep="\t")

    print("[4/4] Exporting PCoA Coordinates & Trees...")
    dm_unw = DistanceMatrix(D_unw, ids)
    dm_eco = DistanceMatrix(D_eco, ids)

    nj(dm_unw).write(f"{EXPORT_DIR}/tree_unweighted.nwk")
    nj(dm_eco).write(f"{EXPORT_DIR}/tree_ecoweighted.nwk")

    pcoa_unw = pcoa(dm_unw)
    pcoa_eco = pcoa(dm_eco)

    coords = meta_df.copy()
    coords["PC1_unw"] = pcoa_unw.samples["PC1"].values
    coords["PC2_unw"] = pcoa_unw.samples["PC2"].values
    coords["PC1_eco"] = pcoa_eco.samples["PC1"].values
    coords["PC2_eco"] = pcoa_eco.samples["PC2"].values
    coords["PC1_unw_var"] = round(float(pcoa_unw.proportion_explained["PC1"] * 100), 2)
    coords["PC2_unw_var"] = round(float(pcoa_unw.proportion_explained["PC2"] * 100), 2)
    coords["PC1_eco_var"] = round(float(pcoa_eco.proportion_explained["PC1"] * 100), 2)
    coords["PC2_eco_var"] = round(float(pcoa_eco.proportion_explained["PC2"] * 100), 2)
    coords.to_csv(f"{EXPORT_DIR}/pcoa_coordinates.tsv", sep="\t", index=False)

    print(f"Export complete! All lightweight files saved in {EXPORT_DIR}")

if __name__ == "__main__":
    main()
