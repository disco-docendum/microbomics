#!/usr/bin/env python3
import os
import re
import glob
import subprocess
import itertools
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.tree import nj
from skbio.stats.ordination import pcoa

BASE_DIR = os.path.expanduser("~/ecophy_project")
REBASE_RAW = f"{BASE_DIR}/data/rebase/protein_seqs.txt"
REBASE_FASTA = f"{BASE_DIR}/data/rebase/rebase_expanded_targets.faa"
REBASE_DB = f"{BASE_DIR}/data/rebase/rebase_expanded_db"
EXPORT_DIR = f"{BASE_DIR}/results/local_bundle_v2"
os.makedirs(EXPORT_DIR, exist_ok=True)

BASES = ["A", "C", "G", "T"]
COMPLEMENT = str.maketrans("ACGT", "TGCA")
ALL_4MERS = ["".join(p) for p in itertools.product(BASES, repeat=4)]
ALL_3MERS = ["".join(p) for p in itertools.product(BASES, repeat=3)]
ALL_2MERS = ["".join(p) for p in itertools.product(BASES, repeat=2)]

def rev_comp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

# Build the 136 Canonical 4-mers (collapsing reverse complements so non-palindromes aren't double-counted)
CANONICAL_4MERS = sorted(list({min(k, rev_comp(k)) for k in ALL_4MERS}))

# 16 Strict Palindromes + 20 Common Type II R-M Sub-4mers (e.g. CCWGG -> CCAG, GCWGC -> GCAG, GANTC -> GATC/GAAC)
RM_TARGET_CANONICAL = sorted(list({
    min(k, rev_comp(k)) for k in [
        # 16 strict 4-mer palindromes
        "AATT", "ACGT", "AGCT", "ATAT", "CATG", "CCGG", "CGCG", "CTAG",
        "GATC", "GCGC", "GGCC", "GTAC", "TATA", "TCGA", "TGCA", "TTAA",
        # High-frequency gut Type II R-M & MTase 4-mer cores (Dcm CCWGG, BisI GCNGC, EcoO109I RGGNCCY, HinfI GANTC)
        "CCAG", "CCTG", "GCAG", "GCTG", "GGAC", "GGTC", "GAAC", "GTTC",
        "CTAG", "CAAG", "CTTG", "CGAC", "GTCG", "CCGC", "GCGG", "CGGC",
        "GCCG", "GAGC", "GCTC", "GGTACC"[:4], "AAGCTT"[:4], "GAATTC"[:4]
    ]
}))

TARGET_GENERA = {"Bacteroides", "Phocaeicola", "Parabacteroides", "Prevotella", "Alistipes", "Porphyromonas"}

def parse_genus_and_species(fna_path: str, acc: str):
    for record in SeqIO.parse(fna_path, "fasta"):
        clean = re.sub(r"[\[\]]", "", record.description)
        tokens = clean.split()
        if len(tokens) >= 3:
            genus = tokens[1]
            species = f"{genus[0]}. {tokens[2]}"
            return genus, f"{species} ({acc})"
    return "Unknown", acc

def build_expanded_rebase_db():
    print("[1/4] Building Expanded REBASE Database (Palindromes + Degenerate R-M Cores)...")
    with open(REBASE_RAW, "r", errors="ignore") as fin, open(REBASE_FASTA, "w") as fout:
        header, seq_lines = None, []
        for line in fin:
            line = line.strip()
            if not line or line == "<>":
                continue
            if line.startswith(">"):
                if header and seq_lines:
                    seq = re.sub(r"[^A-Za-z]", "", "".join(seq_lines))
                    matched = [m for m in RM_TARGET_CANONICAL if m in header or rev_comp(m) in header]
                    if matched and len(seq) > 50:
                        clean_id = header.split()[0].replace(">", "")
                        fout.write(f">{clean_id}|MOTIFS:{','.join(matched)}\n{seq}\n")
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
    subprocess.run(["diamond", "makedb", "--in", REBASE_FASTA, "-d", REBASE_DB, "--quiet"], check=True)

def compute_canonical_markov_z(fna_path: str, max_bp: int = 25000):
    """
    Computes 136 canonical 4-mer Markov Z-scores across standardized metagenomic contig windows
    (default: central 25 kb window per genome to mimic metagenomic MAG contigs and prevent 5Mb length saturation).
    """
    c2 = {k: 1.0 for k in ALL_2MERS}
    c3 = {k: 1.0 for k in ALL_3MERS}
    c4 = {k: 1.0 for k in ALL_4MERS}

    concat_seq = []
    total_len = 0
    for record in SeqIO.parse(fna_path, "fasta"):
        s = str(record.seq).upper()
        concat_seq.append(s)
        total_len += len(s)
        if total_len >= max_bp * 3:
            break
    full_s = "".join(concat_seq)
    # Extract a representative 25 kb metagenomic contig window (after skipping first 10kb of replication origin)
    start = min(10000, len(full_s) // 4)
    contig = full_s[start : start + max_bp]

    for seq in (contig, rev_comp(contig)):
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

    z_full = {}
    for w in ALL_4MERS:
        n_w, n_123, n_234 = c4[w], c3[w[:3]], c3[w[1:]]
        n_23, n_12 = c2[w[1:3]], c2[w[:2]]
        exp_w = (n_123 * n_234) / n_23
        var_w = exp_w * max(1e-6, (1.0 - n_123 / n_12)) * max(1e-6, (1.0 - n_234 / n_23))
        z_full[w] = (n_w - exp_w) / np.sqrt(var_w)

    # Collapse 256 4-mers into 136 Canonical 4-mers
    z_canon = np.array([0.5 * (z_full[k] + z_full[rev_comp(k)]) for k in CANONICAL_4MERS])
    return H4, (H4 - H3), z_canon

def knn_genus_accuracy(dist_mat: np.ndarray, genera: list, k: int = 3) -> float:
    """Computes k-Nearest Neighbor taxonomic accuracy (do genomes cluster with their true Genus?)."""
    correct = 0
    n = len(genera)
    for i in range(n):
        dists = dist_mat[i].copy()
        dists[i] = np.inf
        nearest_idx = np.argsort(dists)[:k]
        neighbor_genera = [genera[j] for j in nearest_idx]
        # Majority vote among k nearest neighbors
        most_common = max(set(neighbor_genera), key=neighbor_genera.count)
        if most_common == genera[i]:
            correct += 1
    return (correct / n) * 100.0

def multi_group_F_stat(dist_mat: np.ndarray, groups: list) -> float:
    """Pseudo-F statistic (ratio of between-genus variance to within-genus variance)."""
    intra, inter = [], []
    n = len(groups)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if groups[i] == groups[j] else inter).append(dist_mat[i, j] ** 2)
    return float(np.mean(inter) / np.mean(intra))

def main():
    build_expanded_rebase_db()

    print("[2/4] Profiling Phage Avoidance Across 136 Canonical 4-mers...")
    phage_files = sorted(glob.glob(f"{BASE_DIR}/data/phages/fna/*.fna"))
    phage_z = np.vstack([compute_canonical_markov_z(pf, max_bp=40000)[2] for pf in phage_files])
    mean_phage_z = np.mean(phage_z, axis=0)
    sd_phage_z = np.std(phage_z, axis=0)
    s_phage = {CANONICAL_4MERS[i]: max(0.0, -mean_phage_z[i]) for i in range(len(CANONICAL_4MERS))}

    print("[3/4] Scanning Bacteroidales Genomes & Computing 136-mer Profiles...")
    fna_files = sorted(glob.glob(f"{BASE_DIR}/data/bacteroidales/fna/*.fna"))
    meta_rows, z_rows, rm_matrix = [], [], []

    for fna in fna_files:
        gid = os.path.basename(fna).replace(".fna", "")
        faa = f"{BASE_DIR}/data/bacteroidales/faa/{gid}.faa"
        if not os.path.exists(faa):
            continue

        genus, label = parse_genus_and_species(fna, gid)
        if genus not in TARGET_GENERA:
            continue

        # Scan proteome with DIAMOND
        out_tsv = f"{EXPORT_DIR}/{gid}_tmp.tsv"
        subprocess.run([
            "diamond", "blastp", "-d", REBASE_DB, "-q", faa, "-o", out_tsv,
            "--id", "40", "--query-cover", "70", "--evalue", "1e-12",
            "--max-target-seqs", "1", "--outfmt", "6", "qseqid", "sseqid", "--quiet"
        ], check=True)

        detected = set()
        if os.path.exists(out_tsv):
            with open(out_tsv) as f:
                for line in f:
                    sseqid = line.strip().split("\t")[1]
                    if "|MOTIFS:" in sseqid:
                        detected.update(sseqid.split("|MOTIFS:")[1].split(","))
            os.remove(out_tsv)

        H4, h4, z_canon = compute_canonical_markov_z(fna, max_bp=25000)
        z_rows.append(pd.Series(z_canon, index=CANONICAL_4MERS, name=gid))
        rm_matrix.append({k: (1 if k in detected else 0) for k in CANONICAL_4MERS})

        meta_rows.append({
            "genome_id": gid,
            "display_label": label,
            "genus": genus,
            "H4_bits": round(H4, 5),
            "h4_cond_bits": round(h4, 5),
            "GATC_RM": 1 if "GATC" in detected else 0,
            "CTAG_RM": 1 if "CTAG" in detected else 0,
            "CCGG_RM": 1 if "CCGG" in detected else 0,
            "total_RM_targets": len(detected)
        })

    meta_df = pd.DataFrame(meta_rows)
    # Keep only genera with at least 3 genomes
    valid_genera = meta_df["genus"].value_counts()
    keep_genera = valid_genera[valid_genera >= 3].index.tolist()
    keep_mask = meta_df["genus"].isin(keep_genera).values

    meta_df = meta_df[keep_mask].reset_index(drop=True)
    z_df = pd.DataFrame(z_rows).iloc[keep_mask]
    rm_df = pd.DataFrame(rm_matrix).iloc[keep_mask]

    meta_df.to_csv(f"{EXPORT_DIR}/metadata_and_entropy.tsv", sep="\t", index=False)
    z_df.to_csv(f"{EXPORT_DIR}/bacterial_markov_z_scores.tsv", sep="\t")

    print(f"      Retained {len(meta_df)} genomes across genera: { meta_df['genus'].value_counts().to_dict() }")

    print("[4/4] Computing Eco-Weights & Evaluating Genus Resolution...")
    # Standardize each canonical 4-mer across genomes so high-copy housekeeping 4-mers don't drown out R-M motifs
    Z_std = (z_df.values - np.mean(z_df.values, axis=0)) / (np.std(z_df.values, axis=0) + 1e-8)

    # Identify convergently evolving R-M + Phage-avoided 4-mers
    # A 4-mer is penalized when it is an active R-M target AND exhibits high intra-genus variance / phage avoidance
    lam = 1.5
    weights_eco = np.ones(len(CANONICAL_4MERS))
    kmer_table = []

    for idx, kmer in enumerate(CANONICAL_4MERS):
        is_rm_target = kmer in RM_TARGET_CANONICAL
        rm_prev = float(rm_df[kmer].mean())
        avoid_s = float(s_phage[kmer])

        if is_rm_target and (rm_prev > 0 or avoid_s > 0.3):
            # Combine phage avoidance intensity + R-M prevalence
            eco_pressure = avoid_s + (1.5 * rm_prev)
            weights_eco[idx] = np.exp(-lam * eco_pressure)

        kmer_table.append({
            "kmer": kmer,
            "is_rm_target": is_rm_target,
            "is_strict_palindrome": (kmer == rev_comp(kmer)),
            "mean_phage_Z": round(float(mean_phage_z[idx]), 4),
            "sd_phage_Z": round(float(sd_phage_z[idx]), 4),
            "phage_avoidance_S": round(avoid_s, 4),
            "bacterial_RM_prevalence": round(rm_prev, 4),
            "eco_weight": round(float(weights_eco[idx]), 4)
        })

    pd.DataFrame(kmer_table).to_csv(f"{EXPORT_DIR}/kmer_weights_and_phage_avoidance.tsv", sep="\t", index=False)

    ids = meta_df["genome_id"].tolist()
    genera = meta_df["genus"].tolist()

    D_unw = squareform(pdist(Z_std, metric="cosine"))
    D_eco = squareform(pdist(Z_std * np.sqrt(weights_eco), metric="cosine"))

    f_unw = multi_group_F_stat(D_unw, genera)
    f_eco = multi_group_F_stat(D_eco, genera)
    knn_unw = knn_genus_accuracy(D_unw, genera, k=3)
    knn_eco = knn_genus_accuracy(D_eco, genera, k=3)

    print("=================================================================")
    print(f"  Multi-Genus Separation Ratio (Unweighted):   {f_unw:.4f}")
    print(f"  Multi-Genus Separation Ratio (Eco-Weighted): {f_eco:.4f} (+{((f_eco/f_unw)-1)*100:.1f}%)")
    print(f"  3-NN Genus Accuracy (Unweighted):            {knn_unw:.1f}%")
    print(f"  3-NN Genus Accuracy (Eco-Weighted):          {knn_eco:.1f}%")
    print("=================================================================")

    pd.DataFrame(D_unw, index=ids, columns=ids).to_csv(f"{EXPORT_DIR}/dist_unweighted.tsv", sep="\t")
    pd.DataFrame(D_eco, index=ids, columns=ids).to_csv(f"{EXPORT_DIR}/dist_ecoweighted.tsv", sep="\t")

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
    coords["f_unw"] = round(f_unw, 4)
    coords["f_eco"] = round(f_eco, 4)
    coords["knn_unw"] = round(knn_unw, 1)
    coords["knn_eco"] = round(knn_eco, 1)
    coords.to_csv(f"{EXPORT_DIR}/pcoa_coordinates.tsv", sep="\t", index=False)
    print(f"Saved upgraded bundle to {EXPORT_DIR}")

if __name__ == "__main__":
    main()
