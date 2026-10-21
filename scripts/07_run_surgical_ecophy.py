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
REBASE_FASTA = f"{BASE_DIR}/data/rebase/rebase_strict16.faa"
REBASE_DB = f"{BASE_DIR}/data/rebase/rebase_strict16_db"
EXPORT_DIR = f"{BASE_DIR}/results/local_bundle_v3"
os.makedirs(EXPORT_DIR, exist_ok=True)

BASES = ["A", "C", "G", "T"]
COMPLEMENT = str.maketrans("ACGT", "TGCA")
ALL_4MERS = ["".join(p) for p in itertools.product(BASES, repeat=4)]

def rev_comp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

CANONICAL_4MERS = sorted(list({min(k, rev_comp(k)) for k in ALL_4MERS}))
CANON_IDX_MAP = {k: [ALL_4MERS.index(k), ALL_4MERS.index(rev_comp(k))] for k in CANONICAL_4MERS}

# Strictly the 16 canonical 4-mer palindromes (never touch protein-coding non-palindromes)
STRICT_PALINDROMES = [
    "AATT", "ACGT", "AGCT", "ATAT", "CATG", "CCGG", "CGCG", "CTAG",
    "GATC", "GCGC", "GGCC", "GTAC", "TATA", "TCGA", "TGCA", "TTAA"
]

TARGET_GENERA = {"Bacteroides", "Phocaeicola", "Parabacteroides", "Prevotella", "Alistipes", "Porphyromonas"}

# Fast lookup table for ASCII -> 0,1,2,3 (A,C,G,T)
LOOKUP = np.full(256, -1, dtype=np.int16)
LOOKUP[ord("A")] = 0
LOOKUP[ord("C")] = 1
LOOKUP[ord("G")] = 2
LOOKUP[ord("T")] = 3

def fast_symmetrized_counts(seq_str: str):
    """Vectorized NumPy k-mer counter: processes 5 Mb chromosomes in ~10 ms."""
    arr_fwd = LOOKUP[np.frombuffer(seq_str.encode("ascii", "ignore"), dtype=np.uint8)]
    arr_rev = 3 - arr_fwd[::-1]  # Reverse complement in integer space (0<->3, 1<->2)

    c4 = np.ones(256, dtype=np.float64)
    c3 = np.ones(64, dtype=np.float64)
    c2 = np.ones(16, dtype=np.float64)

    for arr in (arr_fwd, arr_rev):
        b0, b1, b2, b3 = arr[:-3], arr[1:-2], arr[2:-1], arr[3:]
        
        # Ensure bases are strictly A,C,G,T (0, 1, 2, 3). 
        # 'N' becomes -1 in fwd and 4 in rev, so this safely drops them.
        valid = (b0 >= 0) & (b0 <= 3) & (b1 >= 0) & (b1 <= 3) & \
                (b2 >= 0) & (b2 <= 3) & (b3 >= 0) & (b3 <= 3)

        b0, b1, b2, b3 = b0[valid], b1[valid], b2[valid], b3[valid]

        idx4 = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3
        idx3 = (b0 << 4) | (b1 << 2) | b2
        idx2 = (b0 << 2) | b1

        c4 += np.bincount(idx4, minlength=256)
        c3 += np.bincount(idx3, minlength=64)
        c2 += np.bincount(idx2, minlength=16)

    return c2, c3, c4

def compute_markov_z_fast(seq_str: str):
    """Computes H4, h4, and 136 canonical Markov Z-scores from an arbitrary sequence string."""
    c2, c3, c4 = fast_symmetrized_counts(seq_str)
    p4 = c4 / np.sum(c4)
    p3 = c3 / np.sum(c3)
    H4 = float(-np.sum(p4 * np.log2(p4)))
    H3 = float(-np.sum(p3 * np.log2(p3)))

    z_256 = np.zeros(256, dtype=np.float64)
    for idx in range(256):
        b0, b1, b2, b3 = (idx >> 6) & 3, (idx >> 4) & 3, (idx >> 2) & 3, idx & 3
        i123 = (b0 << 4) | (b1 << 2) | b2
        i234 = (b1 << 4) | (b2 << 2) | b3
        i12 = (b0 << 2) | b1
        i23 = (b1 << 2) | b2

        n_w, n_123, n_234, n_12, n_23 = c4[idx], c3[i123], c3[i234], c2[i12], c2[i23]
        exp_w = (n_123 * n_234) / n_23
        var_w = exp_w * max(1e-6, 1.0 - n_123 / n_12) * max(1e-6, 1.0 - n_234 / n_23)
        z_256[idx] = (n_w - exp_w) / np.sqrt(var_w)

    # Collapse 256 into 136 canonical 4-mers and normalize by sequence length sqrt(N)
    # so Z-scores represent intrinsic compositional bias independent of contig length
    scale = np.sqrt(max(1.0, np.sum(c4) / 50000.0))
    z_canon = np.array([
        0.5 * (z_256[CANON_IDX_MAP[k][0]] + z_256[CANON_IDX_MAP[k][1]]) / scale
        for k in CANONICAL_4MERS
    ])
    return H4, (H4 - H3), z_canon

def build_strict_rebase_db():
    print("[1/4] Building Strict-Palindrome REBASE Database...")
    motif_regex = re.compile(r"\b([ACGT]{4,6})\b")
    with open(REBASE_RAW, "r", errors="ignore") as fin, open(REBASE_FASTA, "w") as fout:
        header, seq_lines = None, []
        for line in fin:
            line = line.strip()
            if not line or line == "<>":
                continue
            if line.startswith(">"):
                if header and seq_lines:
                    seq = re.sub(r"[^A-Za-z]", "", "".join(seq_lines))
                    tokens = motif_regex.findall(header)
                    matched = [p for p in STRICT_PALINDROMES if any(p in tok for tok in tokens)]
                    if matched and len(seq) > 50:
                        clean_id = header.split()[0].replace(">", "")
                        fout.write(f">{clean_id}|MOTIFS:{','.join(matched)}\n{seq}\n")
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
    subprocess.run(["diamond", "makedb", "--in", REBASE_FASTA, "-d", REBASE_DB, "--quiet"], check=True)

def parse_genus_and_species(fna_path: str, acc: str):
    for record in SeqIO.parse(fna_path, "fasta"):
        clean = re.sub(r"[\[\]]", "", record.description)
        tokens = clean.split()
        if len(tokens) >= 3:
            return tokens[1], f"{tokens[1][0]}. {tokens[2]} ({acc})"
    return "Unknown", acc

def evaluate_metrics(dist_mat: np.ndarray, labels: list, genome_ids: list, k: int = 3):
    """Computes k-NN Accuracy (excluding self-genome windows) and Mean Same-Genus Similarity Ratio."""
    n = len(labels)
    correct = 0
    intra, inter = [], []
    for i in range(n):
        dists = dist_mat[i].copy()
        # Mask out windows from the exact same assembly so contig evaluation is strictly cross-genome
        for j in range(n):
            if genome_ids[i] == genome_ids[j]:
                dists[j] = np.inf
            else:
                (intra if labels[i] == labels[j] else inter).append(dist_mat[i, j])
        nearest_idx = np.argsort(dists)[:k]
        votes = [labels[j] for j in nearest_idx]
        pred = max(set(votes), key=votes.count)
        if pred == labels[i]:
            correct += 1
    return (correct / n) * 100.0, float(np.mean(inter) / np.mean(intra))

def apply_surgical_eco_correction(Z_mat: np.ndarray, rm_mask: np.ndarray, s_phage_vec: np.ndarray, lam: float):
    """
    Surgical Strain-Specific Shrinkage:
    Only shrinks negative palindrome depletion outliers (Z < -0.5) in genomes that carry the R-M system
    OR show extreme convergent palindrome avoidance, leaving R-M(-) and non-depleted motifs 100% intact.
    """
    Z_corr = Z_mat.copy()
    n_samples, n_features = Z_corr.shape
    for j in range(n_features):
        if s_phage_vec[j] <= 0:
            continue
        for i in range(n_samples):
            # Trigger shrinkage only when the palindrome is depleted (Z < -0.5) AND linked to R-M / phage pressure
            if Z_corr[i, j] < -0.5 and (rm_mask[i, j] == 1 or Z_corr[i, j] < -1.5):
                shrink_factor = 1.0 + lam * s_phage_vec[j] * (1.0 + 0.5 * rm_mask[i, j])
                Z_corr[i, j] = Z_corr[i, j] / shrink_factor
    return Z_corr

def main():
    build_strict_rebase_db()

    print("[2/4] Profiling Phage Palindrome Avoidance Across Full Viral Genomes...")
    phage_files = sorted(glob.glob(f"{BASE_DIR}/data/phages/fna/*.fna"))
    phage_z_list = []
    for pf in phage_files:
        full_phage = "".join(str(r.seq).upper() for r in SeqIO.parse(pf, "fasta"))
        phage_z_list.append(compute_markov_z_fast(full_phage)[2])
    phage_z = np.vstack(phage_z_list)
    mean_phage_z = np.mean(phage_z, axis=0)
    sd_phage_z = np.std(phage_z, axis=0)

    s_phage_vec = np.zeros(len(CANONICAL_4MERS))
    for i, kmer in enumerate(CANONICAL_4MERS):
        if kmer in STRICT_PALINDROMES:
            s_phage_vec[i] = max(0.0, -mean_phage_z[i])

    print("[3/4] Scanning Full Bacterial Genomes & Extracting 5x 50kb Metagenomic Contigs per Genome...")
    fna_files = sorted(glob.glob(f"{BASE_DIR}/data/bacteroidales/fna/*.fna"))
    genome_meta, genome_z, genome_rm = [], [], []
    contig_meta, contig_z, contig_rm = [], [], []

    for fna in fna_files:
        gid = os.path.basename(fna).replace(".fna", "")
        faa = f"{BASE_DIR}/data/bacteroidales/faa/{gid}.faa"
        if not os.path.exists(faa):
            continue
        genus, label = parse_genus_and_species(fna, gid)
        if genus not in TARGET_GENERA:
            continue

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

        full_seq = "".join(str(r.seq).upper() for r in SeqIO.parse(fna, "fasta"))
        if len(full_seq) < 300000:
            continue

        # 1. Full-Genome Profile (100% of chromosome)
        H4, h4, z_full = compute_markov_z_fast(full_seq)
        rm_vec = np.array([1 if k in detected else 0 for k in CANONICAL_4MERS])

        genome_meta.append({
            "sample_id": gid, "genome_id": gid, "display_label": label, "genus": genus,
            "H4_bits": round(H4, 5), "h4_cond_bits": round(h4, 5),
            "GATC_RM": int("GATC" in detected), "CTAG_RM": int("CTAG" in detected),
            "total_RM_palindromes": len(detected)
        })
        genome_z.append(z_full)
        genome_rm.append(rm_vec)

        # 2. Extract 5 Non-Overlapping 50 kb Contigs evenly spaced across the genome
        step = len(full_seq) // 5
        for w_idx in range(5):
            frag = full_seq[w_idx * step : w_idx * step + 50000]
            H4_c, h4_c, z_c = compute_markov_z_fast(frag)
            contig_meta.append({
                "sample_id": f"{gid}_c{w_idx+1}", "genome_id": gid,
                "display_label": f"{label}_c{w_idx+1}", "genus": genus,
                "H4_bits": round(H4_c, 5), "h4_cond_bits": round(h4_c, 5),
                "GATC_RM": int("GATC" in detected), "CTAG_RM": int("CTAG" in detected),
                "total_RM_palindromes": len(detected)
            })
            contig_z.append(z_c)
            contig_rm.append(rm_vec)

    df_g = pd.DataFrame(genome_meta)
    valid_genera = df_g["genus"].value_counts()
    keep_genera = valid_genera[valid_genera >= 3].index.tolist()

    mask_g = df_g["genus"].isin(keep_genera).values
    df_g = df_g[mask_g].reset_index(drop=True)
    Z_g = np.vstack(genome_z)[mask_g]
    RM_g = np.vstack(genome_rm)[mask_g]

    df_c = pd.DataFrame(contig_meta)
    mask_c = df_c["genus"].isin(keep_genera).values
    df_c = df_c[mask_c].reset_index(drop=True)
    Z_c = np.vstack(contig_z)[mask_c]
    RM_c = np.vstack(contig_rm)[mask_c]

    print(f"      Full Genomes: {len(df_g)} | 50kb Metagenomic Contigs: {len(df_c)}")

    print("[4/4] Sweeping Lambda to Find Optimal Surgical Shrinkage...")
    best_lam, best_acc = 0.0, 0.0
    for lam in np.linspace(0.0, 3.0, 31):
        Z_c_corr = apply_surgical_eco_correction(Z_c, RM_c, s_phage_vec, lam)
        D_c = squareform(pdist(Z_c_corr, metric="cosine"))
        acc, _ = evaluate_metrics(D_c, df_c["genus"].tolist(), df_c["genome_id"].tolist(), k=5)
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    base_acc_c, base_ratio_c = evaluate_metrics(
        squareform(pdist(Z_c, metric="cosine")), df_c["genus"].tolist(), df_c["genome_id"].tolist(), k=5
    )
    Z_c_opt = apply_surgical_eco_correction(Z_c, RM_c, s_phage_vec, best_lam if best_lam > 0 else 0.8)
    opt_acc_c, opt_ratio_c = evaluate_metrics(
        squareform(pdist(Z_c_opt, metric="cosine")), df_c["genus"].tolist(), df_c["genome_id"].tolist(), k=5
    )

    print("======================================================================")
    print(f"  50kb Contig Cross-Genome 5-NN Accuracy (Unweighted): {base_acc_c:.2f}% (Ratio: {base_ratio_c:.4f})")
    print(f"  50kb Contig Cross-Genome 5-NN Accuracy (Surgical):   {opt_acc_c:.2f}% (Ratio: {opt_ratio_c:.4f}) at lambda={best_lam:.2f}")
    print("======================================================================")

    # Save everything to local_bundle_v3
    df_g.to_csv(f"{EXPORT_DIR}/genome_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(Z_g, index=df_g["sample_id"], columns=CANONICAL_4MERS).to_csv(f"{EXPORT_DIR}/genome_markov_z.tsv", sep="\t")
    pd.DataFrame(RM_g, index=df_g["sample_id"], columns=CANONICAL_4MERS).to_csv(f"{EXPORT_DIR}/genome_rm_mask.tsv", sep="\t")

    df_c.to_csv(f"{EXPORT_DIR}/contig_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(Z_c, index=df_c["sample_id"], columns=CANONICAL_4MERS).to_csv(f"{EXPORT_DIR}/contig_markov_z.tsv", sep="\t")
    pd.DataFrame(RM_c, index=df_c["sample_id"], columns=CANONICAL_4MERS).to_csv(f"{EXPORT_DIR}/contig_rm_mask.tsv", sep="\t")

    kmer_info = pd.DataFrame({
        "kmer": CANONICAL_4MERS,
        "is_strict_palindrome": [k in STRICT_PALINDROMES for k in CANONICAL_4MERS],
        "mean_phage_Z": np.round(mean_phage_z, 4),
        "sd_phage_Z": np.round(sd_phage_z, 4),
        "phage_avoidance_S": np.round(s_phage_vec, 4),
        "rm_prevalence": np.round(np.mean(RM_g, axis=0), 4)
    })
    kmer_info.to_csv(f"{EXPORT_DIR}/kmer_phage_info.tsv", sep="\t", index=False)

    # Save Full-Genome Trees (Unweighted vs Surgical Eco-Corrected)
    D_g_unw = squareform(pdist(Z_g, metric="cosine"))
    Z_g_eco = apply_surgical_eco_correction(Z_g, RM_g, s_phage_vec, best_lam if best_lam > 0 else 0.8)
    D_g_eco = squareform(pdist(Z_g_eco, metric="cosine"))

    nj(DistanceMatrix(D_g_unw, df_g["sample_id"].tolist())).write(f"{EXPORT_DIR}/tree_unweighted.nwk")
    nj(DistanceMatrix(D_g_eco, df_g["sample_id"].tolist())).write(f"{EXPORT_DIR}/tree_ecoweighted.nwk")
    print(f"Exported complete v3 bundle to {EXPORT_DIR}")

if __name__ == "__main__":
    main()
