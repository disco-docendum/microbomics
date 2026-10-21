#!/usr/bin/env python3
import os
import re
import glob
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser("~/ecophy_project")
REBASE_RAW = f"{BASE_DIR}/data/rebase/protein_seqs.txt"
REBASE_FASTA = f"{BASE_DIR}/data/rebase/rebase_4mer_targets.faa"
REBASE_DB = f"{BASE_DIR}/data/rebase/rebase_db"
RESULTS_DIR = f"{BASE_DIR}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PALINDROMES_4MER = [
    "AATT", "ACGT", "AGCT", "ATAT",
    "CATG", "CCGG", "CGCG", "CTAG",
    "GATC", "GCGC", "GGCC", "GTAC",
    "TATA", "TCGA", "TGCA", "TTAA"
]

def parse_rebase_to_fasta():
    """Extracts REBASE proteins that recognize any of the 16 canonical 4-mer palindromes."""
    print("[1/3] Filtering REBASE for enzymes targeting 4-mer palindromes...")
    kept = 0
    with open(REBASE_RAW, "r", errors="ignore") as fin, open(REBASE_FASTA, "w") as fout:
        header, seq_lines = None, []
        for line in fin:
            line = line.strip()
            if not line or line == "<>":
                continue
            if line.startswith(">"):
                if header and seq_lines:
                    seq = re.sub(r'[^A-Za-z]', '', "".join(seq_lines))
                    matched_motifs = [p for p in PALINDROMES_4MER if p in header]
                    if matched_motifs and len(seq) > 50:
                        motif_str = ",".join(matched_motifs)
                        clean_id = header.split()[0].replace(">", "")
                        fout.write(f">{clean_id}|MOTIFS:{motif_str}\n{seq}\n")
                        kept += 1
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
    print(f"      Retained {kept} REBASE R-M/MTase proteins linked to 4-mer palindromes.")

def build_and_run_diamond():
    """Builds DIAMOND DB and scans all bacterial proteomes for R-M motif targets."""
    print("[2/3] Building DIAMOND database...")
    subprocess.run(["diamond", "makedb", "--in", REBASE_FASTA, "-d", REBASE_DB, "--quiet"], check=True)

    faa_files = (
        glob.glob(f"{BASE_DIR}/data/entero/faa/*.faa") +
        glob.glob(f"{BASE_DIR}/data/bacteroides/faa/*.faa")
    )
    print(f"[3/3] Scanning {len(faa_files)} bacterial proteomes against REBASE...")

    records = []
    for faa in sorted(faa_files):
        genome_id = os.path.basename(faa).replace(".faa", "")
        clade = "Enterobacteriaceae" if "/entero/" in faa else "Bacteroidaceae"
        out_tsv = f"{RESULTS_DIR}/{genome_id}_diamond.tsv"

        subprocess.run([
            "diamond", "blastp",
            "-d", REBASE_DB,
            "-q", faa,
            "-o", out_tsv,
            "--id", "45",
            "--query-cover", "75",
            "--evalue", "1e-15",
            "--max-target-seqs", "1",
            "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue",
            "--quiet"
        ], check=True)

        detected_motifs = set()
        if os.path.exists(out_tsv):
            with open(out_tsv) as f:
                for line in f:
                    sseqid = line.strip().split("\t")[1]
                    if "|MOTIFS:" in sseqid:
                        motifs = sseqid.split("|MOTIFS:")[1].split(",")
                        detected_motifs.update(motifs)
            os.remove(out_tsv)

        row = {"genome_id": genome_id, "clade": clade}
        for p in PALINDROMES_4MER:
            row[p] = 1 if p in detected_motifs else 0
        records.append(row)

    df = pd.DataFrame(records)
    out_path = f"{RESULTS_DIR}/rm_motif_presence.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"      Saved R-M target matrix to {out_path}")

if __name__ == "__main__":
    parse_rebase_to_fasta()
    build_and_run_diamond()
