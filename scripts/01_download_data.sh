#!/usr/bin/env bash
set -euo pipefail

cd ~/ecophy_project/data

echo "[1/4] Downloading 30 Complete Enterobacteriaceae Genomes (Clade A)..."
datasets download genome taxon "Escherichia coli" \
    --assembly-level complete \
    --assembly-source RefSeq \
    --include genome,protein \
    --dehydrated \
    --filename entero.zip

unzip -q entero.zip -d entero_raw
# Keep first 30 complete assemblies to keep the PoC fast and balanced
head -n 31 entero_raw/ncbi_dataset/data/dataset_catalog.json > /dev/null
datasets rehydrate --directory entero_raw/ --max-workers 8

mkdir -p entero/fna entero/faa
count=0
for dir in entero_raw/ncbi_dataset/data/GCF_*; do
    [ -d "$dir" ] || continue
    acc=$(basename "$dir")
    cp "$dir"/*.fna "entero/fna/${acc}.fna" 2>/dev/null || true
    cp "$dir"/*.faa "entero/faa/${acc}.faa" 2>/dev/null || true
    count=$((count + 1))
    [ "$count" -ge 30 ] && break
done

echo "[2/4] Downloading 30 Complete Bacteroidaceae Genomes (Clade B)..."
datasets download genome taxon "Bacteroidaceae" \
    --assembly-level complete \
    --assembly-source RefSeq \
    --include genome,protein \
    --dehydrated \
    --filename bacteroides.zip

unzip -q bacteroides.zip -d bacteroides_raw
datasets rehydrate --directory bacteroides_raw/ --max-workers 8

mkdir -p bacteroides/fna bacteroides/faa
count=0
for dir in bacteroides_raw/ncbi_dataset/data/GCF_*; do
    [ -d "$dir" ] || continue
    acc=$(basename "$dir")
    cp "$dir"/*.fna "bacteroides/fna/${acc}.fna" 2>/dev/null || true
    cp "$dir"/*.faa "bacteroides/faa/${acc}.faa" 2>/dev/null || true
    count=$((count + 1))
    [ "$count" -ge 30 ] && break
done

echo "[3/4] Downloading Gut Phage Genomes (Crassvirales & Caudoviricetes)..."
datasets download genome taxon "Crassvirales" \
    --assembly-level complete \
    --include genome \
    --filename phages.zip

unzip -q phages.zip -d phages_raw
mkdir -p phages/fna
count=0
for dir in phages_raw/ncbi_dataset/data/GCA_* phages_raw/ncbi_dataset/data/GCF_*; do
    [ -d "$dir" ] || continue
    acc=$(basename "$dir")
    cp "$dir"/*.fna "phages/fna/${acc}.fna" 2>/dev/null || true
    count=$((count + 1))
    [ "$count" -ge 40 ] && break
done

echo "[4/4] Downloading REBASE Protein Sequences with Recognition Motifs..."
cd ~/ecophy_project/data/rebase
wget ftp://ftp.neb.com/pub/rebase/protein_seqs.txt

echo "Done! Data downloaded and structured."
