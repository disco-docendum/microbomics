#!/usr/bin/env bash
set -euo pipefail

cd ~/ecophy_project/data
mkdir -p bacteroidales/{fna,faa}

# 1. Copy the already-downloaded Bacteroidaceae genomes
cp bacteroides/fna/*.fna bacteroidales/fna/ 2>/dev/null || true
cp bacteroides/faa/*.faa bacteroidales/faa/ 2>/dev/null || true

# 2. Download additional gut Bacteroidales genera (Prevotella, Parabacteroides, Alistipes)
for GENUS in "Prevotella" "Parabacteroides" "Alistipes" "Porphyromonas"; do
    echo "Downloading complete RefSeq genomes for ${GENUS}..."
    datasets download genome taxon "$GENUS" \
        --assembly-level complete \
        --assembly-source RefSeq \
        --include genome,protein \
        --dehydrated \
        --filename "${GENUS}.zip" || continue

    rm -rf "${GENUS}_raw"
    unzip -q "${GENUS}.zip" -d "${GENUS}_raw"
    datasets rehydrate --directory "${GENUS}_raw/" --max-workers 8

    count=0
    for dir in "${GENUS}_raw"/ncbi_dataset/data/GCF_*; do
        [ -d "$dir" ] || continue
        acc=$(basename "$dir")
        cp "$dir"/*.fna "bacteroidales/fna/${acc}.fna" 2>/dev/null || true
        cp "$dir"/*.faa "bacteroidales/faa/${acc}.faa" 2>/dev/null || true
        count=$((count + 1))
        [ "$count" -ge 15 ] && break
    done
done

echo "Total Bacteroidales genomes ready: $(ls bacteroidales/fna/*.fna | wc -l)"
