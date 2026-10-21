Server scripts:

01_download_data.sh - downloads the initial benchmark dataset, including complete reference genomes for two distant phyla (Enterobacteriaceae and Bacteroidaceae), gut phage genomes (Crassvirales), and the REBASE restriction enzyme database.

02_annotate_rm.py - parses REBASE for enzymes targeting 4-mer palindromes, builds a DIAMOND database, and scans the bacterial proteomes to map the presence and absence of specific Restriction-Modification (R-M) systems in each genome.

03_run_ecophy.py - computes full-genome 4-mer Shannon entropy and 2nd-order Markov Z-scores, calculates empirical phage palindrome avoidance, and builds the initial proof-of-concept unweighted vs eco-weighted phylogenetic trees.

04_export_for_local.py - extracts and packages the computed Z-score matrices, PCoA coordinates, metadata, and Newick trees into a lightweight bundle (.tsv and .nwk) for local visualization without needing the raw FASTA files.

05_upgrade_bacteroidales.sh - expands the benchmark to a higher-resolution multi-genus dataset by downloading complete genomes for closely related gut bacteria within the same order (Bacteroides, Prevotella, Parabacteroides, Alistipes).

06_run_bacteroidales_ecophy.py - upgrades the feature space to canonical 136-mers to prevent strand bias, expands the REBASE scan to include degenerate Type II R-M motifs, and introduces initial k-nearest neighbor genus classification on simulated contig windows.

07_run_surgical_ecophy.py - optimizes the pipeline with lightning-fast vectorized NumPy k-mer counting, extracts non-overlapping 50 kb contigs to simulate metagenomic binning, and applies strain-specific "surgical" shrinkage to R-M outliers to prevent the destruction of vertical phylogenetic signals.

08_fix_ecophy_math.py - the final evaluation script. It removes data leakage by utilizing unsupervised community baseline centering and intra-genomic spatial instability weights, and runs a strict 200-permutation random null audit to prove the cross-species genus placement accuracy.

Local scripts:

viz.ipynb - local data processing and visualizations of the server work.

tanglegrams.r - the final script of drawing tanglegram and exporting it as pdf file.
