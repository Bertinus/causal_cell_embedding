#run bash load_L1000_PhaseII.sh to load required data
#!/bin/bash

DATA_DIR=L1000_PhaseI
ACC=GSE92742
PREFIX=${ACC}_Broad_LINCS

files=(
    GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx
    GSE92742_Broad_LINCS_cell_info.txt
    GSE92742_Broad_LINCS_gene_info.txt
    GSE92742_Broad_LINCS_sig_info.txt
    GSE92742_Broad_LINCS_sig_metrics.txt
    GSE92742_Broad_LINCS_pert_info.txt
)

mkdir $DATA_DIR && cd $DATA_DIR
mkdir $PREFIX && cd $PREFIX

for f in ${files[*]}; do
  curl "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/$f.gz" \
    | gunzip > $f
done

cd ../../