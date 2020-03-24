#run bash load_L1000_PhaseII.sh to load required data
#!/bin/bash

DATA_DIR=../L1000_PhaseII
ACC=GSE70138
PREFIX=${ACC}_Broad_LINCS
GEO_URL=https://www.ncbi.nlm.nih.gov/geo

files=(
    Level5_COMPZ_n118050x12328_2017-03-06.gctx
    cell_info_2017-04-28.txt
    gene_info_2017-03-06.txt
    sig_info_2017-03-06.txt
    sig_metrics_2017-03-06.txt
    pert_info_2017-03-06.txt
)

mkdir $DATA_DIR && cd $DATA_DIR
mkdir $PREFIX && cd $PREFIX

for f in ${files[*]}; do
  curl "$GEO_URL/download/?acc=$ACC&format=file&file=${PREFIX}_$f.gz" \
    | gunzip > $f
done

# remove "pr_" from gene metadata cols
sed -i "s/pr_//g" gene_info_2017-03-06.txt

cd ../../../
