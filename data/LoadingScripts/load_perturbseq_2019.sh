#run bash load_perturbseq_2019.sh to load required data
#!/bin/bash

echo "
Due to some error, you have to download yourself the file GSE133344_RAW.tar by going to:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344

"

DATA_DIR=../PerturbSeq_2019
ACC=GSE133344
GEO_URL=https://www.ncbi.nlm.nih.gov/geo

files=(
  GSE133344_filtered_barcodes.tsv
  GSE133344_filtered_cell_identities.csv
  GSE133344_filtered_genes.tsv
  GSE133344_filtered_matrix.mtx
  GSE133344_raw_barcodes.tsv
  GSE133344_raw_cell_identities.csv
  GSE133344_raw_genes.tsv
  GSE133344_raw_matrix.mtx
)

mkdir $DATA_DIR && cd $DATA_DIR

for f in ${files[*]}; do
  curl "$GEO_URL/download/?acc=$ACC&format=file&file=$f.gz" \
    | gunzip > $f
done

echo "
Due to some error, you have to download yourself the file GSE133344_RAW.tar by going to:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344

"

cd ../../