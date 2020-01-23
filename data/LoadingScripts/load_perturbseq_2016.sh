#run bash load.sh to load perturbseq 2016 data
#!/bin/bash

ACC=GSE90063
PREFIX=${ACC}
GEO_URL=https://www.ncbi.nlm.nih.gov/geo

files=(

    dc0hr_umi_wt.txt
    dc3hr_umi_wt.txt
    k562_umi_wt.txt

)

data=(

    RAW

)

mkdir $PREFIX && cd $PREFIX

for f in ${files[*]}; do
  curl "$GEO_URL/download/?acc=$ACC&format=file&file=${PREFIX}_$f.gz" \
    | gunzip > $f
done


for d in ${data[*]}; do
  curl "$GEO_URL/download/?acc=$ACC&format=file&file=${PREFIX}_$d.tar"
done

echo "Due to some access issues you will have to manually download the GSE90063_RAW.tar file from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063"

echo "Once you are done downloading, untar that folder and just do: gunzip > *.gz to unzip all files inside that folder "

cd ../../