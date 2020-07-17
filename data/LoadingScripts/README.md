# Information for data retrieval and processing

## Download using provided scripts

Loading scripts are provided to download L1000 data as well as DRKG data

## Files already provided

### Gene name conversion

The ````ensembl_names.txt```` file has been downloaded from https://biomart.genenames.org/ on July 10th 2020 and is 
available in ````Data/Downstream_Tasks````.

### Downstream tasks

Downstream tasks are taken from this paper https://www.cell.com/immunity/comments/S1074-7613(18)30121-3

The file is provided here: ````Data/Downstream_Tasks/1-s2.0-S1074761318301213-mmc2.csv````

## Manual download: TCGA

TCGA original paper: https://www.nature.com/articles/s41587-020-0546-8

TCGA data is downloaded from [link](https://xenabrowser.net/datapages/?dataset=TcgaTargetGtex_rsem_gene_tpm&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=http%3A%2F%2F127.0.0.1%3A7222)

You should store the file in ````Data/Downstream_Tasks````

## Data processing

You need to execute ````get_drugbank_compound_name.py```` and ````create_DRKG_nx_graph.py```` in order 
to process the DRKG graph and make it compatible with L1000 data