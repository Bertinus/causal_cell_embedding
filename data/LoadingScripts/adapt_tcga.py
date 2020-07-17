import numpy as np
import pandas as pd

########################################################################################################################
# Load data
########################################################################################################################

print("Adapt TCGA: Loading data. Might take some time...")

# TCGA gene expression matrix
data = pd.read_csv('Data/Downstream_Tasks/TcgaTargetGtex_rsem_gene_tpm', sep='\t')

# Load Ensembl Id conversion table
conversion_table = pd.read_csv('Data/Downstream_Tasks/ensembl_names.txt', sep='\t')

# Get list of landmark genes
gene_info = pd.read_csv("Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_gene_info.txt", sep="\t")

########################################################################################################################
# Build conversion map
########################################################################################################################

print("Adapt TCGA: build conversion map")

# Build name to ensembl dictionary
name_to_ensembl_dict = {}
for l in conversion_table.iterrows():
    name_to_ensembl_dict[l[1]['Approved symbol']] = l[1]['Ensembl gene ID']

# Manually add  landmark genes that are not in the ensembl ID table
not_matched_dict = {"EPRS": "ENSG00000136628",
                    "AARS": "ENSG00000090861",
                    "TOMM70A": "ENSG00000154174",
                    "KIAA0196": "ENSG00000164961",
                    "KIAA0907": "ENSG00000132680",
                    "PAPD7": "ENSG00000112941",
                    "IKBKAP": "ENSG00000070061",
                    "HIST2H2BE": "ENSG00000184678",
                    "WRB": "ENSG00000182093",
                    "KIAA0355": "ENSG00000166398",
                    "TMEM5": "ENSG00000118600",
                    "HDGFRP3": "ENSG00000166503",
                    "PRUNE": "ENSG00000143363",
                    "HIST1H2BK": "ENSG00000197903",
                    "HN1L": "ENSG00000206053",
                    "H2AFV": "ENSG00000105968",
                    "KIF1BP": "ENSG00000198954",
                    "KIAA1033": "ENSG00000136051",
                    "FAM69A": "ENSG00000154511",
                    "TMEM110": "ENSG00000213533",
                    "ATP5S": "ENSG00000125375",
                    "SQRDL": "ENSG00000137767",
                    "TMEM2": "ENSG00000135048",
                    "ADCK3": "ENSG00000163050",
                    "NARFL": "ENSG00000103245",
                    "FAM57A": "ENSG00000167695",
                    "LRRC16A": "ENSG00000079691",
                    "FAM63A": "ENSG00000143409",
                    "TSTA3": "ENSG00000104522"}


name_to_ensembl_dict = {**name_to_ensembl_dict, **not_matched_dict}
landmark_ensembl_dict = {name: name_to_ensembl_dict[name]
                         for name in gene_info[gene_info['pr_is_lm'] == 1]['pr_gene_symbol']}

landmark_ensembl_to_name_dict = {landmark_ensembl_dict[name]: name for name in landmark_ensembl_dict.keys()}

########################################################################################################################
# Retrieve part of TCGA matrix that corresponds to landmark genes
########################################################################################################################

print("Adapt TCGA: modify TCGA matrix")

# Remove version of the ensembl ID in TCGA data
data['ensembl'] = data['sample'].apply(lambda s: s.split('.')[0])

# Restrict to landmark genes
data_lamdmark_genes = data[data['ensembl'].apply(lambda s: s in landmark_ensembl_dict.values())]
data_lamdmark_genes = data_lamdmark_genes.drop(['sample'], axis=1)

# Add gene names to the matrix
data_lamdmark_genes['name'] = data_lamdmark_genes['ensembl'].apply(lambda s: landmark_ensembl_to_name_dict[s])

data_lamdmark_genes = data_lamdmark_genes.set_index('name')
data_lamdmark_genes = data_lamdmark_genes.drop(['ensembl'], axis=1)

# Save
data_lamdmark_genes.to_csv("Data/Downstream_Tasks/tcga_landmark_genes.csv")