import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm


def get_name_for_l1000(node):
    """
    given the name in DRKG, transform the name of the node so that it corresponds to L1000 IDs
    """
    if node.startswith("Gene"):
        return node[6:]
    else:
        return list(dict_id_to_single_name[node[10:]])[0]


def corresponds_to_a_node(candidate):
    """
    Similar to get_name_to_a_node, but tests all candidates
    :return: (False, 0) if the candidate is not a node
    """
    if candidate.startswith("Compound") and candidate[10:] in dict_id_to_single_name.keys():
        return True, list(dict_id_to_single_name[candidate[10:]])[0]

    if candidate.startswith("Gene") and candidate[6:] in landmark_gene_list:
        return True, candidate[6:]
    else:
        return False, 0


if __name__ == '__main__':

    print("Should take approximately 5min to compute. The code is not optimized, but you only need to run it once")

    ####################################################################################################################
    # Load DRKG
    ####################################################################################################################

    drkg_file = 'data/drkg/drkg.tsv'
    df = pd.read_csv(drkg_file, sep="\t", header=None)

    # Load dictionary referencing all names of drugbank IDs
    dict_id_to_names = np.load('Data/DRKG/dict_id_to_name.npy', allow_pickle=True).item()

    ####################################################################################################################
    # Load L1000
    ####################################################################################################################

    # Load L1000
    pert_info_1 = pd.read_csv("Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_pert_info.txt", sep="\t",
                              index_col="pert_id", usecols=["pert_id", "pert_iname", "canonical_smiles"])
    pert_info_2 = pd.read_csv("Data/L1000_PhaseII/GSE70138_Broad_LINCS/pert_info_2017-03-06.txt", sep="\t",
                              index_col="pert_id", usecols=["pert_id", "pert_iname", "canonical_smiles"])
    pert_info = pd.concat([pert_info_1, pert_info_2], sort=False)

    l1000_names = list(pert_info['pert_iname'].values)
    l1000_names = [name.lower() for name in l1000_names]

    # Get list of landmark genes
    gene_info = pd.read_csv('Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_gene_info.txt', sep="\t")
    landmark_gene_list = list(gene_info[gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str))

    ####################################################################################################################
    # Create dict mapping of IDs for drugs
    ####################################################################################################################

    # Create dict where values are restricted to names that match a l1000 entry
    dict_id_to_single_name = {i: set([name.lower() for name in dict_id_to_names[i]]).intersection(l1000_names)
                              for i in dict_id_to_names.keys()}

    # Remove keys with empty sets
    dict_id_to_single_name = {i: dict_id_to_single_name[i] for i in dict_id_to_single_name.keys()
                              if bool(dict_id_to_single_name[i])}

    ####################################################################################################################
    # Create list of all node names
    ####################################################################################################################

    # Restrict ourselves to STRING and DRUGBANK
    df_string_drugbank = df[df[1].apply(lambda s: s[:6] == 'STRING' or s[:8] == 'DRUGBANK')]

    # Create list of all nodes
    all_nodes = list(df_string_drugbank[0].values) + list(df_string_drugbank[2].values)

    # Remove nodes that are not gene or compound
    all_nodes = [node for node in all_nodes if node.startswith("Compound") or node.startswith("Gene")]

    # restrict to unique values
    all_nodes = list(np.unique(all_nodes))

    # Remove compounds that do not have a name in L1000
    all_nodes = [node for node in all_nodes if (node.startswith("Compound") and
                                                node[10:] in dict_id_to_single_name.keys())
                 or node.startswith("Gene")]

    # Remove genes that are not landmark
    all_nodes = [node for node in all_nodes if node.startswith("Compound")
                 or (node.startswith("Gene") and node[6:] in landmark_gene_list)]

    all_node_names = [get_name_for_l1000(node) for node in all_nodes]

    ####################################################################################################################
    # Create list of all edges
    ####################################################################################################################

    all_edges = []

    for line in tqdm(df_string_drugbank.iterrows()):
        is_node_0, name_0 = corresponds_to_a_node(line[1][0])
        is_node_2, name_2 = corresponds_to_a_node(line[1][2])
        if is_node_0 and is_node_2:
            all_edges.append((name_0, name_2))

    ####################################################################################################################
    # Create networkx graph and save it
    ####################################################################################################################

    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(all_node_names)

    # Add edges
    G.add_edges_from(all_edges)

    print("number of nodes", G.number_of_nodes(), "number of edges", G.number_of_edges())

    nx.write_gpickle(G, "data/drkg/nx_drkg_graph.gpickle")
