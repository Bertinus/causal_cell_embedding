from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw

"""
CLustering experiments with the fingerprints
See : https://www.macinchem.org/reviews/clustering/clustering.php

"""


# Define clustering setup
def cluster_fps(fps, cutoff=0.2):
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs


if __name__ == "__main__":

    A = pd.read_csv("/Users/paul/PycharmProjects/causal_cell_embedding/Data/L1000_PhaseI/GSE92742_Broad_LINCS"
                    "/GSE92742_Broad_LINCS_pert_info.txt", sep="\t")

    ms = [Chem.MolFromSmiles(smile) for smile in A["canonical_smiles"].unique()]
    ms = [i for i in ms if i]  # Remove None values
    # fp = Chem.RDKFingerprint(m, fpSize=256, maxPath=4)
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in ms]
    clusters = cluster_fps(fps, cutoff=0.8)  # Original cutoff: 0.4
    print("Number of clusters", len(clusters))

    cpt = 0
    for c in clusters:
        if (len(c) > 10) and (len(c) < 20):
            Draw.MolsToGridImage([ms[i] for i in c]).show()
            cpt += 1
            if cpt > 5:
                break
