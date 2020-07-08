import numpy as np
import pandas as pd
import requests
import re
from tqdm import tqdm


drugbank_biotech = pd.read_csv("Data/DRKG/drugbank_info/drugbank_biotech.txt", header=None)
drugbank_small_molecules = pd.read_csv("Data/DRKG/drugbank_info/drugbank_small_molecule.txt", header=None)

all_drugbank_ids = list(np.concatenate((drugbank_biotech, drugbank_small_molecules)).T[0])

dict_id_to_name = {}

# Loop over all drugbank IDs and retrieve synonyms
for id in tqdm(all_drugbank_ids):
    xml_file = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/drugbank/' + id + '/XML')

    content = str(xml_file.content)

    all_indices_start = [m.start() for m in re.finditer('<PC-Substance_synonyms_E>', content)]
    all_indices_end = [m.start() for m in re.finditer('</PC-Substance_synonyms_E>', content)]

    assert len(all_indices_start) == len(all_indices_start)

    # Find all synonymes
    synonyms = []
    for i in range(len(all_indices_start)):
        name = content[all_indices_start[i] + 25 : all_indices_end[i]]
        synonyms.append(name)

    dict_id_to_name[id] = synonyms

np.save('Data/DRKG/dict_id_to_name.npy', dict_id_to_name)