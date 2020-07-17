import rdkit
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolHash

print ('RDKit version : ', rdkit.__version__)
print('---------------------')

m1 = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
m2 = Chem.MolFromSmiles('CC(OC1=C(C(=O)O)C=CC=C1)=O')

print('With fingerprint distance')
f1 = GetMorganFingerprintAsBitVect(m1, 4, useChirality=True)
f2 = GetMorganFingerprintAsBitVect(m2, 4, useChirality=True)
print(DataStructs.FingerprintSimilarity(f1,f2))
print(DataStructs.FingerprintSimilarity(f2,f2))  # == 1
print('---------------------')


# or
print('With hashstring')
print(rdMolHash.GenerateMoleculeHashString(m2)==rdMolHash.GenerateMoleculeHashString(m1))
print(rdMolHash.GenerateMoleculeHashString(m1)==rdMolHash.GenerateMoleculeHashString(m1))
print('---------------------')


# or
print('With recomputed SMILES (isomeric and canonical)')
print(Chem.MolToSmiles(m2)==Chem.MolToSmiles(m1))
print(Chem.MolToSmiles(m1)==Chem.MolToSmiles(m1))