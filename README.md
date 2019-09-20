# causal_gene_embedding

To use the spec file to create an identical **conda environment** on your machine:
'''
conda create --name causalcell --file spec-file.txt
'''

## TODO

Add a diagram to the paper
Read papers and add summaries to the git repo (as md files) in the folder Papers
Investigate datasets
Keep working on the ideas : 
Define different experiments we could make

### Datasets

- ```Mambo``` - a framework to synthesize data from various data sources in order to construct and represent multimodal networks
 [Link](http://snap.stanford.edu/mambo/#tutorial).

- ``` Stanford Biomedical Network Dataset Collection ``` [Link](http://snap.stanford.edu/biodata/index.html)

### Explore the L1000 API

https://github.com/dhimmel/lincs/tree/abcb12f942f93e3ee839e5e3593f930df2c56845

### Code a synthetic data generator

Remarks : 
- We start with categorical latent variables (binary at first)
- Should we add noise in the generative process of latent variables + genes?

#### objects :

Gene_expresion_generator : initialized with number of latent variables and number of genes (among other things)
**class attributes :**
- a function object F wich will be used in the generation process (take linear function at first)
- a directed graph objet representing latent DAG + parameters of F for each non-root node in the graph
- a binary matrix of size (number of latent variables, number of genes) which encodes which edges exist

**methods for initialization** :

**build_latent _dag()**
**input** : number of latent variables, some parameters to define how we generate edges of the DAG (define some probability distribution over edges). The graph should not contain any cycle !
categorical case : each conditional probabilty P(l | PA(l)) will be generated from a family of functions that we should define (linear, NN etc)
continuous case : we define the functions that will output the value of latent variables directly. Add noise as input
**output** : a directed graph objet (networkx) whose nodes are the latent variables

**build latent_gene_connexions()**
**input** : number of latent variables, number of genes, some distribution over edges
**Output** : a binary matrix of size (number of latent variables, number of genes) which encodes which edges exist

**build_env()** TODO
input : number of environments
output : matrix

**methods for sampling** :

**sample(env)** TODO
method which is used to get samples from the generator

## Intersting links

- python library for single cell data analysis https://scanpy.readthedocs.io/en/stable/
- task scheduler https://trello.com/en
- analysis toolkit for LINCS L1000 https://github.com/dhimmel/lincs/tree/abcb12f942f93e3ee839e5e3593f930df2c56845
- toolkit to analyse dynamics of scRNAseq https://buildmedia.readthedocs.org/media/pdf/scvelo/latest/scvelo.pdf
- collaboration ? https://mhi-omics.org/people/julie-hussin/
- Fabian Theis https://www.helmholtz-muenchen.de/icb/institute/staff/staff/ma/2494/Prof.%20Dr.%20Dr.-Theis/index.html
- Course ML for healthcare https://mlhc19mit.github.io/
- jupyter notebooks for RNAseq https://amp.pharm.mssm.edu/biojupies/
- pytorch metalearning library https://github.com/tristandeleu/pytorch-meta
- Depmap https://depmap.org/portal/achilles/

## People to talk to for this project
- Fabian Theis and his student David Fischer
- David Gifford
- Alan Aspuru Guzik

