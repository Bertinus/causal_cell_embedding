# causal_gene_embedding

## TODO

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

**build_env()**
input : number of environments
output : matrix

**methods for sampling** :

**sample(env)** method which is used to get samples from the generator

## Intersting links


