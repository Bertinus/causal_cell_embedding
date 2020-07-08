#!/bin/bash
echo "Running experiments using: $(which python)"


cp -r /network/tmp1/bertinpa/causal_cell_embedding/L1000_PhaseI $SLURM_TMPDIR
cp -r /network/tmp1/bertinpa/causal_cell_embedding/L1000_PhaseII $SLURM_TMPDIR

unlink ~/Documents/causal_cell_embedding/Data/L1000_PhaseI
unlink ~/Documents/causal_cell_embedding/Data/L1000_PhaseII

ln -s $SLURM_TMPDIR/L1000_PhaseI ~/Documents/causal_cell_embedding/Data/L1000_PhaseI
ln -s $SLURM_TMPDIR/L1000_PhaseII ~/Documents/causal_cell_embedding/Data/L1000_PhaseII

echo "Finished copying the dataset on local node"

ai.causalcell train --config ai/causalcell/config/l1000_env_prior_VAE.yml

unlink ~/Documents/causal_cell_embedding/Data/L1000_PhaseI
unlink ~/Documents/causal_cell_embedding/Data/L1000_PhaseII

ln -s /network/tmp1/bertinpa/causal_cell_embedding/L1000_PhaseI ~/Documents/causal_cell_embedding/Data/L1000_PhaseI
ln -s /network/tmp1/bertinpa/causal_cell_embedding/L1000_PhaseII ~/Documents/causal_cell_embedding/Data/L1000_PhaseII

#ai.causalcell train-skopt --config ai/causalcell/config/experiment.yml --n-iter ${N_ITER}
# typically N_iter = 30