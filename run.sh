#!/bin/bash
echo "Running experiments using: $(which python)"

ai.causalcell train --config ai/causalcell/config/experiment.yml

#ai.causalcell train-skopt --config ai/causalcell/config/experiment.yml --n-iter ${N_ITER}
# typically N_iter = 30