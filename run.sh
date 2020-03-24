#!/bin/bash
conda activate cge
echo "Running mnist using: $(which python)"

ai.cge train --config ai/seg/config/mnist.yml

#ai.cge train-skopt --config ai/semrep/config/classification/deeplesion/deeplesion_clf_semrep.yml --n-iter ${N_ITER}
# typically N_iter = 30