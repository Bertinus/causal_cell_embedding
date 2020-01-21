#!/bin/bash
conda activate cge
echo "Running mnist using: $(which python)"

ai.cge train --config ai/seg/config/mnist.yml
