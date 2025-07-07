#!/bin/bash

# Script to run condition tokenization for WorldView3 data
# Make sure to update the paths in the config file before running

echo "Starting condition tokenization for WorldView3 data..."

# Run with Hydra configuration
python /Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/scripts/condition_tokenize.py \
    --config-path=configs/condition_tokenization \
    --config-name=worldview3_conditions

echo "Condition tokenization completed!"
