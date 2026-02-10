#!/bin/bash

# Create temp directory
mkdir -p smatch_temp

# Download and extract stog tar.gz file
wget -O stog.tar.gz https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz
tar -xzf stog.tar.gz -C smatch_temp

# Clean up tar file
rm stog.tar.gz

echo "STOG model extracted to smatch_temp/"