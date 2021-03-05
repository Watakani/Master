#!/bin/bash

wget https://zenodo.org/record/1161203/files/data.tar.gz
tar -xzvf data.tar.gz -C NormalizingFlows/data/unprocessed/
mv NormalizingFlows/data/unprocessed/data/* NormalizingFlows/data/unprocessed/
rm -rf NormalizingFlows/data/unprocessed/data
rm -rf data.tar.gz
echo "All files downloaded correctly and compressed file removed"
