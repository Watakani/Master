#!/bin/bash

#wget https://zenodo.org/record/1161203/files/data.tar.gz
tar -xzvf data.tar.gz -C data/unprocessed/
mv data/unprocessed/data/* data/unprocessed/
rm -rf data/unprocessed/data
rm -rf data.tar.gz
echo "All files downloaded correctly and compressed file removed"
