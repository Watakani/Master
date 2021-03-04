# Master

Add datafiles:
Run command: ./get_data.sh 
  If denied, first run command: chmod 777 get_data.sh
 
Or download the files from here: https://zenodo.org/record/1161203/files/data.tar.gz?download=1
Then copy them into data/unprocessed/

This might take some time to download them, as they are quite large.

Running in a python script or jupyternotebook:
  -Import class that corresponds to a dataset
  -Initiate it without any parameters, and it will preprocess the file and store it in data/preprocessed/
  
 Preprocessing might take some time for some datasets, but after the first initiation, will run fast unless
 you wish to tweak some parameters to the class.
