# ThermoAGT
# Data preparation:

All datasets used in this study are available in Data folder. Because of size of datasets, links for downloading them are available.

# Feature extractions from PDBs:

1- 6 atomic distances and 10 atomic orientions can be derived from PDBs.py file. you need to feed pdb to this py file and then extract these features

2- physiochemical properties derived from either sequences or PDBs can be extracted from from Physio.py. 

3- To extract MSA and coevolitionary features. please use the following software:

https://github.com/realbigws/TGT_Package

4- for properties prediction such as SS and RSA, PSSM, PSFM, and ... use:

https://github.com/realbigws/Predict_Property
# final feature generation
after doing these steps you can use read_feature.py and feature_generation.py file to combine all extract features to each other and save them as torsion.npy, atomic_coordinate.npy, ddg.npy, sequential.npy, and sequence_name.npy.

# Model traning: 

model.py is provided all model architectures and data loaders. you can train your model from strach by using this command: python train.py

your model is saved in the MODEL directory. and you can test your data using this command: python test.py

# python3.9

pytorch2.1

numpy

matplotlib

pickle

Pytorch geometric 

Quntiport

biopython

# Other packages:

alnstats

fasta2aln

Loadhmm

CCMpred

TGT_Package

Predict_Property

hh-suite

# Install python environment:

# CUDA 10.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

