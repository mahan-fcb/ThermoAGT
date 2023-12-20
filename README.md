# ThermoAGT
# Data preparation:

All datasets utilized in this study are accessible in the 'Data' folder. All necessary information regarding PDB IDs, DGG, and mutation sites is provided. Wild-type structures can be directly downloaded from the Protein Data Bank. However, for mutant structures, the procedure outlined in the manuscript should be followed. The FoldX algorithm can be employed to generate and relax mutant structures, utilizing their corresponding wild-type structures as templates

Here is a simplified procedure for utilizing FoldX for the generation and relaxation of mutant structures:

https://github.com/shahpr/contingency_entrenchment

# Feature extractions from PDBs:

1- 6 atomic distances and 10 atomic orientions can be derived from PDBs.py file. you need to feed pdb to this py file and then extract these features

2- Physiochemical properties derived from either sequences or PDBs can be extracted from from sequential1.py. 

3- To extract MSA and coevolitionary features. please use the following software:

https://github.com/realbigws/TGT_Package

4- For properties prediction such as SS and RSA, PSSM, PSFM, and ... use:

https://github.com/realbigws/Predict_Property
# Final feature generation
After doing these steps you can use read_feature.py and feature_generation.py file to combine all extract features to each other and save them as torsion.npy, atomic_coordinate.npy, ddg.npy, sequential.npy, and sequence_name.npy. 

# contact map prediction for edges generation in sequence-based models:
In structure-based model, you can use pairwise distances between beta carbon as edges. However, for seuqence-based models, you need to predict contact map (pairwise distances) to be used as edges. For contact map prediciton, you can use our recent model, CGAN-Cmap here:

https://github.com/mahan-fcb/CGAN-Cmap-A-protein-contact-map-predictor

# Model traning: 

Thermal_train.py is provided all model architectures and data loaders. you can train your model from strach by using this command: python Thermal_train.py. please note that you need to geenrate training data and chnage the names of npy files in the Thermal_train.py based on your generated data.

Thermal_test.py is provided all saved models and data loaders. you can test your model by using this command: python Thermal_test.py. please note that you need to geenrate test data and chnage the names of npy files in the Thermal_test.py based on your generated data.

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

