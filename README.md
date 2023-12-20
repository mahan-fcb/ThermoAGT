# ThermoAGT
# Data preparation:

All datasets utilized in this study are accessible in the 'Data' folder. All necessary information regarding PDB IDs, DGG, and mutation sites is provided. Wild-type structures can be directly downloaded from the Protein Data Bank. However, for mutant structures, the procedure outlined in the manuscript should be followed. The FoldX algorithm can be employed to generate and relax mutant structures, utilizing their corresponding wild-type structures as templates

Here is a simplified procedure for utilizing FoldX for the generation and relaxation of mutant structures:

https://github.com/shahpr/contingency_entrenchment

# Feature extractions from PDBs:

# Deriving Atomic Distances and Orientations from PDBs:

Utilize the PDBs.py file to extract six atomic distances and ten atomic orientations.

Feed the PDB file into this Python script to obtain these features.

# Extraction of Physiochemical Properties:

For physiochemical properties, use the sequential1.py script.

Extract physiochemical properties either from sequences or PDB files.

# MSA and Coevolutionary Features Extraction:

Employ the following software to extract MSA (Multiple Sequence Alignment) and coevolutionary features.

https://github.com/realbigws/TGT_Package

For properties prediction such as SS and RSA, PSSM, PSFM, and ... use:

https://github.com/realbigws/Predict_Property
# Final feature generation
After completing the aforementioned steps, the next phase involves combining all extracted features using the read_feature.py and feature_generation.py files. The resulting combined features can be saved as torsion.npy, atomic_coordinate.npy, ddg.npy, sequential.npy, and sequence_name.npy.

# Contact map prediction for edges generation in sequence-based models:
In a structure-based model, pairwise distances between beta carbons serve as edges. However, for sequence-based models, the prediction of contact maps (pairwise distances) is necessary for edge formation. To accomplish contact map prediction, our recent model, CGAN-Cmap, can be employed. Find the model here:

https://github.com/mahan-fcb/CGAN-Cmap-A-protein-contact-map-predictor

# Model traning: 


# Training Your Model:

To train your model from scratch, use the provided Thermal_train.py script, which contains all model architectures and data loaders.

Execute the training command: python Thermal_train.py.

Ensure you generate the required training data and update the names of the .npy files in Thermal_train.py according to your specific dataset.

# Testing Your Model:

For testing your model, utilize the Thermal_test.py script, which includes saved models and data loaders.

Execute the testing command: python Thermal_test.py.

Make sure to generate the necessary test data and update the names of the .npy files in Thermal_test.py based on your specific dataset.

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

