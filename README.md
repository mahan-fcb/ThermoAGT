# ThermoAGT
# Data preparation:

All datasets utilized in this study are accessible in the 'Data' folder. All necessary information regarding PDB IDs, DGG, and mutation sites is provided. Wild-type structures can be directly downloaded from the Protein Data Bank. However, for mutant structures, the procedure outlined in the manuscript should be followed. The FoldX algorithm can be employed to generate and relax mutant structures, utilizing their corresponding wild-type structures as templates

Here is a simplified procedure for utilizing FoldX for generation and relaxation of mutant structures:

https://github.com/shahpr/contingency_entrenchment
all generated pdbs are provided in the zenodo link
# Feature extractions from PDBs:

# Deriving Atomic Distances and Orientations from PDBs:

For structure-based model, you need to extract distance between Ca atoms and all angles 
For sequence-based model, you need to use predict disnatcne and contact map. 

# Extraction of Physiochemical Properties:

you need to use DSSP to extract SS and SASA for structures
you need to use BLAST+ or hhblits for MSA and pssm calculations.

Extract physiochemical properties either from sequences or PDB files are provided in the codes.

# MSA and Coevolutionary Features Extraction for sequences:

Employ the following software to extract MSA (Multiple Sequence Alignment) and coevolutionary features.

https://github.com/realbigws/TGT_Package

For properties prediction such as SS and RSA, PSSM, PSFM, and ... use:

https://github.com/realbigws/Predict_Property
# Contact map prediction for edges generation in sequence-based models:
In a structure-based model, pairwise distances between beta carbons serve as edges. However, for sequence-based models, the prediction of contact maps (pairwise distances) is necessary for edge formation. To accomplish contact map prediction, our recent model, CGAN-Cmap, can be employed. Find the model here:

https://github.com/mahan-fcb/CGAN-Cmap-A-protein-contact-map-predictor

# Final feature generation
After completing the aforementioned steps, the next phase involves combining all extracted features using the graph_gen.py, feature_gen.py, and combining_features_to_graph.py



# Model traning: 

After generation of graphs (graph for whole chain), to train and test the model please use the train.py and prediction.py

please note that in newest version, we build graph for whole chain and we updated codes. Then, in the final step, you can extract 11 residue graphs for mutation sites (SubGraph_extraction.py code). 

If you have any questions please contact me: mohammad.madani@uconn.edu

