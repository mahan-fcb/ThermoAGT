def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords
import os
import numpy as np
import networkx as nx
from Bio import PDB
import pickle
import re
import pandas as pd

aa_dict = {
    'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords

def positional_encoding(distance, dimension=16):
    encoding = np.zeros(dimension)
    for i in range(dimension):
        encoding[i] = np.sin(distance / (10000 ** ((2 * i) / dimension))) if i % 2 == 0 else np.cos(distance / (10000 ** ((2 * i) / dimension)))
    return encoding

def load_conservation_scores(directory, filename):
    conservation_file = os.path.join(directory, filename.replace('.pdb', '_conservation.npy'))
    if os.path.exists(conservation_file):
        return np.load(conservation_file)
    return None

def create_graph_from_structure(structure, pssm_scores, conservation_scores, threshold=8.0, encoding_dim=16):
    ca_coords = get_residue_ca_coords(structure)
    graph = nx.Graph()

    # Add nodes with combined features
    for i, ((chain_id, residue_id), coord) in enumerate(ca_coords.items()):
        residue = structure[0][chain_id][residue_id]
        resname = residue.get_resname()
        one_hot = np.array(aa_dict.get(resname, [0]*20))  # Convert one-hot encoding to a NumPy array
        pssm = np.array(pssm_scores[i]) if pssm_scores is not None and i < len(pssm_scores) else np.zeros(20)
        conservation = np.array([conservation_scores[i]]) if conservation_scores is not None and i < len(conservation_scores) else np.zeros(1)
        features = np.concatenate((one_hot, pssm, conservation))  # Concatenate features
        graph.add_node((chain_id, residue_id), feature=features)

    # Add edges based on distance threshold
    for (chain_id1, res_id1), coord1 in ca_coords.items():
        for (chain_id2, res_id2), coord2 in ca_coords.items():
            if (chain_id1, res_id1) != (chain_id2, res_id2):
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, encoding_dim)
                    graph.add_edge((chain_id1, res_id1), (chain_id2, res_id2), feature=edge_feature)

    return graph

def process_pdb_files(directory, pssm_pickle_file, conservation_directory):
    parser = PDB.PDBParser(QUIET=True)
    graphs = []

    wild_type_files = {}
    mutant_files = {}

    # Load PSSM scores
    with open(pssm_pickle_file, 'rb') as file:
        pssm_scores_dict = pickle.load(file)

    # Classify files as wild-type or mutant
    for file_name in os.listdir(directory):
        if re.match(r'.+_wild_type\.pdb', file_name):
            wild_type_files[file_name.split('_wild_type')[0]] = file_name
        elif re.match(r'.+_mutant_.+\.pdb', file_name):
            base_name = file_name.split('_mutant_')[0]
            if base_name not in mutant_files:
                mutant_files[base_name] = []
            mutant_files[base_name].append(file_name)

    # Process each mutant file with its corresponding wild type
    for base_name, mutants in mutant_files.items():
        if base_name in wild_type_files:
            wild_type_file = wild_type_files[base_name]
            wild_type_structure = parser.get_structure(base_name + '_wild_type', os.path.join(directory, wild_type_file))

            # Load PSSM and conservation scores for wild type
            wild_type_pssm = pssm_scores_dict.get(wild_type_file.replace('.pdb', '.pssm'), None)
            wild_type_conservation = load_conservation_scores(conservation_directory, wild_type_file)
            wild_type_graph = create_graph_from_structure(wild_type_structure, wild_type_pssm, wild_type_conservation)

            for mutant_file in mutants:
                mutant_structure = parser.get_structure(mutant_file, os.path.join(directory, mutant_file))

                # Load PSSM and conservation scores for mutant
                mutant_pssm = pssm_scores_dict.get(mutant_file.replace('.pdb', '.pssm'), None)
                mutant_conservation = load_conservation_scores(conservation_directory, mutant_file)

                mutant_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)

                graphs.append((wild_type_graph, mutant_graph, mutant_file))

    return graphs

