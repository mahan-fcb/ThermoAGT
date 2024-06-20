import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import defaultdict
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.SASA import ShrakeRupley
import mdtraj as md
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
from MDAnalysis.lib.distances import calc_dihedrals

# Define a dictionary for maximum accessible surface areas (max_acc)
max_acc = {
    'ALA': 121.0, 'ARG': 265.0, 'ASN': 187.0, 'ASP': 187.0, 'CYS': 148.0,
    'GLN': 214.0, 'GLU': 214.0, 'GLY': 97.0, 'HIS': 216.0, 'ILE': 195.0,
    'LEU': 191.0, 'LYS': 230.0, 'MET': 203.0, 'PHE': 228.0, 'PRO': 154.0,
    'SER': 143.0, 'THR': 163.0, 'TRP': 264.0, 'TYR': 255.0, 'VAL': 165.0
}

# Kyte-Doolittle hydrophobicity scale
kyte_doolittle_hydrophobicity = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
    'W': -0.9, 'Y': -1.3
}

# Get molecular weight for each amino acid
molecular_weights = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2, 'G': 75.1,
    'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2, 'M': 149.2, 'N': 132.1,
    'P': 115.1, 'Q': 146.2, 'R': 174.2, 'S': 105.1, 'T': 119.1, 'V': 117.1,
    'W': 204.2, 'Y': 181.2
}

def calculate_sasa(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    sasa_calculator = ShrakeRupley()
    sasa_calculator.compute(structure, level="R")
    
    sasa_per_residue = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.id[1]
                sasa_per_residue[res_id] = residue.sasa
    return sasa_per_residue

def calculate_rsa(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    
    sasa_per_residue = calculate_sasa(pdb_file)
    
    rsa_per_residue = {}
    for res_id, sasa in sasa_per_residue.items():
        residue = structure[0]["A"][res_id]
        res_name = residue.resname
        max_sasa = max_acc.get(res_name, max_acc['ALA'])  # Default to ALA if unknown
        rsa = sasa / max_sasa
        rsa_per_residue[res_id] = rsa
    return rsa_per_residue

def calculate_secondary_structure(pdb_file):
    traj = md.load(pdb_file)
    secondary_structure = md.compute_dssp(traj)
    residues = traj.topology.residues
    
    ss_map = {'C': 0, 'H': 1}
    sec_struct_per_residue = {residue.index + 1: ss_map.get(sec_str, 2) for residue, sec_str in zip(residues, secondary_structure[0])}
    return sec_struct_per_residue

def calculate_residue_properties(sequence):
    properties = []
    analysis = ProteinAnalysis(str(sequence))
    
    for i, aa in enumerate(sequence):
        res_properties = [
            molecular_weights.get(aa, 0),
            kyte_doolittle_hydrophobicity.get(aa, 0),
            analysis.charge_at_pH(7.0) / len(sequence),
            kyte_doolittle_hydrophobicity.get(aa, 0),
        ]
        properties.append(res_properties)
    return properties

def calculate_b_factors(pdb_file):
    u = mda.Universe(pdb_file)
    b_factors_per_residue = defaultdict(list)
    
    for residue in u.residues:
        res_id = residue.resid
        b_factors_per_residue[res_id].extend(residue.atoms.tempfactors)
    
    average_b_factors = {res_id: np.mean(b_factors) for res_id, b_factors in b_factors_per_residue.items()}
    return average_b_factors

def calculate_omega(u, protein):
    n_atoms = protein.select_atoms('name N')
    ca_atoms = protein.select_atoms('name CA')
    c_atoms = protein.select_atoms('name C')
    
    omega_angles = []
    for i in range(1, len(n_atoms) - 1):
        a1 = c_atoms[i - 1].position
        a2 = n_atoms[i].position
        a3 = ca_atoms[i].position
        a4 = c_atoms[i].position
        omega = calc_dihedrals(a1, a2, a3, a4)
        omega_angles.append(omega)
    
    return omega_angles

def extract_angles_from_protein(file_path):
    u = mda.Universe(file_path)
    protein = u.select_atoms('protein')
    r = Ramachandran(protein).run()
    phi_psi_angles = r.angles[0]
    
    omega_angles = calculate_omega(u, protein)
    
    return phi_psi_angles, omega_angles

def add_sine_cosine_to_dict(angles_combined):
    new_angles = []
    for angles in angles_combined:
        phi, psi, omega = angles
        new_angles.append([
            phi, psi, omega,
            np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)),
            np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi)),
            np.sin(np.deg2rad(omega)), np.cos(np.deg2rad(omega))
        ])
    return new_angles

def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    ppb = PPBuilder()
    sequences = [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    return "".join(sequences)

def process_pdb_files(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdb'):
            file_path = os.path.join(folder_path, filename)
            
            # Extract sequence
            sequence = extract_sequence_from_pdb(file_path)
            num_residues = len(sequence)

            # Calculate properties
            rsa_results = calculate_rsa(file_path)
            sasa_results = calculate_sasa(file_path)
            secondary_structure = calculate_secondary_structure(file_path)
            b_factors = calculate_b_factors(file_path)
            phi_psi_angles, omega_angles = extract_angles_from_protein(file_path)
            angles_combined = []
            for res_id, (phi, psi) in enumerate(phi_psi_angles):
                omega = omega_angles[res_id] if res_id < len(omega_angles) else None
                angles_combined.append([phi, psi, omega])
            angles_combined = add_sine_cosine_to_dict(angles_combined)
            residue_properties = calculate_residue_properties(sequence)

            # Ensure all lists have the same length as num_residues
            angles_combined = angles_combined[:num_residues]
            while len(angles_combined) < num_residues:
                angles_combined.append([0] * 9)  # Append zeroes if fewer angles

            all_properties = []
            for i in range(num_residues):
                properties = [
                    rsa_results.get(i + 1, 0),
                    sasa_results.get(i + 1, 0),
                    secondary_structure.get(i + 1, 2),
                    b_factors.get(i + 1, 0),
                ]
                properties.extend(angles_combined[i])
                properties.extend(residue_properties[i])
                all_properties.append(properties)
            
            # Collect all properties in a dictionary
            protein_data = {
                'protein_name': filename,
                'properties': all_properties
            }
            results.append(protein_data)
    

    return results

