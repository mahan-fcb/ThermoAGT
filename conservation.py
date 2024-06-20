import numpy as np
import pickle
import os

def calculate_conservation(pssm_array):
    # Normalize the PSSM scores to probabilities
    probabilities = np.exp(pssm_array) / np.exp(pssm_array).sum(axis=1, keepdims=True)
    
    # Calculate entropy for each position
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
    
    # Calculate conservation as 1 - normalized entropy
    max_entropy = np.log(pssm_array.shape[1])
    conservation_scores = 1 - (entropy / max_entropy)
    
    return conservation_scores

def calculate_and_save_conservation_from_pickle(pickle_file, output_directory):
    # Load the PSSM arrays from the pickle file
    with open(pickle_file, 'rb') as file:
        pssm_arrays = pickle.load(file)
    
    # Calculate conservation scores and save to files
    for filename, pssm_array in pssm_arrays.items():
        conservation_scores = calculate_conservation(pssm_array)
        conservation_file = os.path.join(output_directory, filename.replace('.pssm', '_conservation.npy'))
        np.save(conservation_file, conservation_scores)
        print(f"Conservation scores saved to {conservation_file}")

# Ensure the correct path to the pickle file and output directory
pickle_file_path = 'pssm_s669.pkl'  # Replace with the correct path if necessary
output_directory = '.\\cons_s669'  # Replace with the correct path if necessary

# Calculate and save conservation scores
calculate_and_save_conservation_from_pickle(pickle_file_path, output_directory)
