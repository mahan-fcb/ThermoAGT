import os
import numpy as np
import pickle

def parse_pssm_to_array(pssm_file):
    with open(pssm_file, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the PSSM matrix
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("Last position-specific scoring matrix"):
            start_idx = idx + 3
            break

    # Parse the PSSM matrix
    pssm_data = []
    for line in lines[start_idx:]:
        if line.strip() == "":
            break
        parts = line.split()
        scores = list(map(int, parts[2:22]))
        pssm_data.append(scores)

    # Convert to NumPy array
    pssm_array = np.array(pssm_data)
    return pssm_array

def process_pssm_files(directory):
    pssm_arrays = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pssm"):
            file_path = os.path.join(directory, filename)
            pssm_array = parse_pssm_to_array(file_path)
            pssm_arrays[filename] = pssm_array

    # Save the dictionary to a pickle file
    with open(os.path.join(directory, 'pssm_s669.pkl'), 'wb') as pickle_file:
        pickle.dump(pssm_arrays, pickle_file)

    print(f"Processed {len(pssm_arrays)} PSSM files and saved to 'pssm_arrays.pkl'")

# Replace with your directory containing .pssm files
pssm_directory = 'pssm_s669'
process_pssm_files(pssm_directory)
