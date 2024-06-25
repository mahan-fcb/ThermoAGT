import numpy as np
import pickle

def find_mutation_site(wild_type, mutant):
    """
    Identify the mutation site by comparing the one-hot encoded residues of wild type and mutant.
    Returns the index of the mutation site.
    """
    for i in range(len(wild_type)):
        if not np.array_equal(wild_type[i][:20], mutant[i][:20]):  # Considering only the one-hot part of the node features
            return i
    return -1

def extract_subgraph(data, mutation_site, window_size=5):
    """
    Extract subgraph centered around the mutation site.
    """
    start = max(0, mutation_site - window_size)
    end = min(len(data['node_features']), mutation_site + window_size + 1)

    subgraph = {
        'node_features': data['node_features'][start:end],
        'edge_index': [],
        'edge_features': []
    }

    node_mapping = {i: idx for idx, i in enumerate(range(start, end))}
    for i in range(data['edge_index'].shape[1]):
        src, dst = data['edge_index'][:, i]
        if start <= src < end and start <= dst < end:
            subgraph['edge_index'].append([node_mapping[src], node_mapping[dst]])
            subgraph['edge_features'].append(data['edge_features'][i])

    subgraph['edge_index'] = np.array(subgraph['edge_index']).T
    subgraph['edge_features'] = np.array(subgraph['edge_features'])

    return subgraph

# Load the dataset from the pickle file
with open('data_Ssym.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Process each datapoint
processed_data = []
for data in dataset:
    mutation_site = find_mutation_site(data['wild_type']['node_features'], data['mutant']['node_features'])

    if mutation_site == -1:
        print("No mutation found for data point with wild type name:", data['wild_type_name'])
        continue

    wild_type_subgraph = extract_subgraph(data['wild_type'], mutation_site)
    mutant_subgraph = extract_subgraph(data['mutant'], mutation_site)

    processed_data.append({
        'wild_type': wild_type_subgraph,
        'mutant': mutant_subgraph,
        'ddg': data['ddg'],
        'wild_type_name': data['wild_type_name'],
        'mutant_name': data['mutant_name']
    })

# Save the processed data back to a pickle file
with open('dataset_Ssym.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print(f"Processed {len(processed_data)} datapoints.")

