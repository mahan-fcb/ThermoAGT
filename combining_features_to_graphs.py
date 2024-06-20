import pickle
import numpy as np

# Load the graph data and the properties data from pickle files
with open('data_p53.pkl', 'rb') as f:
    graph_data = pickle.load(f)

with open('protein_propertiesp53.pkl', 'rb') as f:
    properties_data = pickle.load(f)

# Convert properties data to a dictionary for quick lookup
properties_dict = {item['protein_name']: item['properties'] for item in properties_data}

# Function to append properties to node features
def append_properties_to_node_features(node_features, properties, protein_name):
    num_nodes = node_features.shape[0]
    properties_array = np.array(properties)
    
    # Handle mismatch by appending a zero line if properties length is one less than num_nodes
    if len(properties_array) == num_nodes - 1:
        print(f"Appending zero line to properties for protein: {protein_name}")
        zero_line = np.zeros_like(properties_array[0])
        properties_array = np.vstack([properties_array, zero_line])
    
    # Ensure properties array is the same length as the number of nodes
    if len(properties_array) != num_nodes:
        print(f"Error in protein: {protein_name}")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of properties: {len(properties_array)}")
        raise ValueError("Number of properties does not match the number of nodes")
    
    # Append properties to node features
    updated_node_features = np.hstack((node_features, properties_array))
    return updated_node_features

# Iterate through the graph data and append properties to node features
for data in graph_data:
    wild_type_name = data['wild_type_name']
    mutant_name = data['mutant_name']

    if wild_type_name in properties_dict:
        wild_type_properties = properties_dict[wild_type_name]
        try:
            data['wild_type']['node_features'] = append_properties_to_node_features(
                data['wild_type']['node_features'], wild_type_properties, wild_type_name)
        except ValueError:
            continue

    if mutant_name in properties_dict:
        mutant_properties = properties_dict[mutant_name]
        try:
            data['mutant']['node_features'] = append_properties_to_node_features(
                data['mutant']['node_features'], mutant_properties, mutant_name)
        except ValueError:
            continue

