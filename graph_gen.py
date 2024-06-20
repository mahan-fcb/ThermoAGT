
def graph_to_gnn_format(graph):
    node_list = list(graph.nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    nodes = np.array([data['feature'] for _, data in graph.nodes(data=True)])
    edges = np.array([(node_index[u], node_index[v]) for u, v in graph.edges]).T  # Transpose to get shape (2, M)
    edge_features = np.array([data['feature'] for _, _, data in graph.edges(data=True)])
    return nodes, edges, edge_features

def save_graphs_as_torch_data(graphs, csv_file, output_file):
    df = pd.read_csv(csv_file)
    dataset = []

    for wild_type_graph, mutant_graph, filename in graphs:
        match = re.match(r'(.+)_([A-Z])_mutant_(.+)\.pdb', filename)
        if match:
            pdb_id, chain, mutation = match.groups()
            ddg_row = df[(df['P'] == pdb_id) & (df['C'] == chain) & (df['M'] == mutation)]

            if not ddg_row.empty:
                ddg = ddg_row['ddg'].values[0]

                wt_nodes, wt_edges, wt_edge_features = graph_to_gnn_format(wild_type_graph)
                mut_nodes, mut_edges, mut_edge_features = graph_to_gnn_format(mutant_graph)

                data = {
                    'wild_type': {
                        'node_features': wt_nodes,
                        'edge_index': wt_edges,
                        'edge_features': wt_edge_features
                    },
                    'mutant': {
                        'node_features': mut_nodes,
                        'edge_index': mut_edges,
                        'edge_features': mut_edge_features
                    },
                    'ddg': ddg,
                    'wild_type_name': f"{pdb_id}_{chain}_wild_type.pdb",
                    'mutant_name': filename
                }
                dataset.append(data)

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

# Example usage
# Example usage
directory = 'S669'  # Directory containing PDB files
pssm_pickle_file = 'new_pssm_s669.pkl'  # Pickle file with PSSM arrays
conservation_directory = 'cons_s669'  # Directory containing conservation scores
csv_file = 's669.csv'  # CSV file with ddg values
output_file = 'data_s669.pkl'  # Output file for the dataset

graphs = process_pdb_files(directory, pssm_pickle_file, conservation_directory)
save_graphs_as_torch_data(graphs, csv_file, output_file)

print(f"Saved dataset to {output_file}")


