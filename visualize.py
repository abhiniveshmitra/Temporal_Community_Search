def visualize_graph(G, num_nodes_to_show=30):
    subgraph_nodes = list(G.nodes)[:num_nodes_to_show]
    subgraph = G.subgraph(subgraph_nodes)
    pos = nx.spring_layout(subgraph, seed=42)  # Layout for visualization

    # Draw nodes with sizes based on weights
    node_weights = nx.get_node_attributes(subgraph, 'weight')
    nx.draw_networkx_nodes(subgraph, pos, node_size=[v * 50 for v in node_weights.values()], node_color='blue')

    # Draw edges with widths based on lengths
    edge_lengths = nx.get_edge_attributes(subgraph, 'length')
    nx.draw_networkx_edges(subgraph, pos, width=1.0, edge_color='gray')

    # Add labels to nodes with weights
    nx.draw_networkx_labels(subgraph, pos, labels={node: f"wt: {weight:.2f}" for node, weight in node_weights.items()}, font_size=6, font_color='black')

    # Add edge labels for lengths and weights
    edge_labels = {(u, v): f"L: {d['length']:.2f}, W: {d['traffic_densities'][0]:.2f}" for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=5)

    plt.figure(figsize=(30, 30), dpi=300)  # Set the figure size and high resolution
    plt.title(f"Visualization of a Subgraph with {num_nodes_to_show} Nodes")
    plt.show()

# Parameters for the smaller graph for visualization
small_num_nodes = 10
small_num_edges = 20
num_timestamps = 24  # Traffic densities for 24 hours

# Generate the smaller graph for visualization
small_G = generate_synthetic_graph(small_num_nodes, small_num_edges, num_timestamps)

# Visualize the smaller graph
visualize_graph(small_G, num_nodes_to_show=10)
