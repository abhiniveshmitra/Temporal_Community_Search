import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import scipy
#import community

def generate_synthetic_large_graph(num_nodes, num_edges, num_timestamps):
    G = nx.Graph()

    # Add nodes with weights drawn from a power-law distribution
    node_weights = np.random.power(2, size=num_nodes)
    for i in range(num_nodes):
        G.add_node(i, weight=node_weights[i])

    # Add edges with weights based on connected node weights
    for i in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u!= v and not G.has_edge(u, v):
            node_u_weight = G.nodes[u]['weight']
            node_v_weight = G.nodes[v]['weight']
            # Use the harmonic mean of the node weights as the base weight
            base_weight = 2 / (1/node_u_weight + 1/node_v_weight)
            G.add_edge(u, v, base_weight=base_weight, length=round(random.uniform(1, 10), 2))

    # Vary edge weights according to timestamp
    for t in range(num_timestamps):
        for u, v in G.edges():
            base_weight = G[u][v]['base_weight']
            # Use a sinusoidal function to capture time-of-day patterns
            weight_variation = 1 + 0.2 * np.sin(2 * np.pi * t / num_timestamps)
            G[u][v][t] = round(base_weight * weight_variation, 2)

    
   

    return G

G = generate_synthetic_large_graph(25, 50, 24)

def community_detection_by_edge_weight(G, weight_attribute='base_weight'):
    # Perform community detection using the Louvain algorithm with edge weights
    partition = nx.algorithms.community.louvain_communities(G, weight=weight_attribute)
    community_assignments = [(node, community_id) for community_id, nodes in enumerate(partition) for node in nodes]
    community_df = pd.DataFrame(community_assignments, columns=['node', 'community'])
    return community_df

def visualize_communities(G, communities):
    colors = [random.randint(0, 255) for _ in set(communities['community'])]
    node_colors = [colors[community] for community in communities['community']]
    nx.draw(G, node_color=node_colors, cmap=plt.cm.rainbow, with_labels=True)
    plt.show(block=True)

# Perform community detection by edge weight
community_df = community_detection_by_edge_weight(G, weight_attribute='base_weight')

# Print the community assignments for the first 100 nodes
print(community_df.loc[community_df['node'] < 100])

# Visualize the communities
visualize_communities(G, community_df)
