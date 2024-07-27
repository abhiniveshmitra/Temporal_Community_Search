import numpy as np
import networkx as nx
import random

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

    # Print node weights and edge weights for a few nodes for 10 timestamps each
    nodes_to_print = [0, 10, 20]
    for node in nodes_to_print:
        print(f"Node {node} weight: {G.nodes[node]['weight']}")
        for t in range(10):
            for neighbor in G.neighbors(node):
                print(f"  Timestamp {t}: Edge {node} - {neighbor} weight: {G[node][neighbor][t]}")
            print()

    return G

G = generate_synthetic_large_graph(1000, 5000, 24)
