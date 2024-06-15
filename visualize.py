import random
import networkx as nx
import matplotlib.pyplot as plt

def generate_synthetic_large_graph(num_nodes, num_edges, num_timestamps):
    G = nx.Graph()

    # Add nodes with fixed weights
    for i in range(num_nodes):
        G.add_node(i, weight=round(random.uniform(0.5, 2.0), 2))  # Node weights vary between 0.5 and 2.0
        if i % 10000 == 0:
            print(f"Added {i} nodes...")

    # Add edges with base weights based on connected node weights
    for i in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u!= v and not G.has_edge(u, v):
            node_u_weight = G.nodes[u]['weight']
            node_v_weight = G.nodes[v]['weight']
            # Use the weighted geometric mean as the base weight
            base_weight = round(node_u_weight * node_v_weight, 2)
            G.add_edge(u, v, base_weight=base_weight, length=round(random.uniform(1, 10), 2))
        if G.degree(u) > 4:
            G.remove_edge(u, v)
        if G.degree(v) > 4:
            G.remove_edge(u, v)
        if i % 10000 == 0:
            print(f"Added {i} edges...")

    # Vary edge weights according to timestamp
    for t in range(num_timestamps):
        for u, v in G.edges():
            base_weight = G[u][v]['base_weight']
            weight_variation = round(random.uniform(0.8, 1.2), 2)
            G[u][v][t] = round(base_weight * weight_variation, 2)

    return G

G = generate_synthetic_large_graph(100, 500, 10)

# Visualize the graph using matplotlib
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=10, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8)
plt.axis('off')
plt.show()
