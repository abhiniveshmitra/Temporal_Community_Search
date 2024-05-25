import networkx as nx
import random

def generate_synthetic_large_graph(num_nodes, num_edges, num_timestamps):
    G = nx.Graph()

    # Add nodes with fixed weights
    for i in range(num_nodes):
        G.add_node(i, weight=random.uniform(0.5, 2.0))  # Fixed weight between 0.5 and 2.0

    # Add edges with fixed lengths and time-varying traffic densities
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            length = random.uniform(1, 10)  # Fixed length between 1 and 10
            traffic_densities = [random.uniform(0.1, 5.0) for _ in range(num_timestamps)]  # Varying densities
            G.add_edge(u, v, length=length, traffic_densities=traffic_densities)

    return G

# Parameters for the large-scale graph
large_num_nodes = 10**6
large_num_edges = 10**6
num_timestamps = 24  # Traffic densities for 24 hours

# Generate the large-scale graph
large_G = generate_synthetic_large_graph(large_num_nodes, large_num_edges, num_timestamps)

# Save the large-scale graph to a file
nx.write_gpickle(large_G, "large_synthetic_graph.gpickle")
