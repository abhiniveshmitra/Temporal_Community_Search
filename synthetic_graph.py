import networkx as nx
import random
import scipy.sparse as sp
import time

def generate_synthetic_large_graph(num_nodes, num_edges, num_timestamps):
    G = nx.Graph()

    # Add nodes with fixed weights
    for i in range(num_nodes):
        G.add_node(i, weight=round(random.uniform(0.5, 2.0), 2))
        if i % 10000 == 0:
            print(f"Added {i} nodes...")

    # Add edges with fixed lengths and time-varying traffic densities
    for i in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u!= v and not G.has_edge(u, v):
            length = round(random.uniform(1, 10), 2)  # Fixed length between 1 and 10
            traffic_densities = {f"t{j}": round(random.uniform(0.1, 5.0), 2) for j in range(num_timestamps)}  # Varying densities for each timestamp
            G.add_edge(u, v, length=length, traffic_densities=traffic_densities)
            if G.degree(u) > 4:
                G.remove_edge(u, v)
            if G.degree(v) > 4:
                G.remove_edge(u, v)
        if i % 10000 == 0:
            print(f"Added {i} edges...")

    return G

# Parameters for the large-scale graph
large_num_nodes = 10**5
large_num_edges = int(2.5 * large_num_nodes)  # Convert to integer
num_timestamps = 24  # Traffic densities for 24 hours

# Generate the large-scale graph
start_time = time.time()
large_G = generate_synthetic_large_graph(large_num_nodes, large_num_edges, num_timestamps)
print(f"Graph generation took {time.time() - start_time:.2f} seconds")

# Print the edges along with their lengths and traffic densities
start_time = time.time()
for i, (u, v) in enumerate(large_G.edges()):
    length = large_G[u][v]['length']
    traffic_densities = large_G[u][v]['traffic_densities']
    print(f"({u}, {v}, {length}, [")
    for timestamp, density in traffic_densities.items():
        print(f"  {timestamp}: {density},")
    print("])")
    if i % 10000 == 0:
        print(f"Printed {i} edges...")
print(f"Printing edges took {time.time() - start_time:.2f} seconds")
