import networkx as nx
import random

def generate_synthetic_graph(num_nodes, num_edges):
    G = nx.Graph()

    # Add nodes with fixed weights
    for i in range(num_nodes):
        G.add_node(i, weight=round(random.uniform(0.5, 2.0), 2))

    # Add edges with base weights based on connected node weights
    for i in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u!= v and not G.has_edge(u, v):
            node_u_weight = G.nodes[u]['weight']
            node_v_weight = G.nodes[v]['weight']
            base_weight = round(node_u_weight * node_v_weight, 2)
            G.add_edge(u, v, base_weight=base_weight, length=round(random.uniform(1, 10), 2), traffic_density=round(random.uniform(0.5, 1.5), 2))

    return G

G = generate_synthetic_graph(1000, 3000)

# Initialize an empty table GravityTable to store gravity values for each pair of nodes
GravityTable = []

# Loop through each edge in the input graph
for edge in G.edges():
    node1, node2 = edge
    weight1 = G.nodes[node1]['weight']
    weight2 = G.nodes[node2]['weight']
    base_weight = G.edges[edge]['base_weight']
    traffic_density = G.edges[edge]['traffic_density']
    distance = G.edges[edge]['length']

    # Calculate the gravity value (Gravity) between node1 and node2
    community_size_penalty = 0.1
    community_size = 5
    gravity = round((weight1 * weight2 * traffic_density * base_weight) / distance - (community_size_penalty * community_size), 2)

    # Insert node1, node2, and Gravity into GravityTable
    GravityTable.append((node1, node2, gravity))

# Initialize an empty table MaxGravityTable to store the maximum gravity value for each node
MaxGravityTable = {}

# Loop through each entry in GravityTable
for entry in GravityTable:
    node1, node2, gravity = entry

    # Check if node1 is already in MaxGravityTable
    if node1 in MaxGravityTable:
        if gravity > MaxGravityTable[node1][0]:
            MaxGravityTable[node1] = (gravity, node2)
    else:
        MaxGravityTable[node1] = (gravity, node2)

    # Check if node2 is already in MaxGravityTable
    if node2 in MaxGravityTable:
        if gravity > MaxGravityTable[node2][0]:
            MaxGravityTable[node2] = (gravity, node1)
    else:
        MaxGravityTable[node2] = (gravity, node1)

def find_optimal_threshold(G, num_communities_desired_range, max_iterations=1000):
    low = 0
    high = 15
    iteration = 0
    while iteration < max_iterations:
        threshold = (low + high) / 2
        CommunityTable = {}
        community_number = 1
        for node, (gravity, pair_node) in MaxGravityTable.items():
            if node not in CommunityTable:
                if gravity > threshold:
                    CommunityTable[node] = community_number
                    community_number += 1
                else:
                    CommunityTable[node] = community_number
        num_communities = len(set(CommunityTable.values()))
        if num_communities_desired_range[0] <= num_communities <= num_communities_desired_range[1]:
            return threshold
        elif num_communities < num_communities_desired_range[0]:
            high = threshold
        else:
            low = threshold
        iteration += 1
    print("Failed to find optimal threshold after", max_iterations, "iterations.")
    return None

threshold = find_optimal_threshold(G, (18, 25))

# Initialize an empty table CommunityTable to store communities based on the maximum gravity values
CommunityTable = {}
community_number = 1

# Loop through each entry in MaxGravityTable
for node, (gravity, pair_node) in MaxGravityTable.items():
    if node not in CommunityTable:
        if gravity > threshold:
            CommunityTable[node] = community_number
            community_number += 1
        else:
            CommunityTable[node] = community_number

# Return CommunityList containing labeled sets of nodes assigned to particular communities
CommunityList = {}
for community in range(1, community_number):
    CommunityList[community] = [node for node, comm in CommunityTable.items() if comm == community]

print("CommunityList:")
for community, nodes in CommunityList.items():
    print(f"Community {community}: {nodes}")

# Check if all nodes have been assigned to a community
for node in G.nodes():
    if node not in CommunityTable:
        print(f"Node {node} has not been assigned to a community")
