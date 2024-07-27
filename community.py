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

G = generate_synthetic_graph(1000, 3500)

print("Generated graph with 1000 nodes and 3500 edges")
print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges(data=True))

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
    gravity = round((weight1 * weight2 * traffic_density * base_weight) / distance, 2)
    print(f"Calculating gravity between {node1} and {node2}: {gravity}")

    # Insert node1, node2, and Gravity into GravityTable
    GravityTable.append((node1, node2, gravity))

print("GravityTable:")
for entry in GravityTable:
    print(entry)

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

print("MaxGravityTable:")
for node, (gravity, pair_node) in MaxGravityTable.items():
    print(f"Node {node}: Max Gravity = {gravity}, Pair Node = {pair_node}")

# Set a threshold for the maximum gravity value
threshold = 10.0

# Initialize an empty table CommunityTable to store communities based on the maximum gravity values
CommunityTable = {}
community_number = 1

# Loop through each entry in MaxGravityTable
for node, (gravity, pair_node) in MaxGravityTable.items():
    if node not in CommunityTable:
        if gravity > threshold:
            CommunityTable[node] = community_number
            print(f"Assigning node {node} to community {community_number}")
            community_number += 1
        else:
            CommunityTable[node] = community_number
            print(f"Assigning node {node} to community {community_number}")

print("CommunityTable:")
for node, community in CommunityTable.items():
    print(f"Node {node}: Community = {community}")

# Return CommunityList containing labeled sets of nodes assigned to particular communities
CommunityList = {}
for community in range(1, community_number):
    CommunityList[community] = [node for node, comm in CommunityTable.items() if comm == community]

print("CommunityList:")
for community, nodes in CommunityList.items():
    print(f"Community {community}: {nodes}")
