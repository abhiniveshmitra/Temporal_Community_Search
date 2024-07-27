import networkx as nx
import random

def generate_synthetic_graph(num_nodes, num_edges):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, weight=round(random.uniform(0.5, 2.0), 2))
    for i in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            node_u_weight = G.nodes[u]['weight']
            node_v_weight = G.nodes[v]['weight']
            base_weight = round(node_u_weight * node_v_weight, 2)
            G.add_edge(u, v, base_weight=base_weight, length=round(random.uniform(1, 10), 2), traffic_density=round(random.uniform(0.5, 1.5), 2))
    return G

def calculate_gravity_table(G):
    GravityTable = []
    for edge in G.edges():
        node1, node2 = edge
        weight1 = G.nodes[node1]['weight']
        weight2 = G.nodes[node2]['weight']
        base_weight = G.edges[edge]['base_weight']
        traffic_density = G.edges[edge]['traffic_density']
        distance = G.edges[edge]['length']
        community_size_penalty = 0.1
        community_size = 5
        gravity = round((weight1 * weight2 * traffic_density * base_weight) / distance - (community_size_penalty * community_size), 2)
        GravityTable.append((node1, node2, gravity))
    return GravityTable

def calculate_max_gravity_table(GravityTable):
    MaxGravityTable = {}
    for entry in GravityTable:
        node1, node2, gravity = entry
        if node1 in MaxGravityTable:
            if gravity > MaxGravityTable[node1][0]:
                MaxGravityTable[node1] = (gravity, node2)
        else:
            MaxGravityTable[node1] = (gravity, node2)
        if node2 in MaxGravityTable:
            if gravity > MaxGravityTable[node2][0]:
                MaxGravityTable[node2] = (gravity, node1)
        else:
            MaxGravityTable[node2] = (gravity, node1)
    return MaxGravityTable

def find_optimal_threshold(G, num_communities_desired_range, max_iterations=1000):
    low = 0
    high = 15
    iteration = 0
    while iteration < max_iterations:
        threshold = (low + high) / 2
        CommunityTable = {}
        community_number = 1
        for node, (gravity, pair_node) in calculate_max_gravity_table(calculate_gravity_table(G)).items():
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

def assign_nodes_to_communities(G, threshold):
    CommunityTable = {}
    community_number = 1
    for node, (gravity, pair_node) in calculate_max_gravity_table(calculate_gravity_table(G)).items():
        if node not in CommunityTable:
            if gravity > threshold:
                CommunityTable[node] = community_number
                community_number += 1
            else:
                CommunityTable[node] = community_number
    return CommunityTable

def main():
    G = generate_synthetic_graph(1000, 3000)
    threshold = find_optimal_threshold(G, (18, 25))
    CommunityTable = assign_nodes_to_communities(G, threshold)
    CommunityList = {}
    for community in range(1, len(set(CommunityTable.values())) + 1):
        CommunityList[community] = [node for node, comm in CommunityTable.items() if comm == community]
    print("CommunityList:")
    for community, nodes in CommunityList.items():
        print(f"Community {community}: {nodes}")

if __name__ == "__main__":
    main()
