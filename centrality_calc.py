import networkx as nx
import random
import heapq

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

def dijkstra(G, source, target, connector_edges_only=False):
    queue = [(0, source, [])]
    seen = set()
    while queue:
        (cost, v, path) = heapq.heappop(queue)
        if v not in seen:
            seen.add(v)
            path = path + [v]
            if v == target:
                return cost, path
            for u, edge_attr in G[v].items():
                if connector_edges_only and not edge_attr.get('connector', False):
                    continue
                if u not in seen:
                    heapq.heappush(queue, (cost + edge_attr['weight'], u, path))
    return float("inf"), []

def brandes_betweenness_centrality(G):
    betweenness = {node: 0 for node in G}
    for s in G:
        S = []
        P = {node: [] for node in G}
        sigma = {node: 0 for node in G}
        sigma[s] = 1
        D = {}
        Q = [(0, s)]
        while Q:
            (dist, v) = heapq.heappop(Q)
            if v not in D:
                D[v] = dist
                for w in G[v]:
                    if w not in D:
                        heapq.heappush(Q, (dist + 1, w))
                        sigma[w] = sigma[w] + sigma[v]
                        P[w].append(v)
        delta = {node: 0 for node in G}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
        S = list(D.keys())
        S.reverse()
    return betweenness

def calculate_betweenness_centrality(G, CommunityTable, communities, time_interval):
    bridge_nodes = {}
    for community1 in communities:
        for community2 in communities:
            if community1 != community2:
                nodes_in_community1 = [node for node in G.nodes() if node in CommunityTable and CommunityTable[node] == community1]
                nodes_in_community2 = [node for node in G.nodes() if node in CommunityTable and CommunityTable[node] == community2]
                for node1 in nodes_in_community1:
                    for node2 in nodes_in_community2:
                        if G.has_edge(node1, node2):
                            if node1 not in bridge_nodes:
                                bridge_nodes[node1] = set()
                            bridge_nodes[node1].add(community1)
                            bridge_nodes[node1].add(community2)
                            print(f"Bridge node {node1} connects communities {community1} and {community2}")
    print(f"Total number of bridge nodes: {len(bridge_nodes)}")
    for node, communities in bridge_nodes.items():
        print(f"Bridge node {node} is part of {len(communities)} communities")
    return bridge_nodes
def main():
    G = generate_synthetic_graph(100, 300)
    threshold = find_optimal_threshold(G, (18, 25))
    CommunityTable = assign_nodes_to_communities(G, threshold)
    CommunityList = {}
    for community in range(1, len(set(CommunityTable.values())) + 1):
        CommunityList[community] = [node for node, comm in CommunityTable.items() if comm == community]
    print("CommunityList:")
    for community, nodes in CommunityList.items():
        print(f"Community {community}: {nodes}")
    
    betweenness_centrality = calculate_betweenness_centrality(G, CommunityTable, CommunityList, "morning_peak_hours")
    print("Betweenness Centrality:")
    for node, centrality in betweenness_centrality.items():
        print(f"Node {node}: {centrality}")

if __name__ == "__main__":
    main()
