import networkx as nx
import random
import heapq

# Generate synthetic graph with weighted edges
def generate_synthetic_graph(num_nodes, num_edges):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, weight=round(random.uniform(0.5, 2.0), 2))
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and not G.has_edge(u, v):
            node_u_weight = G.nodes[u]['weight']
            node_v_weight = G.nodes[v]['weight']
            base_weight = round(node_u_weight * node_v_weight, 2)
            G.add_edge(u, v, base_weight=base_weight, length=round(random.uniform(1, 10), 2), traffic_density=round(random.uniform(0.5, 1.5), 2))
    return G

# Calculate gravity table for edges
def calculate_gravity_table(G):
    GravityTable = []
    for edge in G.edges():
        node1, node2 = edge
        weight1 = G.nodes[node1]['weight']
        weight2 = G.nodes[node2]['weight']
        base_weight = G.edges[edge].get('base_weight', 1)
        traffic_density = G.edges[edge].get('traffic_density', 1)
        distance = G.edges[edge].get('length', 1)
        community_size_penalty = 0.1
        community_size = 5
        gravity = round((weight1 * weight2 * traffic_density * base_weight) / distance - (community_size_penalty * community_size), 2)
        GravityTable.append((node1, node2, gravity))
    return GravityTable

# Calculate max gravity table
def calculate_max_gravity_table(GravityTable):
    MaxGravityTable = {}
    for entry in GravityTable:
        node1, node2, gravity = entry
        if node1 not in MaxGravityTable:
            MaxGravityTable[node1] = (gravity, node2)
        else:
            if gravity > MaxGravityTable[node1][0]:
                MaxGravityTable[node1] = (gravity, node2)
        if node2 not in MaxGravityTable:
            MaxGravityTable[node2] = (gravity, node1)
        else:
            if gravity > MaxGravityTable[node2][0]:
                MaxGravityTable[node2] = (gravity, node1)
    return MaxGravityTable

# Find optimal threshold for community assignment
def find_optimal_threshold(G, num_communities_desired_range, max_iterations=1000):
    low = 0
    high = 15
    iteration = 0
    while iteration < max_iterations:
        threshold = (low + high) / 2
        CommunityTable = assign_nodes_to_communities(G, threshold)
        if not CommunityTable:
            print("CommunityTable is empty. Adjusting threshold.")
            low = threshold
            iteration += 1
            continue
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

# Assign nodes to communities based on gravity threshold
def assign_nodes_to_communities(G, threshold):
    MaxGravityTable = calculate_max_gravity_table(calculate_gravity_table(G))
    CommunityTable = {}
    community_number = 1
    for node, (gravity, pair_node) in MaxGravityTable.items():
        if gravity is not None and gravity > threshold:
            if node not in CommunityTable:
                CommunityTable[node] = community_number
            if pair_node not in CommunityTable:
                CommunityTable[pair_node] = community_number
            community_number += 1
        elif node not in CommunityTable:
            CommunityTable[node] = community_number
    return CommunityTable

# Find community bridges
def find_community_bridges(G, communities):
    bridge_nodes = set()
    community_pairs = [(c1, c2) for c1 in set(communities.values()) for c2 in set(communities.values()) if c1 < c2]
    for community1, community2 in community_pairs:
        for node in G.nodes:
            if communities[node] == community1 or communities[node] == community2:
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if communities[neighbor] != communities[node]:
                        bridge_nodes.add(node)
                        break
    return bridge_nodes

# A* algorithm for shortest path considering weights
def a_star(G, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic[start], start))
    came_from = {}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in G.nodes}
    f_score[start] = heuristic[start]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor, default={})
            weight = edge_data.get('base_weight', 1)  # Default to 1 if 'base_weight' is missing
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

# Find shortest path with community jumps
def find_shortest_path_with_community_jumps(G, source, target, bridge_nodes, timestamp, communities):
    def heuristic(node):
        return 0

    def get_edge_weight(u, v):
        return G[u][v].get('base_weight', 1)

    def dijkstra_with_community_jumps(start, end, visited):
        queue = [(0, start, [])]
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]
            if node == end:
                return cost, path
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    edge_weight = get_edge_weight(node, neighbor)
                    heapq.heappush(queue, (cost + edge_weight, neighbor, path))
        return float('inf'), []

    # Compute shortest path within each community
    intra_community_paths = {}
    for community in set(communities.values()):
        nodes_in_community = [node for node in G.nodes if communities[node] == community]
        for node in nodes_in_community:
            for target_node in nodes_in_community:
                if node != target_node:
                    cost, path = dijkstra_with_community_jumps(node, target_node, set())
                    intra_community_paths[(node, target_node)] = (cost, path)

    # Compute shortest paths using bridge nodes
    best_cost = float('inf')
    best_path = []

    for bridge_node in bridge_nodes:
        if communities[source] != communities[bridge_node] and communities[target] != communities[bridge_node]:
            cost1, path1 = dijkstra_with_community_jumps(source, bridge_node, set())
            cost2, path2 = dijkstra_with_community_jumps(bridge_node, target, set())
            total_cost = cost1 + cost2
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path1 + path2[1:]

    return best_cost, best_path

# Main function to integrate all calculations
def main():
    num_nodes = 100
    num_edges = 500
    G = generate_synthetic_graph(num_nodes, num_edges)

    # Find optimal threshold for community assignment
    num_communities_desired_range = (20, 25)
    threshold = find_optimal_threshold(G, num_communities_desired_range)

    if threshold is None:
        print("Failed to find a suitable threshold.")
        return

    # Assign nodes to communities
    communities = assign_nodes_to_communities(G, threshold)
  #  print("Communities:", communities)

    # Find community bridges
    bridge_nodes = find_community_bridges(G, communities)
  #  print("Bridge Nodes:", bridge_nodes)

    # Randomly select source, target, and timestamp
    source = random.choice(list(G.nodes))
    target = random.choice(list(G.nodes))
    while source == target:  # Ensure source and target are different
        target = random.choice(list(G.nodes))
    timestamp = random.randint(0, 23)  # Adjust range as needed

    # Find shortest path with community jumps
    cost, path = find_shortest_path_with_community_jumps(G, source, target, bridge_nodes, timestamp, communities)
    print(f"Source: {source}, Target: {target}, Timestamp: {timestamp}")
    print(f"Shortest path cost from {source} to {target}: {cost:.2f}")
    print(f"Shortest path: {path}")

if __name__ == "__main__":
    main()
