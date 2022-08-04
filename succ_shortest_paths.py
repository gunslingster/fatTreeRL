import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# This program aims to model backup server assignment in a software defined
# network as a minimum cost flow problem. 

# In any given network you may have:
# m = number of VNF
# n = number of backup servers
# r = resource capacity of each backup server (how many VNF it can backup)
# Failure probabilities for each VNF and backup server

# We want to model this problem as a minimum cost flow problem to find a the
# optimal mapping between vnf -> backup server

# We have an arbitrary source(s) and destination(t) nodes and we want to push
# f amount of flow from s->t. Each edge in the graph has a capacity and cost. 
# The node s is connected to every VNF. The vnf nodes and backup server nodes
# compose a complete bipartite graph. Each VNF has a connection to every backup
# server. Keep in mind that this is not necissarily a physical connection, but 
# rather how we choose to model the graph to achieve the optimal backup server
# assignment. 

# We will be using reinforcement learning, specifically Q learning, in order 
# to implement the shorest successive paths algorithm to solve the minimum cost 
# flow problem. 

# Steps

# Step 1: Generate a networkx graph with the parameters stated above. We will 
# have m VNF, n backup servers, and source and sink nodes. Total number of nodes
# will be m + n + 2. Total number of edges will be m + n*n + n. 

# Step 2: Implement s shorest path algorithm with Q learning. This function will
# take a residual graph as input and output the shortest path from s->t. 

# Step 3: Apply the shortest successive paths algorithm using the Q learning 
# approach at each iteraiton. At each iteration we produce a new residual graph.
# The algorithm will terminate when there are no possible paths left. The final
# residual graph will yield the optimal vnf-> backup server mapping for optimal
# network availibility. 

def printGraph(G):
    for node in G.nodes:
        print(f'node: {node}')
        print(f'Neighbors: {G[node]}')
    print('\n')

def plotGraph(G):
    capacities = nx.get_edge_attributes(G, 'capacity').values()
    colors = nx.get_edge_attributes(G, 'weight').values()
    plt.figure(figsize=(12,12))
    pos = nx.multipartite_layout(G, subset_key='layer')
    nx.draw(G, pos=pos, with_labels=True, edge_color=colors, width=list(capacities))
    plt.show()

def genNetwork(m, n, r, vnf_fail_prob, backup_server_fail_prob):
    '''
    Generate a networkx digraph from given parameters. 

    m = number VNF
    n = number backup server
    r = resource capacity of each backup server
    
    vnf_fail_prob = function that gives failure probability of each VNF
    backup_server_fail_prob = function that gives failure probability of each VNF

    Returns: a networkx graph with weight and capcity attributes
    '''
    
    G = nx.DiGraph()
    def genNodes(G):
        num_nodes = m + n + 2
        G.add_node(0, layer='source')
        for i in range(1, m+1):
            G.add_node(i, layer='vnf')
        for i in range(m+1, m+n+1):
            G.add_node(i, layer='backup_server')
        G.add_node(m+n+1, layer='sink')
    # Set source node to node 0 and sink to num_nodes-1
    # Nodes [1, m] will be VNF
    # Nodes [m+1, m+n] will be backup servers

    # First lets assign a failure probability for each VNF and backup server
    def genFailureProbs(m, n):
        failure_probs = {}
        for vnf in range(1, m+1):
            failure_probs[vnf] = vnf_fail_prob(1)
        for backup_server in range(m+1, m+n+1):
            failure_probs[backup_server] = backup_server_fail_prob(1)
        return failure_probs

    def genEdges(G):
        # First add an edge from s to each vnf
        failure_probs = genFailureProbs(m, n)
        for node in range(1, m+1):
            G.add_edge(0, node, weight=0, capacity=1, flow=0)
        # Now add edges from each VNF to backup server
        for vnf in range (1, m+1):
            p1 = failure_probs[vnf]
            for backup_server in range(m+1, m+n+1):
                p2 = failure_probs[backup_server]
                w = int(math.log(1 / (1 - p1 * p2)) * 10e10)
                G.add_edge(vnf, backup_server, weight=w, capacity=1, flow=0)
        # Now add edges from each backup server to the destination
        for backup_server in range(m+1, m+n+1):
            G.add_edge(backup_server, m+n+1, weight=0, capacity=r, flow=0)
    genNodes(G)
    genEdges(G)
    print('Original Graph:')
    printGraph(G)
    plotGraph(G)
    return G

def shortestPath(G, m, n):
    def initQ(G):
        num_nodes = len(G.nodes)
        Q = np.matrix(np.zeros(shape=(num_nodes,num_nodes)))
        Q -= 1000
        for node1 in G.nodes:
            for node2 in G[node1]:
                Q[node1, node2] = 0
        Q = np.int_(Q)
        return Q
    
    def initR(G, m, n):
        num_nodes = len(G.nodes)
        R = np.matrix(np.zeros(shape=(num_nodes,num_nodes)))
        for node1 in G.nodes:
            for node2 in G[node1]:
                weight = G[node1][node2]['weight']
                R[node1, node2] = 1000000000 - weight
        for node in range(m+1, m+n+1):
            R[node, m+n+1] = 100000000000000
        R = np.int_(R)
        #print("Reward matrix")
        #print(pd.DataFrame(data=R))
        return R

    def chooseNextNode(G, Q, current_node, thresh):
        random_action_prob = random.uniform(0, 1)
        possible_actions = []
        if random_action_prob < thresh:
            for node in G[current_node]:
                possible_actions.append(node)
        else:
            max_value = np.max(Q[current_node])
            for possible_next_node in G[current_node]:
                if Q[current_node, possible_next_node] == max_value:
                    possible_actions.append(possible_next_node)
        next_node = random.choice(possible_actions)
        return next_node
    
    def updateQ(G, current_node, next_node, Q, R, lr, discount):
        max_value = np.max(Q[next_node])
        Q[current_node, next_node] = (1 - lr) * Q[current_node, next_node] + \
                lr * (R[current_node, next_node] + discount * max_value)

    def learn(G, Q, R, thresh, lr, discount):
        nodes = [i for i in range(len(G.nodes) - 1)]
        for i in range(10000):
            source = random.choice(nodes) 
            next_node = chooseNextNode(G, Q, source, thresh)
            updateQ(G, source, next_node, Q, R, lr, discount)
        #print("Q matrix after learning:")
        #print(pd.DataFrame(data=Q))

    def shortest_path(G, source, dest, Q):
        path = [source]
        next_node = np.argmax(Q[source,])
        path.append(next_node)
        count = 0
        while next_node != dest:
            next_node = np.argmax(Q[next_node,])
            path.append(next_node)
            count += 1
        return path

    Q = initQ(G)
    R = initR(G, m, n)
    learn(G, Q, R, 0.2, 0.8, 0.8)
    return shortest_path(G, 0, len(G.nodes)-1, Q)

def residualNetwork(G, path):
    # Residual network is the same as the original graph but the 
    # forward edges are relpaced with backward edges 
    R = G.copy()
    for i in range(len(path)-1):
        node1 = path[i]
        node2 = path[i+1]
        weight = G[node1][node2]['weight']
        R.remove_edge(node1, node2)
        R.add_edge(node2, node1, weight=-weight, capacity=1)
    return R

def residualNetwork2(G, path):
    R = G.copy()
    for i in range(len(path)-1):
        node1 = path[i]
        node2 = path[i+1]
        R[node1][node2]['capacity'] -= 1
        weight = G[node1][node2]['weight']
        if R[node1][node2]['capacity'] <= 0:
            R.remove_edge(node1, node2)
            R.add_edge(node2, node1, weight=-weight, capacity=1)
        else:
            R.add_edge(node2, node1, weight=-weight, capacity=1)
    return R

def succShortestPaths(G, m, n):
    # Find shortest path then generate residual network, do this m times
    # Positive and negative flows will cancel out, resulting in all positive
    # flow from vnf->backup server. The final flows will give use the optimal 
    # mapping of vnf-> backup servers
    # We can use the final shortest path to find the optimal mapping
    R = G.copy()
    shortest_paths = []
    for num_iters in range(m):
        # Can use dijkstra or Q learning approach
        shortest_path = nx.dijkstra_path(R, 0, m+n+1)
        shortest_paths.append(shortest_path)
        print(f'\n{shortest_path}\n')
        R = residualNetwork2(R, shortest_path)
        printGraph(R)
    return shortest_paths

def findOptimalMapping(path):
    optimal_mapping = []
    for i in range(1, len(path)-1, 2):
        optimal_mapping.append((path[i], path[i+1]))
    print(optimal_mapping)
    return optimal_mapping

def findOptimalMapping2(paths):
    optimal_mapping = []
    for path in paths:
        for i in range(1, len(path)-2):
            if ((path[i+1], path[i]) in optimal_mapping):
                optimal_mapping.remove((path[i+1], path[i]))
            else:
                optimal_mapping.append((path[i], path[i+1]))
    print(optimal_mapping)
    return optimal_mapping

def test():
    m = 4
    n = 2 
    r = 2
    f1 = lambda _: random.uniform(0, 0.02)
    f2 = lambda _: random.uniform(0, 0.1)
    G = genNetwork(m, n, r, f1, f2)
    final_paths = succShortestPaths(G, m, n)
    findOptimalMapping2(final_paths) 

test()
