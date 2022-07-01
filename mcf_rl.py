import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pandas as pd
from fat_tree import genFatTree, fatTreeToFlowNetwork

# The goal for now is to simply find the shortest path from
# souce -> destination in a fat tree network using reinforcement 
# learning. This example will use Q learning. 

# As input we will have a networkx digraph with weight and capacity 
# attributes. For now we will not be taking capacity into account 
# because we simply want to find the first shortest path. We do have to take 
# weight into accont.

# Pseudo algorithm: 

# Step 1: Initialize Q matrix 
# - Init all possible Q values to 0. The size of the matrix will be nxn where n is the
# number of nodes in the network. When I say 'possible' I mean there is an edge connecting
# the two nodes. We can add an offset of -1000 to all other nodes since there is no connection.

# Step 2: Initialize R matrix (reward matrix) 
# - To initialize the R matrix, first we set all value to a very high negative value, since it can
# be assumed that most of the edges don't exist. We then iterate to find the edges that do exist, and
# set the value = weight. For instance, if weight(edge(u,v)) = 10, then R[u,v] = 10. This gives possible actions
# a reward proportional to their weight. Then we iterate to find all edges that connect to the desination node. 
# Since getting to the destination is the goal, this will have a very high reward value. In practice we can
# scale the max reward and min reward to be proportionate with the weights. 

# Since the nodes in our fat tree have prefixed indicies, e.g 4001, 2002, etc.,
# We are going to use a 1:1 lookup table which maps nodes to a range of indicies [0,n)
# where n is the number of nodes. This will make it much easier to work with numpy matricies
# To implement the 1:1 lookup table we will just use two dictionaries.
def nodeLookupTable(nodes):
	n = len(nodes)
	node_to_index = {}
	index_to_node = {}
	for i,node in enumerate(nodes):
		node_to_index[node] = i
		index_to_node[i] = node
	return node_to_index, index_to_node

def genNetwork(k, weight_function, capacity_function):
	# First generate a fat tree
	nodes, edges = genFatTree(k)
	# Now transform into flow network with random weight and capacity functions
	source, dest, flow_network = fatTreeToFlowNetwork(edges, capacity_function, weight_function)
	for node in flow_network.nodes:
		print(node)
		print(flow_network[node])
	return source, dest, flow_network

# Initialize Q matrix according to step 1
def initQ(G, node_lookup, index_lookup):
	n = len(G.nodes)
	Q = np.matrix(np.zeros(shape=(n,n))) 
	Q -= 1000
	for node1 in G.nodes:
		for node2 in G[node1]:
			Q[node_lookup[node1], node_lookup[node2]] = 0
			Q[node_lookup[node2], node_lookup[node1]] = 0
	print(pd.DataFrame(data=Q, index=[node for node in G.nodes], columns=[node for node in G.nodes]))
	return Q

# Initialize R matrix according to the step 2
def initR(G, dest, node_lookup, index_lookup, reward_bias):
	n = len(G.nodes)
	R = np.matrix(np.zeros(shape=(n,n)))
	# First we iterate over all edges and initialize R[u,v] = reward_bias - weight(u, v)
	# This will give nodes with lower weight higher reward 
	for node1 in G.nodes:
		for node2 in G[node1]:
			weight = G[node1][node2]['weight']
			R[node_lookup[node1], node_lookup[node2]] = reward_bias - weight
			R[node_lookup[node2], node_lookup[node1]] = reward_bias - weight
	for node in G[dest]:
		R[node_lookup[node], node_lookup[dest]] += 1000
	print(pd.DataFrame(data=R, index=[node for node in G.nodes], columns=[node for node in G.nodes]))
	return R

def chooseNextNode(G, Q, node_lookup, index_lookup, current_node, thresh):
	# This function will look at the current state(node)
	# and then choose an action(next node). Thresh is some value between
	# 0 and 1. Thresh will dictate "exploration vs. exploitation". 
	# This means there is some probability that an action will be chosen randomly, 
	# and some probability where the action will be the highest Q value of the possible
	# actions. 
	random_action_probability = random.uniform(0, 1)
	possible_actions = [] 
	if random_action_probability < thresh:
		for node in G[current_node]:
			possible_actions.append(node)
	else:
		max_value = np.max(Q[node_lookup[current_node]])
		for node in G[current_node]:
			possible_next_node = node_lookup[node]
			if Q[node_lookup[current_node], possible_next_node] == max_value:
				possible_actions.append(node)
	next_node = random.choice(possible_actions)
	return next_node

def updateQ(G, current_node, next_node, Q, R, node_lookup, index_lookup, lr, discount):
	# Here we have a current node and next node
	# We are going to update the Q matrix according to to the standard Q update method
	# We combine the current Q value and the expected future reward in proportion to the lr
	# First we need to calculate the quality of being in the new state
	max_value = np.max(Q[node_lookup[next_node]])
	Q[node_lookup[current_node], node_lookup[next_node]] = \
	(1 - lr) * Q[node_lookup[current_node], node_lookup[next_node]] + \
	lr * (R[node_lookup[current_node], node_lookup[next_node]] + \
	discount * max_value)	

def learn(G, Q, R, node_lookup, index_lookup, thresh, lr, discount):
	for i in range(50000):
		source = random.choice(list(G.nodes))
		next_node = chooseNextNode(G, Q, node_lookup, index_lookup, source, thresh)
		updateQ(G, source, next_node, Q, R, node_lookup, index_lookup, lr, discount)
	print(pd.DataFrame(data=Q, index=[node for node in G.nodes], columns=[node for node in G.nodes]))

def shortestPath(G, source, dest, node_lookup, index_lookup, Q,):
	path = [source]
	next_node = index_lookup[np.argmax(Q[node_lookup[source],])]
	path.append(next_node)
	while next_node != dest:
		next_node = index_lookup[np.argmax(Q[node_lookup[next_node],])]
		path.append(next_node)
	print(path)
	return path

def main():
	k = 4
	capacity_function = lambda _: random.randint(1,10)
	weight_function = lambda _: random.randint(1,10)
	source, dest, G = genNetwork(k, weight_function, capacity_function)
	thresh = 0.2
	lr = 0.8
	discount = 0.8
	print(f"Source: {source}")
	print(f"Dest: {dest}")
	node_lookup, index_lookup = nodeLookupTable(G.nodes)
	print(f"Node lookup table: {node_lookup}")
	Q = initQ(G, node_lookup, index_lookup)
	R = initR(G, dest, node_lookup, index_lookup, 100)
	learn(G, Q, R, node_lookup, index_lookup, thresh, lr, discount)
	shortestPath(G, source, dest, node_lookup, index_lookup, Q)

if __name__ == "__main__":
	main()


