# Implementation of a fat tree topology

import math
import random


class fatTree:
    """
    A class to represent a fat tree network.

    ...

    Attributes
    ----------
    CoreSwitchList : List[str]
        List of IDs for core switches
    AggSwitchList : List[str]
        List of IDs for aggregation switches
    EdgeSwitchList : List[str]
        List of IDs for edge switches
    PodList : List[List[Str]]
        List of lists where each list contains IDs of switches for a pod
    HostList : List[str]
        List of IDs for hosts
    LinkList : List[tup]
        List of tuples where each tuple is a link in the network
    k : int
        Number of ports for each switch
    podSize : int
        The width of each pod
    iCoreLayerSwitch : int
        Number of core switches in the network
    iAggLayerSwitch : int
        Number of aggregation switches in the network
    iEdgeLayerSwitch : int
        Number of edge switches in the network
    iHost : int
        Number of hosts in the network
    iPods : int
        Number of pods in the network
    hostsPerPod : int
        Number of hosts connected to each pod

    Methods
    -------


    """

    def __init__(self, k):
        # Lists for all switches in the network
        self.CoreSwitchList = []
        self.AggSwitchList = []
        self.EdgeSwitchList = []
        # Each pod will be a list of agg switches and edge switches
        # k/2 agg switchs and k/2 edge switches
        self.PodList = []
        self.HostList = []
        # a list that contains tuples of all the lists
        self.linkList = []
        # k is the number of ports for each switch in the network
        self.k = k
        # Each pod will containg k/2 aggregation switches and k/2 edge switches
        self.podSize = int(k/2)
        self.iCoreLayerSwitch = int(pow(k/2,2))
        self.iAggLayerSwitch = int(k/2) * k
        self.iEdgeLayerSwitch = self.iAggLayerSwitch
        self.iHost = self.iEdgeLayerSwitch * self.podSize
        self.iPods = k
        self.hostsPerPod = self.iCoreLayerSwitch
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        self.createHost(self.iHost)
        self.createPod(self.iPods)
        self.allNodes = self.CoreSwitchList + self.AggSwitchList + self.EdgeSwitchList + self.HostList
        print('Core Switches:')
        print(self.CoreSwitchList)
        print('Aggregation Switches: ')
        print(self.AggSwitchList)
        print('Edge Switches: ')
        print(self.EdgeSwitchList)
        print('Pods:')
        print(self.PodList)
        print('Hosts:')
        print(self.HostList)
        self.createLink()

    def createCoreLayerSwitch(self, NUMBER):
        for x in range(1, NUMBER+1):
            PREFIX = "100"
            if x >= int(10):
                PREFIX = "10"
            self.CoreSwitchList.append(PREFIX + str(x))

    def createAggLayerSwitch(self, NUMBER):
        for x in range(1, NUMBER+1):
            PREFIX = "200"
            if x >= int(10):
                PREFIX = "20"
            self.AggSwitchList.append(PREFIX + str(x))

    def createEdgeLayerSwitch(self, NUMBER):
        for x in range(1, NUMBER+1):
            PREFIX = "300"
            if x >= int(10):
                PREFIX = "30"
            self.EdgeSwitchList.append(PREFIX + str(x))

    def createPod(self, NUMBER):
        podSize = self.podSize
        def genPod(index):
            pod = []
            for i in range(index, index+podSize):
                pod.append(self.AggSwitchList[i])
                pod.append(self.EdgeSwitchList[i])
            return pod

        counter = 0
        for x in range(1, NUMBER+1):
            pod = genPod(counter)
            self.PodList.append(pod)
            counter += podSize

    def createHost(self, NUMBER):
        for x in range(1, NUMBER+1):
            PREFIX = "400"
            if x >= int(10):
                PREFIX = "40"
            self.HostList.append(PREFIX + str(x))

    def addLink(self, node1, node2):
        self.linkList.append((node1, node2))

    def createLink(self):
        # Each core switch must be connected to each pod
        # Each agg switch is connected to k/2 core switches
        aggCount = -1
        podSize = int(self.k/2)
        for i in range(self.iCoreLayerSwitch):
            if i % podSize == 0:
                aggCount += 1
            for j in range(aggCount, self.iAggLayerSwitch, podSize):
                self.addLink(self.CoreSwitchList[i], self.AggSwitchList[j])

        # Connect agg switches in each pod to edge switches in the pod
        for pod in self.PodList:
            aggSwitches = [switch for switch in pod if switch[0] == '2']
            edgeSwitches = [switch for switch in pod if switch[0] == '3']
            for aggswitch in aggSwitches:
                for edgeswitch in edgeSwitches:
                    self.addLink(aggswitch, edgeswitch)

        # Connect each edge switch to k/2 servers
        pmcount = 0
        for i in range(self.iEdgeLayerSwitch):
            for j in range(pmcount, pmcount + podSize):
                self.addLink(self.EdgeSwitchList[i], self.HostList[j])
            pmcount += podSize

    def calc_dist(self, node1, node2):
        # Can use the IDs and pods to calculate distance on a case by case basis

        # Both nodes are the same
        if node1 == node2:
            return 0

        # Case 1: host-host
        if node1[0] == '4' and node2[0] == '4':
            # Both hosts are connected to the same edge switch
            if math.ceil(int(node1[2:]) / self.podSize) == math.ceil(int(node2[2:]) / self.podSize):
                return 2
            # Both hosts are connected ot the same pod
            if math.ceil(int(node1[2:]) / self.hostsPerPod) == math.ceil(int(node2[2:]) / self.hostsPerPod):
                return 4
            # Both hosts are in different pods
            return 6

        # Case 2: host-edge
        if node1[0] == '4' and node2[0] == '3' or node1[0] == '3' and node2[0] == '4':
            check1 = node1[2:] if node1[0] == '4' else node2[2:]
            check2 = node1[2:] if node1[0] == '3' else node2[2:]

            # Host connected to edge
            if math.ceil(int(check1) / self.podSize) == int(check2):
                return 1
            # Host connected to edge within same pod
            if math.ceil(int(check1) / self.hostsPerPod) == math.ceil(int(check2) / self.podSize):
                return 3
            return 5

        # Case 3: host-agg
        # Distance is 2 if host connected to that pod, or 4 if not
        if node1[0] == '4' and node2[0] == '2' or node1[0] == '2' and node2[0] == '4':
            check1 = node1[2:] if node1[0] == '4' else node2[2:]
            check2 = node1[2:] if node1[0] == '2' else node2[2:]

            if math.ceil(int(check1) / self.hostsPerPod) == math.ceil(int(check2) / self.podSize):
                return 2
            return 4

        # Case 4: host-core
        # Distance is always 3
        if node1[0] == '4' and node2[0] == '1' or node1[0] == '1' and node2[0] == '4':
            return 3

        # Case 5: edge-edge
        # Distance is 2 if in same pod, 4 if not
        if node1[0] == '3' and node2[0] == '3':
            if math.ceil(int(node1[2:]) / self.podSize) == math.ceil(int(node2[2:]) / self.podSize):
                return 2
            else:
                return 4

        # Case 6: edge-agg
        # Distance is 1 if in the same pod, 3 if not
        if node1[0] == '3' and node2[0] == '2' or node1[0] == '2' and node2[0] == '3':
            if math.ceil(int(node1[2:]) / self.podSize) == math.ceil(int(node2[2:]) / self.podSize):
                return 1
            else:
                return 3

        # Case 7: edge-core
        # Distance always 2
        if node1[0] == '3' and node2[0] == '1' or node1[0] == '1' and node2[0] == '3':
            return 2

        # Case 8: agg-agg
        # Distance is 2 if they are in the same pod, or connected to the same core switch, otherwise distance is 4
        if node1[0] == '2' and node2[0] == '2':
            if int(node1[2:]) % self.podSize == int(node2[2:]) % self.podSize or math.ceil(int(node1[2:]) / self.podSize) == math.ceil(int(node2[2:]) / self.podSize):
                return 2
            else:
                return 4

        # Case 9: agg-core
        # Distance is 1 if they are directely connected, 3 if not
        if node1[0] == '2' and node2[0] == '1':
            if math.ceil(int(node2[2:]) / self.podSize) == self.podSize:
                if int(node1[2:]) % (self.podSize) == 0:
                    return 1
            if int(node1[2:]) % (self.podSize) == math.ceil(int(node2[2:]) / self.podSize):
                return 1
            else:
                return 3
        if node1[0] == '1' and node2[0] == '2':
            if math.ceil(int(node1[2:]) / self.podSize) == self.podSize:
                if int(node2[2:]) % (self.podSize) == 0:
                    return 1
            if int(node2[2:]) % (self.podSize) == math.ceil(int(node1[2:]) / self.podSize):
                return 1
            else:
                return 3

        # Case 10: core-core
        # Distance is 2 if they are connected to the same agg switches, 4 if not
        if node1[0] == '1'and node2[0] == '1':
            if math.ceil(int(node1[2:]) / self.podSize) == math.ceil(int(node2[2:]) / self.podSize):
                return 2
            else:
                return 4

class FlowNetwork():
    """
    Flow network

    k : int
        Number of ports
    m : int
        Number of VNF
    f : frequency of the link

    """
    def __init__(self, k, m, f):
        self.k = k
        self.m = m
        self.f = f
        self.network = fatTree(self.k)
        self.source = random.choice(self.network.allNodes)
        self.sink = random.choice(self.network.allNodes)
        self.supply = self.m
        self.demand = -self.m
        self.resource_capacity = f
        self.vnf_list, self.available_backup_servers = self.generate_VNF_and_available_backup_servers()
        self.failure_probabilities = self.assign_failure_probabilities()
        self.V_prime, self.E_prime, self.capacities_and_costs, self.node_mapping = self.dataCenter_to_flowNetwork()

    def generate_VNF_and_available_backup_servers(self):
        available_switches = self.network.CoreSwitchList + self.network.AggSwitchList + self.network.EdgeSwitchList
        if self.source in available_switches:
            available_switches.remove(self.source)
        if self.sink in available_switches:
            available_switches.remove(self.sink)
        vnf_list = []
        for i in range(self.m):
            vnf = random.choice(available_switches)
            vnf_list.append(vnf)
            available_switches.remove(vnf)
        return vnf_list, available_switches

    def assign_failure_probabilities(self):
        failure_probabilities = {}
        for vnf in self.vnf_list:
            failure_probabilities[vnf] = random.uniform(0.025, 0.175)
        for switch in self.available_backup_servers:
            failure_probabilities[switch] = random.uniform(0.01, 1)
        return failure_probabilities

    def dataCenter_to_flowNetwork(self):
        V_prime = [self.source] + [self.sink] + self.vnf_list + self.available_backup_servers
        num_nodes = len(V_prime)
        node_mapping = {}
        for i in range(num_nodes):
            node_mapping[V_prime[i]] = i
        E_prime = []
        capacities_and_costs = {}
        for vnf in self.vnf_list:
            E_prime.append((self.source, vnf))
            capacities_and_costs[(self.source, vnf)] = (1, 0)
        for vnf in self.vnf_list:
            for switch in self.available_backup_servers:
                E_prime.append((vnf, switch))
                capacity = 1
                cost = 1e16 * math.log(1 / (1 - self.failure_probabilities[vnf] * self.failure_probabilities[switch]), 10)
                capacities_and_costs[(vnf, switch)] = (capacity, cost)
        for switch in self.available_backup_servers:
            E_prime.append((switch, self.sink))
            capacities_and_costs[(switch, self.sink)] = (self.resource_capacity, 0)
        return V_prime, E_prime, capacities_and_costs, node_mapping

    def greedy1(self):
        res = 1
        vnf_probabilities = [(vnf, vnf_prob) for (vnf, vnf_prob) in self.failure_probabilities.items() if vnf in self.vnf_list]
        switch_probabilities = [(switch, switch_prob) for (switch, switch_prob) in self.failure_probabilities.items() if switch in self.available_backup_servers]
        switch_probabilities.sort(key = lambda x: x[1])
        num_switches = len(vnf_probabilities) // self.resource_capacity + 1
        vnf_probabilities.sort(key = lambda x: x[1])
        greedy1_switches = switch_probabilities[:num_switches]
        greedy1_switches.sort(reverse=True, key = lambda x: x[1])
        offset = 0
        for i in range(len(greedy1_switches)):
            for j in range(offset, min(offset + self.resource_capacity, len(vnf_probabilities))):
                res *= (1 - vnf_probabilities[j][1] * greedy1_switches[i][1])
            offset += self.resource_capacity
        return res

    def greedy2(self):
        res = 1
        vnf_probabilities = [(vnf, vnf_prob) for (vnf, vnf_prob) in self.failure_probabilities.items() if vnf in self.vnf_list]
        switch_probabilities = [(switch, switch_prob) for (switch, switch_prob) in self.failure_probabilities.items() if switch in self.available_backup_servers]
        switch_probabilities.sort(key = lambda x: x[1])
        num_switches = len(vnf_probabilities) // self.resource_capacity + 1
        vnf_probabilities.sort(key = lambda x: x[1])
        greedy2_switches = switch_probabilities[:num_switches]
        offset = 0
        for i in range(len(greedy2_switches)):
            for j in range(offset, min(offset + self.resource_capacity, len(vnf_probabilities))):
                res *= (1 - vnf_probabilities[j][1] * greedy2_switches[i][1])
            offset += self.resource_capacity
        return res

    def gen_information_file(self, f):
        sfc_availibility1 = self.greedy1()
        sfc_availibility2 = self.greedy2()
        f = open(f, 'w')
        f.write('This is a flow network consisting of m vnf and multiple backup servers.\n\n')
        f.write('Source Node: ' + self.source + '\n')
        f.write('Sink Node: ' + self.sink + '\n')
        f.write('VNF host nodes: \n')
        for vnf in self.vnf_list:
            f.write(vnf + '  failure_probability: ' + str(self.failure_probabilities[vnf]) + ' \n')
        f.write('\nAvailable backup servers will be all other switches in the network\n')
        for switch in self.available_backup_servers:
            f.write(switch + ' failure_probability: ' + str(self.failure_probabilities[switch]) + ' \n')
        f.write('\nSFC Availability greedy 1: ' + str(sfc_availibility1) + '\n')
        f.write('SFC Availability greedy 2: ' + str(sfc_availibility2) + '\n')
        f.write('\n\n\n\nNode mapping: \n')
        for item in self.node_mapping.items():
            f.write(str(item[0]) + ': ' + str(item[1]) + '\n')

    def gen_output_file(self, f):
        f = open(f, 'w')
        num_nodes = str(len(self.V_prime))
        num_arcs = str(len(self.E_prime))
        f.write('p min ' + num_nodes + ' ' + num_arcs + '\n')
        f.write('n ' + str(self.node_mapping[self.source]) + ' ' + str(self.supply) + '\n')
        f.write('n ' + str(self.node_mapping[self.sink]) + ' ' + str(self.demand) + '\n')
        for edge in self.E_prime:
            f.write('a ' + str(self.node_mapping[edge[0]]) + ' ' + str(self.node_mapping[edge[1]]) + ' ' + '0 ' + str(self.capacities_and_costs[(edge)][0]) + ' ' + str(self.capacities_and_costs[(edge)][1]) + '\n')
