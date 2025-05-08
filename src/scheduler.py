#!/usr/bin/env -S python -u

import networkx as nx
import copy
import numpy as np
from topograph import is_bus_node, is_data_node, is_magic_node


def trim_dangling_nodes(g):
    while True:
        dangling_nodes = []
        for node in g.nodes:
            if is_bus_node(node) and g.degree(node) == 1:
                dangling_nodes.append(node)
        if len(dangling_nodes) == 0:
            break
        g.remove_nodes_from(dangling_nodes)


def schedule_pauli_product_bfs(topo_graph, pauli_product, root_node):
    visited = {root_node}
    num_found_operators = 0
    for node in topo_graph.nodes():
        if is_magic_node(node):
            visited.add(node)
    queue = [root_node]
    pauli_product_graph = nx.Graph()
    num_expected_ops = pauli_product.qubits_used
    for op in pauli_product.operators:
        if op == "Y":
            num_expected_ops += 1
    while len(queue):
        node = queue.pop(0)
        pauli_product_graph.add_node(node)
        # look for data nodes first
        for nb in topo_graph[node]:
            if nb not in visited and is_data_node(nb):
                visited.add(nb)
                qubit_index = int(nb[1:-1])
                qubit_basis = nb[-1]
                if qubit_index >= len(pauli_product.operators):
                    continue
                qbs = [pauli_product.operators[qubit_index]]
                if qbs[0] == "Y":
                    qbs = ["X", "Z"]
                if qubit_basis in qbs:
                    # print("Found basis", qubit_basis, "at node", nb)
                    pauli_product_graph.add_edge(node, nb)
                    num_found_operators += 1
                    if num_found_operators == num_expected_ops:
                        trim_dangling_nodes(pauli_product_graph)
                        return pauli_product_graph
        # now extend along the bus
        for nb in topo_graph[node]:
            if not is_data_node(nb) and nb not in visited:
                visited.add(nb)
                queue.append(nb)
                pauli_product_graph.add_edge(node, nb)
    return None


def get_topo_digraph(topo_graph):
    topo_digraph = topo_graph.to_directed()
    # now strip the directed edges coming out of the data nodes, to prevent paths that go into then out of data nodes
    edges_to_remove = []
    for edge in topo_digraph.edges():
        if is_data_node(edge[0]):
            edges_to_remove.append(edge)
    topo_digraph.remove_edges_from(edges_to_remove)
    return topo_digraph


def mehlhorn_steiner_tree(topo_graph, terminal_nodes):
    # this is exactly like the steiner tree computation in the networkx library, except that for the dijkstra path calculation
    # and the shortest path, we use a digraph with the edges that go from the data nodes outwards removed. This prevents trees
    # that pass through the data nodes, instead of just terminating at the data nodes
    topo_digraph = get_topo_digraph(topo_graph)
    paths = nx.multi_source_dijkstra_path(topo_digraph, terminal_nodes)

    d_1 = {}
    s = {}
    for v in topo_graph.nodes():
        if v not in paths:
            continue
        s[v] = paths[v][0]
        d_1[(v, s[v])] = len(paths[v]) - 1

    # G1-G4 names match those from the Mehlhorn 1988 paper.
    G_1_prime = nx.Graph()
    for u, v, data in topo_graph.edges(data=True):
        if u not in s or v not in s:
            continue
        su, sv = s[u], s[v]
        weight_here = d_1[(u, su)] + data.get("weight", 1) + d_1[(v, sv)]
        if not G_1_prime.has_edge(su, sv):
            G_1_prime.add_edge(su, sv, weight=weight_here)
        else:
            new_weight = min(weight_here, G_1_prime[su][sv]["weight"])
            G_1_prime.add_edge(su, sv, weight=new_weight)

    G_2 = nx.minimum_spanning_edges(G_1_prime, data=True)

    G_3 = nx.Graph()
    for u, v, d in G_2:
        path = nx.shortest_path(topo_digraph, u, v, "weight")
        for n1, n2 in nx.utils.pairwise(path):
            G_3.add_edge(n1, n2)

    G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
    G_4 = topo_graph.edge_subgraph(G_3_mst).copy()
    nx.approximation.steinertree._remove_nonterminal_leaves(G_4, terminal_nodes)
    edges = G_4.edges()
    T = topo_graph.edge_subgraph(edges)
    for node in T.nodes():
        if is_data_node(node) and T.degree(node) > 1:
            print("Failure in tree construction: data node", node, "has degree", T.degree(node))
    return T


def schedule_pauli_product_steiner(topo_graph, pauli_product, root_node):
    # print("trying steiner tree from root", root_node, "for", pauli_product.__str__(), "terminals", terminal_nodes)
    terminal_nodes = [root_node]
    for oi, operator in enumerate(pauli_product.operators):
        if operator != " ":
            ops = ["X", "Z"] if operator == "Y" else [operator]
            for op in ops:
                node = "d" + str(oi) + op
                if node not in topo_graph:
                    return None
                terminal_nodes.append(node)
    for terminal_node in terminal_nodes[1:]:
        if not nx.has_path(topo_graph, root_node, terminal_node):
            return None
    g = mehlhorn_steiner_tree(topo_graph, terminal_nodes)
    if not all([node in g for node in terminal_nodes]):
        return None
    return g


def find_best_magic_node(topo_graph, pauli_product):
    magic_nodes = []
    for node in topo_graph.nodes:
        if is_magic_node(node):
            magic_nodes.append(node)
    if len(magic_nodes) == 0:
        # print("Could not find starting node for Pauli product", pauli_product.__str__())
        return None
    terminal_nodes = []
    for oi, operator in enumerate(pauli_product.operators):
        if operator != " ":
            ops = ["X", "Z"] if operator == "Y" else [operator]
            for op in ops:
                node = "d" + str(oi) + op
                if node not in topo_graph:
                    return None
                terminal_nodes.append(node)
    # as the magic node, choose the one that connects to all terminals with the summed shortest path
    best_path_len = None
    best_magic_node = None
    for magic_node in magic_nodes:
        try:
            sum_path_len = 0.0
            for terminal_node in terminal_nodes:
                sum_path_len += nx.shortest_path_length(topo_graph, magic_node, terminal_node)
            if best_path_len == None or sum_path_len < best_path_len:
                best_path_len = sum_path_len
                best_magic_node = magic_node
        except nx.NetworkXNoPath:
            # path not found - can't use this magic node
            continue
    return best_magic_node


def add_double_edges(topo_graph, pauli_product):
    for i in range(0, len(pauli_product.operators), 2):
        if i + 1 >= len(pauli_product.operators):
            break
        operators = pauli_product.operators
        if operators[i] == " " or operators[i + 1] == " ":
            continue
        if operators[i] == operators[i + 1]:
            # print("found matching operators", operators[i], "at positions", i, i + 1)
            node = "d" + str(i) + operators[i]
            if not topo_graph.has_node(node):
                return topo_graph
            other_node = topo_graph.nodes[node]["other"]
            # print("adding edge", node, other_node)
            topo_graph.add_edge(node, other_node)
            node = "d" + str(i + 1) + operators[i + 1]
            # print("adding edge", node, other_node)
            topo_graph.add_edge(node, other_node)
        if operators[i] == "Y" or operators[i + 1] == "Y":
            # not sure how exactly to handle this
            continue
            print("found Y operators", operators[i], operators[i + 1], "at positions", i, i + 1)
    return topo_graph


def schedule_pauli_product(args, topo_graph, pauli_product):
    # if args.verbose:
    #    print(pauli_product.__str__())
    if args.topbottom:
        topo_graph = add_double_edges(topo_graph, pauli_product)
    root_node = find_best_magic_node(topo_graph, pauli_product)
    if root_node == None:
        return None
    # schedule from each available magic node in turn to find the first that works
    # for root_node in magic_nodes:
    if args.path_method == "bfs":
        g = schedule_pauli_product_bfs(topo_graph, pauli_product, root_node)
    elif args.path_method == "steiner":
        g = schedule_pauli_product_steiner(topo_graph, pauli_product, root_node)
    else:
        raise ValueError("Unknown path method " + args.path_method)
    if g == None:
        # continue
        return None
    # return the first one we find - far more efficient and seems to give similar results to trying to find the shortest
    return copy.deepcopy(g)


class Scheduler:
    def __init__(self, args, rank, num_ranks, rng, topo_graph):
        self.args = args
        self.rank = rank
        self.num_ranks = num_ranks
        self.rng = rng
        self.topo_graph = topo_graph

    def schedule_circuit(self, real_circuit):
        to_schedule = []
        circuit = copy.deepcopy(real_circuit)
        for pp in circuit:
            if len(pp.parents) == 0:
                to_schedule.append(pp)
        f = open("sched.txt", "w")
        num_steps = 0
        scheduled = set()
        while len(to_schedule) > 0:
            num_steps += 1
            print("Step:", num_steps, [str(pp.id) + ":" + pp.get_product_str() for pp in to_schedule], file=f)
            prev_to_schedule = to_schedule.copy()
            title_str, pp_paths, to_schedule = self.schedule_timestep(to_schedule, circuit, f)
            if prev_to_schedule == to_schedule:
                raise RuntimeError("Cannot schedule on current layout")
            if title_str is not None and "paths" in self.args.plot and num_steps <= 20:
                fname = "lssp-topo-path-" + str(num_steps) + "-" + self.args.path_method
                self.topo_graph.plot(fname, pp_paths, title_str)
            if pp_paths is not None:
                # check for failed dependencies
                for pp, _ in pp_paths:
                    if pp.id in scheduled:
                        raise RuntimeError("pp " + str(pp.id) + " already scheduled")
                    for parent_id in pp.parents:
                        if parent_id not in scheduled:
                            raise RuntimeError("pp " + str(pp.id) + " scheduled before parent " + str(parent_id))
                    scheduled.add(pp.id)
            # if num_steps == 3:
            #    break
        return num_steps, len(scheduled)

    def schedule_timestep(self, to_schedule, circuit, f):
        pp_paths = []
        working_topo_graph = copy.deepcopy(self.topo_graph)
        num_qubits_scheduled = 0
        num_bus_qubits_scheduled = 0
        num_dependent_nodes = 0
        next_to_schedule = []
        for pp in to_schedule:
            if working_topo_graph.number_of_nodes() == 0:
                print("No more nodes", file=f)
                break
            pp_graph = schedule_pauli_product(self.args, working_topo_graph, pp)
            if pp_graph == None:
                print("* Could not schedule", pp, file=f)
                next_to_schedule.append(pp)
                # now the circuit could include multiple timeteps, so we need to ensure dependencies are met
                # if the product couldn't be scheduled, then every qubit in that product is now out of bounds so remove from
                # the graph
                nodes_to_remove = []
                for i, operator in enumerate(pp.operators):
                    if operator != " ":
                        nodes_to_remove.append("d" + str(i) + operator)
                if len(nodes_to_remove) > 0:
                    num_dependent_nodes += len(nodes_to_remove)
                    working_topo_graph.remove_nodes_from(nodes_to_remove)
            else:
                print("Scheduled", pp.__str__(), "with", pp_graph.number_of_nodes(), "nodes", file=f)
                pp_paths.append((pp, pp_graph))
                num_qubits_scheduled += pp.qubits_used
                for node in pp_graph.nodes():
                    if is_bus_node(node):
                        num_bus_qubits_scheduled += 1
                # now remove the Pauli product path from the graph
                working_topo_graph.remove_nodes_from(pp_graph.nodes)
                orphaned_nodes = []
                for node in working_topo_graph.nodes:
                    if working_topo_graph.degree(node) == 0:
                        orphaned_nodes.append(node)
                working_topo_graph.remove_nodes_from(orphaned_nodes)

        for pp, _ in pp_paths:
            # now check if children should be added to next to schedule
            for child_id in pp.children:
                circuit[child_id].parents.remove(pp.id)
                if len(circuit[child_id].parents) == 0:
                    next_to_schedule.append(circuit[child_id])

        if self.args.verbose:
            print("Scheduling results:")
        frac_paths = float(len(pp_paths)) / len(to_schedule)
        frac_data_qubits = float(num_qubits_scheduled) / self.topo_graph.num_data_qubits
        frac_bus_qubits = float(num_bus_qubits_scheduled) / self.topo_graph.num_bus_qubits
        if self.args.verbose:
            print("  Pauli products:  %d/%d (%.2f)" % (len(pp_paths), len(to_schedule), frac_paths))
            print("  data qubits:     %d/%d (%.2f)" % (num_qubits_scheduled, self.topo_graph.num_data_qubits, frac_data_qubits))
            print("  bus qubits:     %d/%d (%.2f)" % (num_bus_qubits_scheduled, self.topo_graph.num_bus_qubits, frac_bus_qubits))
            print("Removed", num_dependent_nodes, "dependent nodes")

        if len(pp_paths) > 0:
            title_str = self.args.path_method + " (pps %.2f, data %.2f, bus %.2f)" % (
                frac_paths,
                frac_data_qubits,
                frac_bus_qubits,
            )
            return title_str, pp_paths, next_to_schedule
        return None, None, next_to_schedule

    def schedule_rnd_circuit(self, circuit):
        all_cycles = []
        for cycle in circuit:
            all_cycles.extend(cycle)
        num_steps = 0
        while len(all_cycles) > 0:
            title_str, pauli_product_paths, remaining_circuit_cycle = self.schedule_cycle(all_cycles)
            all_cycles = remaining_circuit_cycle
            if title_str is not None and "paths" in self.args.plot:
                fname = "lssp-topo-path-" + str(num_steps) + "-" + self.args.path_method
                self.topo_graph.plot(fname, pauli_product_paths, title_str)
            num_steps += 1
        return num_steps

    def schedule_circuit_barrier(self, circuit):
        num_steps = 0
        for ci, circuit_cycle in enumerate(circuit):
            if ci % self.num_ranks == self.rank:
                remaining_circuit_cycle = circuit_cycle
                for i in range(100):
                    title_str, pauli_product_paths, remaining_circuit_cycle = self.schedule_cycle(circuit_cycle)
                    if title_str is not None and "paths" in self.args.plot:
                        fname = "lssp-topo-path-" + str(i) + "-" + str(ci) + "-" + self.args.path_method
                        self.topo_graph.plot(fname, pauli_product_paths, title_str)
                    circuit_cycle = remaining_circuit_cycle
                    if len(circuit_cycle) == 0:
                        break
                    # now add products from the next cycle
                if self.args.verbose:
                    print("Scheduled full circuit cycle in", i + 1, "time steps")
                num_steps += i + 1
        return num_steps

    def schedule_cycle(self, circuit):
        pauli_product_paths = []
        working_topo_graph = copy.deepcopy(self.topo_graph)
        num_qubits_scheduled = 0
        num_bus_qubits_scheduled = 0
        remaining_circuit = []
        num_dependent_nodes = 0
        for pauli_product in circuit:
            if working_topo_graph.number_of_nodes() == 0:
                # print("No more nodes")
                break
            pauli_product_graph = schedule_pauli_product(self.args, working_topo_graph, pauli_product)
            if pauli_product_graph == None:
                # print("* Could not schedule Pauli product", pauli_product)
                remaining_circuit.append(pauli_product)
                # now the circuit could include multiple cycles, so we need to ensure dependencies are met
                # if the product couldn't be scheduled, then every qubit in that product is now out of bounds so remove from
                # the graph
                nodes_to_remove = []
                for i, operator in enumerate(pauli_product.operators):
                    if operator != " ":
                        nodes_to_remove.append("d" + str(i) + operator)
                if len(nodes_to_remove) > 0:
                    num_dependent_nodes += len(nodes_to_remove)
                    working_topo_graph.remove_nodes_from(nodes_to_remove)
                continue
            # print("Scheduled Pauli product", pauli_product.__str__(), "with", pauli_product_graph.number_of_nodes(), "nodes")
            pauli_product_paths.append((pauli_product, pauli_product_graph))
            num_qubits_scheduled += pauli_product.qubits_used
            for node in pauli_product_graph.nodes():
                if is_bus_node(node):
                    num_bus_qubits_scheduled += 1
            # now remove the Pauli product path from the graph
            working_topo_graph.remove_nodes_from(pauli_product_graph.nodes)
            orphaned_nodes = []
            for node in working_topo_graph.nodes:
                if working_topo_graph.degree(node) == 0:
                    orphaned_nodes.append(node)
            working_topo_graph.remove_nodes_from(orphaned_nodes)

        if self.args.verbose:
            print("Scheduling results:")
        frac_paths = float(len(pauli_product_paths)) / len(circuit)
        frac_data_qubits = float(num_qubits_scheduled) / self.topo_graph.num_data_qubits
        frac_bus_qubits = float(num_bus_qubits_scheduled) / self.topo_graph.num_bus_qubits
        if self.args.verbose:
            print("  Pauli products:  %d/%d (%.2f)" % (len(pauli_product_paths), len(circuit), frac_paths))
            print("  data qubits:     %d/%d (%.2f)" % (num_qubits_scheduled, self.topo_graph.num_data_qubits, frac_data_qubits))
            print("  bus qubits:     %d/%d (%.2f)" % (num_bus_qubits_scheduled, self.topo_graph.num_bus_qubits, frac_bus_qubits))
            print("Removed", num_dependent_nodes, "dependent nodes")

        if len(pauli_product_paths) > 0:
            title_str = self.args.path_method + " (pps %.2f, data %.2f, bus %.2f)" % (
                frac_paths,
                frac_data_qubits,
                frac_bus_qubits,
            )
            return title_str, pauli_product_paths, remaining_circuit
        return None, None, remaining_circuit
