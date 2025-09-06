#!/usr/bin/env -S python -u

import os
import copy
from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once")
    import networkx as nx

from topograph import is_bus_node, is_data_node, is_magic_node, is_ancilla_node


def trim_dangling_nodes(g):
    while True:
        dangling_nodes = []
        for node in g.nodes:
            if is_bus_node(node) and g.degree(node) == 1:
                dangling_nodes.append(node)
        if len(dangling_nodes) == 0:
            break
        g.remove_nodes_from(dangling_nodes)


def get_topo_digraph(topo_graph, root_node, ancilla_node):
    topo_digraph = topo_graph.to_directed()
    # now strip the directed edges coming out of the data nodes, to prevent paths that go into then out of data nodes
    edges_to_remove = []
    for edge in topo_digraph.edges():
        if is_data_node(edge[0]):
            edges_to_remove.append(edge)
    topo_digraph.remove_edges_from(edges_to_remove)
    nodes_to_remove = []
    for node in topo_digraph.nodes():
        if (is_magic_node(node) and node != root_node) or (is_ancilla_node(node) and node != ancilla_node):
            nodes_to_remove.append(node)
    topo_digraph.remove_nodes_from(nodes_to_remove)
    return topo_digraph


def mehlhorn_steiner_tree(topo_graph, terminal_nodes, root_node, ancilla_node):
    # this is exactly like the steiner tree computation in the networkx library, except that for the dijkstra path calculation
    # and the shortest path, we use a digraph with the edges that go from the data nodes outwards removed. This prevents trees
    # that pass through the data nodes, instead of just terminating at the data nodes
    topo_digraph = get_topo_digraph(topo_graph, root_node, ancilla_node)
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
    nx.approximation.steinertree._remove_nonterminal_leaves(G_4, terminal_nodes)  # type: ignore
    edges = G_4.edges()
    T = topo_graph.edge_subgraph(edges)
    for node in T.nodes():
        if is_data_node(node) and T.degree(node) > 1:
            print("Failure in tree construction: data node", node, "has degree", T.degree(node))
    return T


def find_terminal_nodes(topo_graph, pauli_product, sched_file):
    terminal_nodes = []
    for oi, operator in enumerate(pauli_product.operators):
        # the product has blank spaces for unused qubits
        if operator == " ":
            continue
        ops = ["X", "Z"] if operator == "Y" else [operator]
        for op in ops:
            node = "d" + str(oi) + op
            if node not in topo_graph:
                print(f"Node {node} not in topo graph for finding terminal nodes", file=sched_file)
                return []
            terminal_nodes.append(node)
    return terminal_nodes


def find_best_starting_node(topo_graph, terminal_nodes, starting_nodes, sched_file):
    best_path_len = None
    best_start_node = None
    for start_node in starting_nodes:
        try:
            sum_path_len = 0.0
            for terminal_node in terminal_nodes:
                sum_path_len += nx.shortest_path_length(topo_graph, start_node, terminal_node)
            if best_path_len == None or sum_path_len < best_path_len:
                best_path_len = sum_path_len
                best_start_node = start_node
        except nx.NetworkXNoPath:
            # path not found - can't use this magic node
            print(f"Path not found", file=sched_file)
            continue
    return best_start_node


def find_best_magic_node(topo_graph, pauli_product, terminal_nodes, sched_file):
    magic_nodes = []
    for node in topo_graph.nodes:
        if is_magic_node(node):
            if topo_graph.nodes[node]["busy_count"] == 0:
                magic_nodes.append(node)
    if len(magic_nodes) == 0:
        print("Could not find starting node for Pauli product", pauli_product.__str__(), file=sched_file)
        return None
    # as the magic node, choose the one that connects to all terminals with the summed shortest path
    return find_best_starting_node(topo_graph, terminal_nodes, magic_nodes, sched_file)


def find_best_bus_node(topo_graph, terminal_nodes, sched_file):
    # can start from any bus node adjacent to a terminal node
    starting_nodes = set(
        neighbor for terminal in terminal_nodes for neighbor in topo_graph.neighbors(terminal) if is_bus_node(neighbor)
    )
    return find_best_starting_node(topo_graph, terminal_nodes, starting_nodes, sched_file)


def schedule_pauli_product(topo_graph, pauli_product, sched_file):
    terminal_nodes = find_terminal_nodes(topo_graph, pauli_product, sched_file)
    if len(terminal_nodes) == 0:
        return None
    root_node = None
    if not pauli_product.is_clifford():
        root_node = find_best_magic_node(topo_graph, pauli_product, terminal_nodes, sched_file)
        if root_node == None:
            print(f"Could not find root node for product {pauli_product}", file=sched_file)
            return None
        terminal_nodes.insert(0, root_node)
    else:
        if len(terminal_nodes) == 1:
            g = nx.Graph()
            g.add_node(terminal_nodes[0])
            return copy.deepcopy(g)
        else:
            # if there is more than one terminal, root node must be a bus node
            root_node = find_best_bus_node(topo_graph, terminal_nodes, sched_file)
            terminal_nodes.insert(0, root_node)

    ancilla_node = None
    if ancilla_node is not None:
        terminal_nodes.append(ancilla_node)

    # check path exists from root node to all other terminals
    for terminal_node in terminal_nodes[1:]:
        if not nx.has_path(topo_graph, root_node, terminal_node):
            print(
                f"Check: no path from root node {root_node} to terminal node {terminal_node} "
                f"for pp {pauli_product.get_product_str()}",
                file=sched_file,
            )
            return None
    print(
        "Trying steiner tree from root", root_node, "for", pauli_product.__str__(), "terminals", terminal_nodes, file=sched_file
    )
    g = mehlhorn_steiner_tree(topo_graph, terminal_nodes, root_node, ancilla_node)
    if not all([node in g for node in terminal_nodes]):
        print(
            f"Steiner tree: no path from root node {root_node} to terminal node for pp {pauli_product.get_product_str()}"
            f" {g.nodes}",
            file=sched_file,
        )
        return None
    return copy.deepcopy(g)


class Scheduler:
    def __init__(self, args, rank, num_ranks, rng, topo_graph):
        self.args = args
        self.rank = rank
        self.num_ranks = num_ranks
        self.rng = rng
        self.topo_graph = topo_graph
        self.sum_data_qubits = 0
        self.sum_bus_qubits = 0
        self.sched_file = None

    def check_dependencies(self, pp, scheduled):
        if pp.id in scheduled:
            raise RuntimeError("pp " + str(pp.id) + " already scheduled")
        for parent_id in pp.parents:
            if parent_id not in scheduled:
                raise RuntimeError("pp " + str(pp.id) + " scheduled before parent " + str(parent_id))

    def schedule_circuit(self, real_circuit):
        to_schedule = []
        circuit = copy.deepcopy(real_circuit)
        for pp in circuit:
            if len(pp.parents) == 0:
                to_schedule.append(pp)

        sched_fname = Path(self.args.circuit).stem + ".sched"
        self.sched_file = open(sched_fname, "w")
        num_steps = 0
        scheduled = set()
        path_dir = None
        if "paths" in self.args.plot:
            path_dir = Path(self.args.circuit).stem + ".paths"
            Path(path_dir).mkdir(exist_ok=True)
        while len(to_schedule) > 0:
            num_steps += 1
            print("Step:", num_steps, [str(pp.id) + ":" + pp.get_product_str() for pp in to_schedule], file=self.sched_file)
            title_str, pp_paths, to_schedule = self.schedule_timestep(num_steps, to_schedule)
            if pp_paths is None:
                for node in self.topo_graph.nodes:
                    if is_magic_node(node) and self.topo_graph.nodes[node]["busy_count"] > 0:
                        break
                else:
                    raise RuntimeError("Cannot schedule on current layout")
                continue
            for pp, _ in pp_paths:
                # now check if children should be added to following to_schedule
                for child_id in pp.children:
                    circuit[child_id].parents.remove(pp.id)
                    if len(circuit[child_id].parents) == 0:
                        to_schedule.append(circuit[child_id])
            if path_dir is not None and title_str is not None and num_steps > 0 and num_steps < 30:
                # don't plot too many steps
                fname_added = "." + str(num_steps)
                os.chdir(path_dir)
                self.topo_graph.plot(fname_added, pp_paths, title_str)
                os.chdir("..")
            if pp_paths is not None:
                for pp, _ in pp_paths:
                    self.check_dependencies(pp, scheduled)
                    scheduled.add(pp.id)
        print("Scheduled", len(real_circuit), "products:")
        print("  data qubit fraction: %.3f" % (float(self.sum_data_qubits) / (self.topo_graph.num_data_qubits * num_steps)))
        print("  bus qubit fraction: %.3f" % (float(self.sum_bus_qubits) / (self.topo_graph.num_bus_qubits * num_steps)))
        return num_steps, len(scheduled)

    def schedule_timestep(self, step_i, to_schedule):
        for node in self.topo_graph.nodes:
            if is_magic_node(node) and self.topo_graph.nodes[node]["busy_count"] > 0:
                self.topo_graph.nodes[node]["busy_count"] -= 1
        pp_paths = []
        working_topo_graph = copy.deepcopy(self.topo_graph)
        num_qubits_scheduled = 0
        num_bus_qubits_scheduled = 0
        num_dependent_nodes = 0
        next_to_schedule = []
        for pp in to_schedule:
            if pp.is_clifford():
                print(pp.id, "PI/4 rotation", pp, file=self.sched_file)
            if working_topo_graph.number_of_nodes() == 0:
                print("No more nodes", file=self.sched_file)
                break
            pp_graph = schedule_pauli_product(working_topo_graph, pp, self.sched_file)
            if pp_graph == None:
                print("* Could not schedule", pp, file=self.sched_file)
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
                print("Scheduled", pp.__str__(), "with", pp_graph.number_of_nodes(), "nodes", file=self.sched_file)
                pp_paths.append((pp, pp_graph))
                num_qubits_scheduled += pp.qubits_used
                for node in pp_graph.nodes():
                    if is_bus_node(node):
                        num_bus_qubits_scheduled += 1
                    if is_magic_node(node):
                        self.topo_graph.nodes[node]["busy_count"] = self.args.magic_steps

                # now remove the Pauli product path from the graph
                working_topo_graph.remove_nodes_from(pp_graph.nodes)
                orphaned_nodes = []
                for node in working_topo_graph.nodes:
                    if working_topo_graph.degree(node) == 0:
                        orphaned_nodes.append(node)
                working_topo_graph.remove_nodes_from(orphaned_nodes)

        print("Scheduling results:", file=self.sched_file)
        frac_paths = float(len(pp_paths)) / len(to_schedule)
        frac_data_qubits = float(num_qubits_scheduled) / self.topo_graph.num_data_qubits
        frac_bus_qubits = float(num_bus_qubits_scheduled) / self.topo_graph.num_bus_qubits
        print("  products:    %d/%d (%.2f)" % (len(pp_paths), len(to_schedule), frac_paths), file=self.sched_file)
        print(
            "  data qubits: %d/%d (%.2f)" % (num_qubits_scheduled, self.topo_graph.num_data_qubits, frac_data_qubits),
            file=self.sched_file,
        )
        print(
            "  bus qubits:  %d/%d (%.2f)" % (num_bus_qubits_scheduled, self.topo_graph.num_bus_qubits, frac_bus_qubits),
            file=self.sched_file,
        )
        # print("Removed", num_dependent_nodes, "dependent nodes", file=f)
        self.sum_data_qubits += num_qubits_scheduled
        self.sum_bus_qubits += num_bus_qubits_scheduled

        if len(pp_paths) > 0:
            title_str = f"Step {step_i} (pps %.2f, data %.2f, bus %.2f)" % (
                frac_paths,
                frac_data_qubits,
                frac_bus_qubits,
            )
            return title_str, pp_paths, next_to_schedule
        return None, None, next_to_schedule
