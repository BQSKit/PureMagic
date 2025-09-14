#!/usr/bin/env -S python -u

import os
import sys
import copy
from pathlib import Path
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once")
    import networkx as nx

from topograph import is_bus_node, is_data_node, is_magic_node, is_ancilla_node, is_estabilizer_node
from utils import timer


class Scheduler:
    def __init__(self, args, rank, num_ranks, rng, topo_graph):
        self.args = args
        self.rank = rank
        self.num_ranks = num_ranks
        self.rng = rng
        self.topo_graph = topo_graph
        self.sum_data_qubits = 0
        self.sum_bus_qubits = 0
        self.sum_magic_qubits = 0
        self.sum_ancilla_qubits = 0
        self.sum_estabilizer_qubits = 0
        self.sched_file = None
        self.busy_count_list = []

    def print_sched(self, message):
        if self.sched_file is not None:
            # only evaluate the message function if schedule printing is enabled
            print(message, file=self.sched_file)

    def check_dependencies(self, pp, scheduled):
        if pp.id in scheduled:
            raise RuntimeError("pp " + str(pp.id) + " already scheduled")
        for parent_id in pp.parents:
            if parent_id not in scheduled:
                raise RuntimeError(
                    "pp " + str(pp.id) + " scheduled before parent " + str(parent_id)
                )

    def get_topo_digraph(self, g, root_node, ancilla_node, estabilizer_node):
        dg = g.to_directed()
        # now strip the directed edges coming out of the data nodes, to prevent paths that go into
        # then out of data nodes
        edges_to_remove = []
        for edge in dg.edges():
            if is_data_node(edge[0]):
                edges_to_remove.append(edge)
        dg.remove_edges_from(edges_to_remove)
        nodes_to_remove = []
        for node in dg.nodes():
            if (
                (is_magic_node(node) and node != root_node)
                or (is_ancilla_node(node) and node != ancilla_node)
                or (is_estabilizer_node(node) and node != estabilizer_node)
            ):
                nodes_to_remove.append(node)
        dg.remove_nodes_from(nodes_to_remove)
        return dg

    def mehlhorn_steiner_tree(self, g, terminal_nodes, root_node, ancilla_node, estabilizer_node):
        # this is exactly like the steiner tree computation in the networkx library, except that
        # for the dijkstra path calculation and the shortest path, we use a digraph with the edges
        # that go from the data nodes outwards removed. This prevents trees that pass through the
        # data nodes, instead of just terminating at the data nodes
        dg = self.get_topo_digraph(g, root_node, ancilla_node, estabilizer_node)
        paths = nx.multi_source_dijkstra_path(dg, terminal_nodes)

        d_1 = {}
        s = {}
        for v in g.nodes():
            if v not in paths:
                continue
            s[v] = paths[v][0]
            d_1[(v, s[v])] = len(paths[v]) - 1

        # G1-G4 names match those from the Mehlhorn 1988 paper.
        G_1_prime = nx.Graph()
        for u, v, data in g.edges(data=True):
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
        for u, v, _ in G_2:
            path = nx.shortest_path(dg, u, v, "weight")
            for n1, n2 in nx.utils.pairwise(path):
                G_3.add_edge(n1, n2)

        G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
        G_4 = g.edge_subgraph(G_3_mst).copy()
        nx.approximation.steinertree._remove_nonterminal_leaves(G_4, terminal_nodes)  # type: ignore
        edges = G_4.edges()
        T = g.edge_subgraph(edges)
        for node in T.nodes():
            if is_data_node(node) and T.degree(node) > 1:
                print("Failure in tree construction: data node", node, "has degree", T.degree(node))
        return T

    def trim_dangling_nodes(self, g):
        while True:
            dangling_nodes = []
            for node in g.nodes:
                if is_bus_node(node) and g.degree(node) == 1:
                    dangling_nodes.append(node)
            if len(dangling_nodes) == 0:
                break
            g.remove_nodes_from(dangling_nodes)

    def get_topo_subgraph(self, g, terminal_nodes, root_node, ancilla_node, estabilizer_node):
        subg = copy.deepcopy(g)
        nodes_to_remove = []
        for node in subg.nodes():
            if is_magic_node(node) and node != root_node:
                nodes_to_remove.append(node)
            elif is_ancilla_node(node) and node != ancilla_node:
                nodes_to_remove.append(node)
            elif is_estabilizer_node(node) and node != estabilizer_node:
                nodes_to_remove.append(node)
            elif is_data_node(node) and node not in terminal_nodes:
                nodes_to_remove.append(node)
        subg.remove_nodes_from(nodes_to_remove)
        return subg

    def get_bfs_schedule(
        self, working_topo_graph, terminal_nodes, root_node, ancilla_node, estabilizer_node
    ):
        g = self.get_topo_subgraph(
            working_topo_graph, terminal_nodes, root_node, ancilla_node, estabilizer_node
        )
        visited = set([root_node])
        # need to keep track of whether we have visited the estabilizer because we visit it twice
        visited_estabilizer_again = False
        num_found_terminals = 1
        queue = [root_node]
        pauli_product_graph = nx.Graph()
        while len(queue):
            node = queue.pop(0)
            pauli_product_graph.add_node(node)
            if is_data_node(root_node):
                return pauli_product_graph
            for nb in g[node]:
                if nb in visited:
                    if not is_estabilizer_node(nb):
                        continue
                    elif visited_estabilizer_again:
                        continue
                    else:
                        visited_estabilizer_again = True
                visited.add(nb)
                pauli_product_graph.add_edge(node, nb)
                # if is_bus_node(nb) or is_estabilizer_node(node):
                if is_bus_node(nb):
                    queue.append(nb)
                else:
                    num_found_terminals += 1
                    if num_found_terminals == len(terminal_nodes):
                        self.trim_dangling_nodes(pauli_product_graph)
                        return pauli_product_graph
        return None

    def find_terminal_nodes(self, g, pauli_product):
        terminal_nodes = []
        for operator in pauli_product.operators:
            node = "d" + str(operator.qubit) + operator.basis.upper()
            if node not in g:
                return []
            terminal_nodes.append(node)
        return terminal_nodes

    def find_best_starting_node(self, g, terminal_nodes, starting_nodes):
        # find the starting node that connects to all terminals with the summed shortest path
        best_path_len = None
        best_start_node = None
        for start_node in starting_nodes:
            try:
                sum_path_len = 0.0
                for terminal_node in terminal_nodes:
                    sum_path_len += nx.shortest_path_length(g, start_node, terminal_node)
                if best_path_len == None or sum_path_len < best_path_len:
                    best_path_len = sum_path_len
                    best_start_node = start_node
            except nx.NetworkXNoPath:
                # path not found - can't use this starting node
                self.print_sched(f"Path not found from {start_node} to terminals {terminal_nodes}")
                continue
        return best_start_node

    def find_best_magic_node(self, g, terminal_nodes):
        magic_nodes = [
            node for node in g.nodes if is_magic_node(node) and g.nodes[node]["busy_count"] == 0
        ]
        if len(magic_nodes) == 0:
            return None
        return self.find_best_starting_node(g, terminal_nodes, magic_nodes)

    def find_best_bus_node(self, g, terminal_nodes):
        # can start from any bus node adjacent to a terminal node
        starting_nodes = set(
            neighbor
            for terminal in terminal_nodes
            for neighbor in g.neighbors(terminal)
            if is_bus_node(neighbor)
        )
        if len(starting_nodes) == 0:
            self.print_sched(f"Could not find starting bus node for terminals {terminal_nodes}")
            return None
        return self.find_best_starting_node(g, terminal_nodes, starting_nodes)

    def find_best_ancilla_node(self, g, terminal_nodes):
        ancilla_nodes = [node for node in g.nodes if is_ancilla_node(node)]
        if len(ancilla_nodes) == 0:
            self.print_sched(f"Could not find ancilla node for terminals {terminal_nodes}")
            return None
        return self.find_best_starting_node(g, terminal_nodes, ancilla_nodes)

    def find_best_estabilizer_node(self, g, terminal_nodes):
        estabilizer_nodes = [node for node in g.nodes if is_estabilizer_node(node)]
        if len(estabilizer_nodes) == 0:
            self.print_sched(f"Could not find estabilizer node for terminals {terminal_nodes}")
            return None
        return self.find_best_starting_node(g, terminal_nodes, estabilizer_nodes)

    def schedule_pauli_product(self, working_topo_graph, pauli_product):
        terminal_nodes = self.find_terminal_nodes(working_topo_graph, pauli_product)
        if len(terminal_nodes) == 0:
            return None
        root_node = None
        if not pauli_product.is_clifford():
            root_node = self.find_best_magic_node(working_topo_graph, terminal_nodes)
            if root_node == None:
                return None
            terminal_nodes.insert(0, root_node)
        else:
            if len(terminal_nodes) == 1:
                g = nx.Graph()
                g.add_node(terminal_nodes[0])
                return copy.deepcopy(g)
            else:
                # if there is more than one terminal, root node must be a bus node
                root_node = self.find_best_bus_node(working_topo_graph, terminal_nodes)
                if root_node is None:
                    return None
                terminal_nodes.insert(0, root_node)

        ancilla_node = None
        if pauli_product.num_ys > 0 and pauli_product.num_ys % 2 != 0:
            ancilla_node = self.find_best_ancilla_node(working_topo_graph, terminal_nodes)
            if ancilla_node == None:
                return None
            terminal_nodes.append(ancilla_node)

        estabilizer_node = None
        if pauli_product.need_estabilizer:
            estabilizer_node = self.find_best_estabilizer_node(working_topo_graph, terminal_nodes)
            if estabilizer_node == None:
                return None
            terminal_nodes.append(estabilizer_node)
            # append a second time to force a path through the estabilizer
            terminal_nodes.append(estabilizer_node)

        # check path exists from root node to all other terminals
        for terminal_node in terminal_nodes[1:]:
            if not nx.has_path(working_topo_graph, root_node, terminal_node):
                self.print_sched(
                    f"Check: no path from root node {root_node} to terminal node "
                    f"{terminal_node} for pp {pauli_product.get_product_str()}",
                )
                return None
        if self.args.use_steiner_trees:
            self.print_sched(
                f"Trying steiner tree from root {root_node} for {pauli_product}"
                f", terminals {terminal_nodes}"
            )
            g = self.mehlhorn_steiner_tree(
                working_topo_graph, terminal_nodes, root_node, ancilla_node, estabilizer_node
            )
        else:
            self.print_sched(
                f"Trying BFS from root {root_node} for {pauli_product}"
                f", terminals {terminal_nodes}"
            )
            g = self.get_bfs_schedule(
                working_topo_graph, terminal_nodes, root_node, ancilla_node, estabilizer_node
            )

        if g is None or not all([node in g for node in terminal_nodes]):
            self.print_sched(
                f"No path from root node {root_node} to terminal node for pp"
                f" {pauli_product.get_product_str()}",
            )
            return None
        return copy.deepcopy(g)

    def gen_busy_count(self):
        busy_count = int(round(self.rng.exponential(scale=1.0 / self.args.magic_state_lambda)))
        busy_count += 1
        self.busy_count_list.append(busy_count)
        return busy_count

    def schedule_timestep(self, step_i, to_schedule):
        num_busy = 0
        for node in self.topo_graph.nodes:
            if is_magic_node(node) and self.topo_graph.nodes[node]["busy_count"] > 0:
                self.topo_graph.nodes[node]["busy_count"] -= 1
                if self.topo_graph.nodes[node]["busy_count"] > 0:
                    num_busy += 1
        pp_paths = []
        working_topo_graph = copy.deepcopy(self.topo_graph)
        num_scheduled = 0
        num_bus_scheduled = 0
        num_data_scheduled = 0
        num_magic_scheduled = num_busy
        num_ancilla_scheduled = 0
        num_estabilizers_scheduled = 0
        num_dependent_nodes = 0
        next_to_schedule = []
        for pp in to_schedule:
            if pp.is_clifford():
                self.print_sched(f"{pp.id} PI/4 rotation {pp}")
            if working_topo_graph.number_of_nodes() == 0:
                self.print_sched("No more nodes")
                break
            pp_graph = self.schedule_pauli_product(working_topo_graph, pp)
            if pp_graph == None:
                self.print_sched(f"* Could not schedule {pp}")
                next_to_schedule.append(pp)
                # now the circuit could include multiple timeteps, so we need to ensure dependencies
                # are met if the product couldn't be scheduled, then every qubit in that product is
                # now out of bounds so remove from the graph
                nodes_to_remove = []
                for operator in pp.operators:
                    nodes_to_remove.append("d" + operator.__str__())
                if len(nodes_to_remove) > 0:
                    num_dependent_nodes += len(nodes_to_remove)
                    working_topo_graph.remove_nodes_from(nodes_to_remove)
            else:
                num_nodes = pp_graph.number_of_nodes()
                self.print_sched(f"Scheduled {pp.__str__()} with {num_nodes} nodes")
                pp_paths.append((pp, pp_graph))
                num_scheduled += len(pp.operators)
                for node in pp_graph.nodes():
                    if is_bus_node(node):
                        num_bus_scheduled += 1
                    elif is_magic_node(node):
                        self.topo_graph.nodes[node]["busy_count"] = self.gen_busy_count()
                        # self.print_sched(
                        #    f"Busy count for node {node} is set to "
                        #    f"{self.topo_graph.nodes[node]["busy_count"]}"
                        # )
                        num_magic_scheduled += 1
                    elif is_data_node(node):
                        num_data_scheduled += 1
                    elif is_ancilla_node(node):
                        num_ancilla_scheduled += 1
                    elif is_estabilizer_node(node):
                        num_estabilizers_scheduled += 1
                # now remove the Pauli product path from the graph
                working_topo_graph.remove_nodes_from(pp_graph.nodes)
                orphaned_nodes = []
                for node in working_topo_graph.nodes:
                    if working_topo_graph.degree(node) == 0:
                        orphaned_nodes.append(node)
                working_topo_graph.remove_nodes_from(orphaned_nodes)

        self.print_sched("Scheduling results:")
        frac_paths = float(len(pp_paths)) / len(to_schedule)
        frac_data = float(num_data_scheduled) / self.topo_graph.num_data_qubits
        frac_bus = float(num_bus_scheduled) / self.topo_graph.num_bus_qubits
        frac_magic = float(num_magic_scheduled) / self.topo_graph.num_magic_qubits
        frac_ancilla = float(num_ancilla_scheduled) / self.topo_graph.num_ancilla_qubits
        frac_estabilizers = (
            float(num_estabilizers_scheduled) / self.topo_graph.num_estabilizer_qubits
        )
        self.print_sched(f"  products:    {len(pp_paths)}/{len(to_schedule)} ({frac_paths:.2f})")
        self.print_sched(
            f"  data:        {num_data_scheduled}/{self.topo_graph.num_data_qubits} "
            f"({frac_data:.2f})"
        )
        self.print_sched(
            f"  bus:         {num_bus_scheduled}/{self.topo_graph.num_bus_qubits} "
            f"({frac_bus:.2f})",
        )
        self.print_sched(
            f"  magic:       {num_magic_scheduled}/{self.topo_graph.num_magic_qubits} "
            f"({frac_magic:.2f})",
        )
        self.print_sched(
            f"  ancilla:     {num_ancilla_scheduled}/{self.topo_graph.num_ancilla_qubits} "
            f"({frac_ancilla:.2f})",
        )
        self.print_sched(
            f"  estabilizer: {num_estabilizers_scheduled}/{self.topo_graph.num_estabilizer_qubits} "
            f"({frac_estabilizers:.2f})",
        )
        self.print_sched(f"Removed {num_dependent_nodes} dependent nodes")
        self.sum_data_qubits += num_scheduled
        self.sum_bus_qubits += num_bus_scheduled
        self.sum_magic_qubits += num_magic_scheduled
        self.sum_ancilla_qubits += num_ancilla_scheduled
        self.sum_estabilizer_qubits += num_estabilizers_scheduled

        if len(pp_paths) > 0:
            title_str = (
                f"Step {step_i} Products scheduled {frac_paths:.2f}, data {frac_data:.2f}"
                f", bus {frac_bus:.2f}"  # , magic {frac_magic:.2f}, ancilla {frac_ancilla:.2f}"
            )
            return title_str, pp_paths, next_to_schedule
        else:
            return None, None, next_to_schedule

    @timer
    def schedule_circuit(self, real_circuit):
        global sched_file
        if self.args.log_scheduler:
            sched_fname = Path(self.args.circuit).stem + ".sched"
            self.sched_file = open(sched_fname, "w")
        # initialize all magic nodes to require cultivation starting from round 0
        for node in self.topo_graph.nodes():
            if is_magic_node(node):
                self.topo_graph.nodes[node]["busy_count"] = self.gen_busy_count()
                # self.print_sched(
                #    f"Busy count for node {node} is set to "
                #    f"{self.topo_graph.nodes[node]["busy_count"]}"
                # )
        to_schedule = []
        circuit = copy.deepcopy(real_circuit)
        for pp in circuit:
            if len(pp.parents) == 0:
                to_schedule.append(pp)
        num_steps = 0
        scheduled = set()
        path_dir = None
        plot_steps = 0
        if "paths" in self.args.plot:
            path_dir = Path(self.args.circuit).stem + ".paths"
            Path(path_dir).mkdir(exist_ok=True)
            plot_steps = 100
        total_to_schedule = len(real_circuit)
        prev_perc_complete = 0
        print(f"Scheduling {total_to_schedule} products:    ", end="")
        if plot_steps > 0:
            print("")
        while len(to_schedule) > 0:
            num_steps += 1
            if plot_steps == 0:
                perc_complete = int((len(scheduled) / total_to_schedule) * 100)
                if perc_complete > prev_perc_complete:
                    print(f"\x08\x08\x08{perc_complete:02}%", end="")
                    prev_perc_complete = perc_complete
            self.print_sched(
                f"Step: {num_steps}"
                f"{[str(pp.id) + ":" + pp.get_product_str() for pp in to_schedule]}"
            )
            node_labels = {}
            if path_dir is not None and num_steps > 0 and plot_steps > 0:
                for _, node in enumerate(self.topo_graph.nodes()):
                    if is_magic_node(node) and self.topo_graph.nodes[node]["busy_count"] > 0:
                        node_labels[node] = f"{self.topo_graph.nodes[node]["busy_count"] - 1}"
                    else:
                        node_labels[node] = node

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
            if path_dir is not None and title_str is not None and num_steps > 0 and plot_steps > 0:
                # don't plot too many steps
                fname_added = "." + str(num_steps)
                os.chdir(path_dir)
                self.topo_graph.plot(fname_added, pp_paths, title_str, node_labels)
                os.chdir("..")
                plot_steps -= 1
            if pp_paths is not None:
                for pp, _ in pp_paths:
                    self.check_dependencies(pp, scheduled)
                    scheduled.add(pp.id)
        data_frac = float(self.sum_data_qubits) / (self.topo_graph.num_data_qubits * num_steps)
        bus_frac = float(self.sum_bus_qubits) / (self.topo_graph.num_bus_qubits * num_steps)
        magic_frac = float(self.sum_magic_qubits) / (self.topo_graph.num_magic_qubits * num_steps)
        ancilla_frac = float(self.sum_ancilla_qubits) / (
            self.topo_graph.num_ancilla_qubits * num_steps
        )
        estabilizer_frac = float(self.sum_estabilizer_qubits) / (
            self.topo_graph.num_estabilizer_qubits * num_steps
        )
        overall_frac = (
            float(
                # we always need all the data qubits, even if they don't get fully utilized
                self.topo_graph.num_data_qubits * num_steps
                # self.sum_data_qubits
                + self.sum_bus_qubits
                + self.sum_magic_qubits
                + self.sum_ancilla_qubits
                + self.sum_estabilizer_qubits
            )
            / num_steps
            / self.topo_graph.num_qubits
        )

        print("\nQubit fractions used:")
        print(f"  data:        {data_frac:.3f}")
        print(f"  bus:         {bus_frac:.3f}")
        print(f"  magic:       {magic_frac:.3f}")
        print(f"  ancilla:     {ancilla_frac:.3f}")
        print(f"  estabilizer: {estabilizer_frac:.3f}")
        # print(f"  overall:     {overall_frac:.3f}")
        print("Magic state cultivation time:")
        print(f"  average: {np.mean(self.busy_count_list):.2f}")
        print(f"  min:     {np.min(self.busy_count_list):.0f}")
        print(f"  max:     {np.max(self.busy_count_list):.0f}")
        return num_steps, len(scheduled), overall_frac
