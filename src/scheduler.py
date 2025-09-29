#!/usr/bin/env -S python -u

import os
import sys
import copy
from pathlib import Path
import numpy as np
import warnings
import time
import math

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

    def get_node_dist(self, node1, node2):
        pos1 = self.topo_graph.nodes[node1]["pos"]
        pos2 = self.topo_graph.nodes[node2]["pos"]
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def trim_dangling_nodes(self, g):
        while True:
            dangling_nodes = []
            for node in g.nodes:
                if is_bus_node(node) and g.degree(node) == 1:
                    dangling_nodes.append(node)
            if len(dangling_nodes) == 0:
                break
            g.remove_nodes_from(dangling_nodes)

    def find_ancilla(self, g, which_ancilla):
        for node in g.nodes:
            if is_bus_node(node):
                for nb in self.topo_graph.neighbors(node):
                    if (
                        is_bus_node(nb)
                        and not self.topo_graph.nodes[nb]["used"]
                        and nb not in g.nodes
                    ):
                        # FIXME: is the left and right really necessary given we can create the
                        # ancilla on the fly?
                        match_ancilla = False
                        # Match the which_ancilla with the correct side
                        # (left or top for X, right or bottom for Y)
                        if match_ancilla:
                            node_col, node_row = self.topo_graph.node[node]["pos"]
                            nb_col, nb_row = self.topo_graph.node[nb]["pos"]
                            self.print_sched(
                                f"NODE {nb} pos {nb_col},{nb_row} "
                                f"attr {self.topo_graph.nodes[nb]["pos"]}"
                            )
                            if which_ancilla == "X":
                                if node_col >= nb_col and node_row >= nb_row:
                                    continue
                            elif which_ancilla == "Z":
                                if node_col <= nb_col and node_row <= nb_row:
                                    continue
                            else:
                                raise RuntimeError(f"illegal which_ancilla {which_ancilla}")
                        self.print_sched(f"    Selecting {nb} as {which_ancilla}")
                        g.add_node(nb)
                        g.add_edge(node, nb)
                        return True
        return False

    def get_bfs_graph(self, root_node, terminal_nodes, which_ancilla, exclude):
        visited = set([root_node])
        queue = [root_node]
        bfs_graph = nx.Graph()
        num_terminals_reqd = len(terminal_nodes)
        num_found_terminals = 0
        while len(queue):
            node = queue.pop(0)
            bfs_graph.add_node(node)
            for nb in self.topo_graph.neighbors(node):
                if self.topo_graph.nodes[nb]["used"]:
                    continue
                if nb in visited:
                    continue
                if exclude is not None and nb in exclude and nb not in terminal_nodes:
                    continue
                if not is_bus_node(nb) and not nb in terminal_nodes:
                    continue
                visited.add(nb)
                bfs_graph.add_edge(node, nb)
                if is_bus_node(nb):
                    queue.append(nb)
                else:
                    if not is_ancilla_node(nb):
                        num_found_terminals += 1
                    if num_found_terminals == num_terminals_reqd:
                        self.trim_dangling_nodes(bfs_graph)
                        if which_ancilla != "":
                            if not self.find_ancilla(bfs_graph, which_ancilla):
                                return None
                        return bfs_graph
        return None

    def get_nodes_by_dist(self, nodes, pauli_product):
        # sort nodes by distance to pp data nodes
        min_distances = []
        for node in nodes:
            min_d = 1000000
            for op in pauli_product.operators:
                data_node = f"d{str(op).upper()}"
                d = self.get_node_dist(node, data_node)
                if d < min_d:
                    min_d = d
            min_distances.append(min_d)
        node_distances = list(zip(nodes, min_distances))
        node_distances.sort(key=lambda x: x[1])
        return node_distances

    def find_tree(self, root_nodes, data_nodes, pauli_product):
        self.print_sched(f"  Find tree for {pauli_product}:")
        which_ancilla = (
            pauli_product.operators[0].basis.upper() if pauli_product.need_ancilla else ""
        )
        for root_node in root_nodes:
            g = self.get_bfs_graph(root_node, data_nodes, which_ancilla, None)
            if g is None:
                self.print_sched(
                    f"    No tree from root node {root_node} to {data_nodes}, "
                    f"{which_ancilla if which_ancilla != "" else ""} "
                )
                continue
            self.print_sched(
                f"    Tree from {root_node} to {data_nodes} has size " f"{g.number_of_edges()}"
            )
            return g
        return None

    def find_estabilizer_tree(self, magic_nodes, data_nodes, pauli_product):
        which_ancilla = (
            pauli_product.operators[0].basis.upper() if pauli_product.need_ancilla else ""
        )
        estabilizer_nodes = [
            node
            for node in self.topo_graph.nodes
            if is_estabilizer_node(node) and not self.topo_graph.nodes[node]["used"]
        ]
        estabilizer_distances = self.get_nodes_by_dist(estabilizer_nodes, pauli_product)
        magic_path_dists = []
        for magic_node in magic_nodes:
            for estabilizer_node, estabilizer_d in estabilizer_distances:
                d = self.get_node_dist(magic_node, estabilizer_node) + estabilizer_d
                magic_path_dists.append((magic_node, estabilizer_node, d))
        magic_path_dists.sort(key=lambda x: x[2])
        for magic_node, estabilizer_node, d in magic_path_dists:
            magic_path_g = self.get_bfs_graph(magic_node, [estabilizer_node], "", exclude=None)
            if magic_path_g is None:
                self.print_sched(f"  No path from {magic_node} to {estabilizer_node}")
                continue
            self.print_sched(
                f"  Found graph from {magic_node} to {estabilizer_node} of size "
                f"{magic_path_g.number_of_edges()}"
            )
            estabilizer_g = self.get_bfs_graph(
                estabilizer_node, data_nodes, which_ancilla, exclude=magic_path_g
            )
            if estabilizer_g is None:
                self.print_sched(f"  No path from {estabilizer_node} to {pauli_product}")
                continue
            self.print_sched(
                f"  Found graph from {estabilizer_node} ({magic_node}) of size "
                f"{estabilizer_g.number_of_edges()}"
            )
            # Now connect the magic->estabilizer graph with the estabilizer-terminals graph
            for node in magic_path_g.nodes:
                if not is_estabilizer_node(node):
                    estabilizer_g.add_node(node)
            for edge in magic_path_g.edges:
                estabilizer_g.add_edge(*edge)
            self.print_sched(
                f"  Final graph has {estabilizer_g.number_of_edges()} edges "
                f"(estimated distance {d:.0f})"
            )
            return estabilizer_g
        self.print_sched(f"  No path from estabilizer nodes {estabilizer_nodes} to {data_nodes}")
        return None

    def schedule_non_clifford(self, data_nodes, pauli_product):
        magic_nodes = [
            node
            for node in self.topo_graph.nodes
            if is_magic_node(node) and self.topo_graph.nodes[node]["busy_count"] == 0
        ]
        if len(magic_nodes) == 0:
            self.print_sched("  No available magic nodes")
            return None
        if pauli_product.need_estabilizer:
            return self.find_estabilizer_tree(magic_nodes, data_nodes, pauli_product)
        else:
            magic_nodes_sorted = [
                node for node, _ in self.get_nodes_by_dist(magic_nodes, pauli_product)
            ]
            return self.find_tree(magic_nodes_sorted, data_nodes, pauli_product)

    def schedule_clifford(self, data_nodes, pauli_product):
        # FIXME: deal with estabilizers and ancilla
        if len(data_nodes) == 1:
            node = data_nodes[0]
            if self.topo_graph.nodes[node]["used"]:
                return None
            g = nx.Graph()
            g.add_node(node)
            if pauli_product.need_ancilla:
                for nb in self.topo_graph.neighbors(node):
                    if is_bus_node(nb) and not self.topo_graph.nodes[nb]["used"]:
                        g.add_node(nb)
                        g.add_edge(node, nb)
                        break
                else:
                    return None
            self.print_sched(f"Scheduled clifford on {g.nodes} nodes")
            return g
        # root node needs to be a bus node next to one of the data nodes
        root_nodes = set()
        for node in data_nodes:
            if self.topo_graph.nodes[node]["used"]:
                return None
            for nb in self.topo_graph.neighbors(node):
                if self.topo_graph.nodes[nb]["used"]:
                    continue
                if is_bus_node(nb):
                    root_nodes.add(nb)
        if len(root_nodes) == 0:
            return None
        g = self.find_tree(root_nodes, data_nodes, pauli_product)
        if not g is None:
            self.print_sched(f"Scheduled clifford in {g.nodes} nodes")
        return g

    def schedule_pauli_product(self, pauli_product):
        self.print_sched(f"Trying to schedule {pauli_product}")
        # initially terminal nodes contain only the data qubits
        data_nodes = []
        for operator in pauli_product.operators:
            node = "d" + str(operator.qubit) + operator.basis.upper()
            if self.topo_graph.nodes[node]["used"]:
                self.print_sched(f"  Node {node} is already used")
                return None
            data_nodes.append(node)
        if len(data_nodes) == 0:
            self.print_sched(f"  No data nodes found in working graph")
            return None
        if not pauli_product.is_clifford:
            return self.schedule_non_clifford(data_nodes, pauli_product)
        else:
            return self.schedule_clifford(data_nodes, pauli_product)

    def gen_busy_count(self):
        busy_count = int(round(self.rng.exponential(scale=1.0 / self.args.magic_state_lambda)))
        busy_count += 1
        self.busy_count_list.append(busy_count)
        return busy_count

    def schedule_timestep(self, step_i, to_schedule):
        start_t = time.perf_counter()

        num_busy = 0
        for node in self.topo_graph.nodes:
            if is_magic_node(node) and self.topo_graph.nodes[node]["busy_count"] > 0:
                self.topo_graph.nodes[node]["busy_count"] -= 1
                if self.topo_graph.nodes[node]["busy_count"] > 0:
                    num_busy += 1
            # ensure all nodes are available at the start of the timestep
            self.topo_graph.nodes[node]["used"] = False

        # sort the pps to schedule; reverse sort seems to work better
        to_schedule.sort(key=lambda pp: len(pp.operators), reverse=True)

        pp_paths = []
        # working_topo_graph = copy.deepcopy(self.topo_graph)
        num_scheduled = 0
        num_bus_scheduled = 0
        num_data_scheduled = 0
        num_magic_scheduled = num_busy
        num_ancilla_scheduled = 0
        num_estabilizers_scheduled = 0
        num_dependent_nodes = 0
        next_to_schedule = []
        schedule_pp_timer = 0
        for pp in to_schedule:
            t = time.perf_counter()
            pp_graph = self.schedule_pauli_product(pp)
            schedule_pp_timer += time.perf_counter() - t

            if pp_graph == None:
                self.print_sched("  * Could not schedule on graph")
                next_to_schedule.append(pp)
                # now the circuit could include multiple timeteps, so we need to ensure dependencies
                # are met if the product couldn't be scheduled, then every qubit in that product is
                # now out of bounds so remove from the graph
                for operator in pp.operators:
                    self.topo_graph.nodes["d" + operator.__str__().upper()]["used"] = True
                    num_dependent_nodes += 1
            else:
                num_nodes = pp_graph.number_of_nodes()
                self.print_sched(
                    f"  * Scheduled with {num_nodes} nodes and {pp_graph.number_of_edges()} edges"
                )
                pp_paths.append((pp, pp_graph))
                num_scheduled += len(pp.operators)
                for node in pp_graph.nodes():
                    if is_bus_node(node):
                        num_bus_scheduled += 1
                    elif is_magic_node(node):
                        self.topo_graph.nodes[node]["busy_count"] = self.gen_busy_count()
                        num_magic_scheduled += 1
                    elif is_data_node(node):
                        num_data_scheduled += 1
                    elif is_ancilla_node(node):
                        num_ancilla_scheduled += 1
                    elif is_estabilizer_node(node):
                        num_estabilizers_scheduled += 1
                    # ensure we can't use this node again
                    self.topo_graph.nodes[node]["used"] = True

        self.print_sched("Scheduling results:")
        frac_paths = float(len(pp_paths)) / len(to_schedule)
        frac_data = float(num_data_scheduled) / self.topo_graph.num_data_qubits
        frac_bus = float(num_bus_scheduled) / self.topo_graph.num_bus_qubits
        frac_magic = float(num_magic_scheduled) / self.topo_graph.num_magic_qubits
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
            f"  estabilizer: {num_estabilizers_scheduled}/{self.topo_graph.num_estabilizer_qubits} "
            f"({frac_estabilizers:.2f})",
        )
        self.print_sched(f"Removed {num_dependent_nodes} dependent nodes")
        end_t = time.perf_counter()
        elapsed_time = end_t - start_t
        self.print_sched(
            f"Scheduling timestep took {elapsed_time:0.4f} s, "
            f" scheduling paulis took {schedule_pp_timer:0.4f} s"
        )

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
                f"Step {num_steps}: "
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
        # ancilla_frac = float(self.sum_ancilla_qubits) / (
        #    self.topo_graph.num_ancilla_qubits * num_steps
        # )
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
        # print(f"  ancilla:     {ancilla_frac:.3f}")
        print(f"  estabilizer: {estabilizer_frac:.3f}")
        # print(f"  overall:     {overall_frac:.3f}")
        print("Magic state cultivation time:")
        print(f"  average: {np.mean(self.busy_count_list):.2f}")
        print(f"  min:     {np.min(self.busy_count_list):.0f}")
        print(f"  max:     {np.max(self.busy_count_list):.0f}")
        return num_steps, len(scheduled), overall_frac
