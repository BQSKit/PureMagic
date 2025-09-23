#!/usr/bin/env -S python -u

import os
import sys
import copy
from pathlib import Path
import numpy as np
import warnings
import time

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once")
    import networkx as nx

from topograph import is_bus_node, is_data_node, is_magic_node, is_ancilla_node, is_estabilizer_node
from utils import timer


def get_node_pos(node):
    node_col, node_row = node[1:].split("-")
    return int(node_col), int(node_row)


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
                        # Match the which_ancilla with the correct side
                        # (left or top for X, right or bottom for Y)
                        node_col, node_row = get_node_pos(node)
                        nb_col, nb_row = get_node_pos(nb)
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

    def get_bfs_graph(self, root_node, terminal_nodes, which_ancilla, nodes_to_exclude):
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
                if (
                    nodes_to_exclude is not None
                    and nb in nodes_to_exclude
                    and nb not in terminal_nodes
                ):
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

    def find_best_tree(self, root_nodes, data_nodes, pauli_product):
        best_graph = None
        self.print_sched(f"  Find best tree for {pauli_product}:")
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
            if best_graph is None or g.number_of_edges() < best_graph.number_of_edges():
                best_graph = g
        return best_graph

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
            which_ancilla = (
                pauli_product.operators[0].basis.upper() if pauli_product.need_ancilla else ""
            )
            estabilizer_nodes = [
                node
                for node in self.topo_graph.nodes
                if is_estabilizer_node(node) and not self.topo_graph.nodes[node]["used"]
            ]
            magic_graphs = []
            # find shortest path from a magic node to an estabilizer node
            for magic_node in magic_nodes:
                for estabilizer_node in estabilizer_nodes:
                    magic_path_g = self.get_bfs_graph(magic_node, [estabilizer_node], "", None)
                    if magic_path_g is None:
                        continue
                    self.print_sched(
                        f"  Found graph from {magic_node} to {estabilizer_node} of size "
                        f"{magic_path_g.number_of_edges()}"
                    )
                    magic_graphs.append((magic_node, estabilizer_node, magic_path_g))
            if len(magic_graphs) == 0:
                self.print_sched(f"  No path found from {magic_nodes} to " f"  {estabilizer_nodes}")
                return None
            # sort the magic graphs that we will try the shortest paths first when looking for the
            # following estabilizer-data tree
            magic_graphs.sort(key=lambda x: x[2].number_of_edges())

            # get graph connecting estabilizers to terminal nodes
            best_g = None
            best_size = None
            selected_magic_graph = None
            for magic_node, estabilizer_node, magic_path_g in magic_graphs:
                if best_size is not None and best_size <= magic_path_g.number_of_edges():
                    # in this case we cannot get a shorter graph from this magic->estabilizer path
                    continue
                estabilizer_graph = self.get_bfs_graph(
                    estabilizer_node, data_nodes, which_ancilla, magic_path_g
                )
                if estabilizer_graph is not None:
                    self.print_sched(
                        f"  Found graph from {estabilizer_node} ({magic_node}) of size "
                        f"{estabilizer_graph.number_of_edges()}"
                    )
                    graph_size = (
                        magic_path_g.number_of_edges() + estabilizer_graph.number_of_edges()
                    )
                    if best_g is None or graph_size < best_size:
                        best_g = estabilizer_graph
                        best_size = graph_size
                        selected_magic_graph = magic_path_g
            if best_g is None or selected_magic_graph is None:
                self.print_sched(
                    f"  No path from estabilizer nodes {estabilizer_nodes} to {data_nodes}"
                )
                return None
            # finally, connect the magic->estabilizer graph with the estabilizer-terminals graph
            for node in best_g.nodes:
                if not is_estabilizer_node(node):
                    selected_magic_graph.add_node(node)
            for edge in best_g.edges:
                selected_magic_graph.add_edge(*edge)
            return selected_magic_graph
        else:
            return self.find_best_tree(magic_nodes, data_nodes, pauli_product)

    def schedule_clifford(self, data_nodes, pauli_product):
        # FIXME: deal with estabilizers and ancilla
        # root node needs to be a bus node next to one of the data nodes
        root_nodes = set()
        for node in data_nodes:
            if self.topo_graph.nodes[node]["used"]:
                continue
            for nb in self.topo_graph.neighbors(node):
                if self.topo_graph.nodes[nb]["used"]:
                    continue
                if is_bus_node(nb):
                    root_nodes.add(nb)
        if len(root_nodes) == 0:
            return None
        return self.find_best_tree(root_nodes, data_nodes, pauli_product)

    def schedule_pauli_product(self, pauli_product):
        self.print_sched(f"Trying to schedule {pauli_product}")
        # initially terminal nodes contain onlly the data qubits
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
            if self.topo_graph.number_of_nodes() == 0:
                self.print_sched("No more nodes")
                break
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
        # frac_ancilla = float(num_ancilla_scheduled) / self.topo_graph.num_ancilla_qubits
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
        # self.print_sched(
        #    f"  ancilla:     {num_ancilla_scheduled}/{self.topo_graph.num_ancilla_qubits} "
        #    f"({frac_ancilla:.2f})",
        # )
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
