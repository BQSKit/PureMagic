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

    def trim_dangling_nodes(self, g):
        while True:
            dangling_nodes = []
            for node in g.nodes:
                if is_bus_node(node) and g.degree(node) == 1:
                    dangling_nodes.append(node)
            if len(dangling_nodes) == 0:
                break
            g.remove_nodes_from(dangling_nodes)

    def get_bfs_graph(self, g, root_node, terminal_nodes, need_ancilla, nodes_to_exclude):
        visited = set([root_node])
        queue = [root_node]
        bfs_graph = nx.Graph()
        found_ancilla = False
        num_terminals_reqd = len(terminal_nodes)
        num_found_terminals = 0
        while len(queue):
            node = queue.pop(0)
            bfs_graph.add_node(node)
            for nb in g[node]:
                if (
                    nodes_to_exclude is not None
                    and nb in nodes_to_exclude
                    and nb not in terminal_nodes
                ):
                    continue
                if not is_bus_node(nb):
                    if need_ancilla and not found_ancilla and is_ancilla_node(nb):
                        found_ancilla = True
                    elif not nb in terminal_nodes:
                        continue
                if nb in visited:
                    continue
                visited.add(nb)
                bfs_graph.add_edge(node, nb)
                if is_bus_node(nb):
                    queue.append(nb)
                else:
                    if not is_ancilla_node(nb):
                        num_found_terminals += 1
                    if num_found_terminals == num_terminals_reqd:
                        if need_ancilla and not found_ancilla:
                            continue
                        self.trim_dangling_nodes(bfs_graph)
                        return bfs_graph
        return None

    def find_best_tree(self, working_topo_graph, root_nodes, data_nodes, pauli_product):
        best_graph = None
        self.print_sched(f"Paths for {pauli_product}:")
        for root_node in root_nodes:
            g = self.get_bfs_graph(
                working_topo_graph, root_node, data_nodes, pauli_product.need_ancilla, None
            )
            if g is None:
                self.print_sched(f"  no tree from root node {root_node}")
                continue
            self.print_sched(f"  tree from {root_node} has length {g.number_of_edges()}")
            if best_graph is None or g.number_of_edges() < best_graph.number_of_edges():
                best_graph = g
        return best_graph

    def schedule_non_clifford(self, working_topo_graph, data_nodes, pauli_product):
        magic_nodes = [
            node
            for node in working_topo_graph.nodes
            if is_magic_node(node) and working_topo_graph.nodes[node]["busy_count"] == 0
        ]
        if len(magic_nodes) == 0:
            return None
        if pauli_product.need_estabilizer:
            self.print_sched(f"Scheduling non-clifford with estabilizer {pauli_product}")
            estabilizer_nodes = [
                node for node in working_topo_graph.nodes if is_estabilizer_node(node)
            ]
            # get candidate graphs connecting estabilizer to terminal nodes
            estabilizer_graphs = []
            for estabilizer_node in estabilizer_nodes:
                estabilizer_graph = self.get_bfs_graph(
                    working_topo_graph,
                    estabilizer_node,
                    data_nodes,
                    pauli_product.need_ancilla,
                    None,
                )
                if estabilizer_graph is not None:
                    self.print_sched(
                        f"  found graph from {estabilizer_node} of size "
                        f"{estabilizer_graph.number_of_edges()}"
                    )
                    estabilizer_graphs.append((estabilizer_node, estabilizer_graph))
            # get shortest path from magic node to terminals through estabilizer
            best_path_g = None
            best_graph_size = None
            selected_estabilizer_graph = None
            for magic_node in magic_nodes:
                for estabilizer_node, estabilizer_graph in estabilizer_graphs:
                    path_g = self.get_bfs_graph(
                        working_topo_graph,
                        magic_node,
                        [estabilizer_node],
                        False,
                        estabilizer_graph.nodes,
                    )
                    if path_g is None:
                        continue
                    graph_size = path_g.number_of_edges() + estabilizer_graph.number_of_edges()
                    if best_path_g is None or graph_size < best_graph_size:
                        best_path_g = path_g
                        best_graph_size = graph_size
                        selected_estabilizer_graph = estabilizer_graph
            if best_path_g is None or selected_estabilizer_graph is None:
                return None
            # finally, connect the magic->estabilizer graph with the estabilizer-terminals graph
            for node in best_path_g.nodes:
                if not is_estabilizer_node(node):
                    selected_estabilizer_graph.add_node(node)
            for edge in best_path_g.edges:
                selected_estabilizer_graph.add_edge(*edge)
            return selected_estabilizer_graph
        else:
            return self.find_best_tree(working_topo_graph, magic_nodes, data_nodes, pauli_product)

    def schedule_clifford(self, working_topo_graph, data_nodes, pauli_product):
        # cliffords don't have Ys
        assert not pauli_product.need_estabilizer and not pauli_product.need_ancilla
        # root node needs to be a bus node next to one of the data nodes
        root_nodes = set()
        for node in data_nodes:
            for nb in working_topo_graph[node]:
                if is_bus_node(nb):
                    root_nodes.add(nb)
        if len(root_nodes) == 0:
            return None
        return self.find_best_tree(working_topo_graph, root_nodes, data_nodes, pauli_product)

    def schedule_pauli_product(self, working_topo_graph, pauli_product):
        # initially terminal nodes contain onlly the data qubits
        data_nodes = []
        for operator in pauli_product.operators:
            node = "d" + str(operator.qubit) + operator.basis.upper()
            if node not in working_topo_graph:
                return None
            data_nodes.append(node)
        if len(data_nodes) == 0:
            return None
        if not pauli_product.is_clifford():
            return self.schedule_non_clifford(working_topo_graph, data_nodes, pauli_product)
        else:
            return self.schedule_clifford(working_topo_graph, data_nodes, pauli_product)

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
                        num_magic_scheduled += 1
                    elif is_data_node(node):
                        num_data_scheduled += 1
                    elif is_ancilla_node(node):
                        num_ancilla_scheduled += 1
                    elif is_estabilizer_node(node):
                        num_estabilizers_scheduled += 1
                # now remove the Pauli product path from the graph
                working_topo_graph.remove_nodes_from(pp_graph.nodes)  # type: ignore
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
