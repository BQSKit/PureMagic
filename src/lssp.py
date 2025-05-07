#!/usr/bin/env -S python -u

import os
import sys
import numpy as np
import argparse
import multiprocessing as mp
import topograph
import rndcircuit
import realcircuit
import scheduler
from utils import timer

exec_path = os.path.dirname(os.path.realpath(__file__))
print(exec_path)
sys.path.insert(0, exec_path + "/../../quilt")
import quilt


def get_args():
    parser = argparse.ArgumentParser(description="Experimental scheduler for the LSSP")
    parser.add_argument("--min-num-qubits", "-n", type=int, default=10, help="Minimum number of data qubits")
    parser.add_argument(
        "--qubits-per-pauli-product",
        "-q",
        type=float,
        default=0.1,
        help="Mean fraction data qubits per Pauli product (normal distribution)",
    )
    parser.add_argument("--circuit-depth", "-d", type=int, default=1, help="Depth of the circuit")
    parser.add_argument("--rseed", "-r", type=int, default=29, help="Random seed")
    parser.add_argument("--gap-prob", "-g", type=float, default=0.5, help="Probability of a gap in the circuit at a qubit")
    path_methods = ["steiner", "bfs"]
    parser.add_argument(
        "--path-method",
        type=str,
        default="bfs",
        choices=path_methods,
        help="Method to use for finding paths: " + ", ".join(path_methods),
    )
    parser.add_argument("--threads", "-t", type=int, default=0, help="Number of processes for multiprocessing")
    plot_options = ["none", "circuit", "paths", "freqs"]
    parser.add_argument(
        "--plot", "-p", nargs="+", type=str, default="none", choices=plot_options, help="Plot: " + ", ".join(plot_options)
    )
    layout_options = ["spaced", "compact"]
    parser.add_argument(
        "--layout", "-l", type=str, default="spaced", choices=layout_options, help="Layout, one of " + ", ".join(plot_options)
    )
    parser.add_argument("--circuit", "-c", type=str, default="random", help="Circuit: random or pickle file name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--topbottom", action="store_true", help="Use top and bottom of double data qubits")
    parser.add_argument("--barrier", "-b", action="store_true", help="Use barrier after every cycle")
    args = parser.parse_args()
    if args.barrier == False:
        args.threads = 1
        print("No barrier used: setting threads to 1 because multithreading doesn't work without the barrier")
    print("Arguments:\n ", "\n  ".join(f"{k}={v}" for k, v in vars(args).items()))
    return args


class ScheduleProcess(mp.Process):
    def __init__(self, rank, num_ranks, rng, topo_graph, circuit):
        mp.Process.__init__(self)
        self.num_steps = mp.Value("i", 0)
        self.circuit = circuit
        self.scheduler = scheduler.Scheduler(args, rank, num_ranks, rng, topo_graph)
        self.use_barrier = args.barrier

    def run(self):
        if self.use_barrier == True:
            self.num_steps.value = self.scheduler.schedule_circuit_barrier(self.circuit)
        else:
            self.num_steps.value = self.scheduler.schedule_circuit(self.circuit)


@timer
def schedule_multiprocessing(num_ranks, rng, topo_graph, circuit):
    proc = [None] * num_ranks
    for rank in range(num_ranks):
        proc[rank] = ScheduleProcess(0, num_ranks, rng, topo_graph, circuit)
        proc[rank].start()
    for rank in range(num_ranks):
        proc[rank].join()
    tot_num_steps = 0
    for rank in range(num_ranks):
        tot_num_steps += proc[rank].num_steps.value
    return tot_num_steps


@timer
def main():
    num_ranks = mp.cpu_count() if args.threads == 0 else args.threads
    print("Running on", num_ranks, "cores")
    rng = np.random.default_rng(seed=args.rseed)
    topo_graph = topograph.TopoGraph()
    topo_graph.set_dims(args)
    if topo_graph.num_data_qubits != args.min_num_qubits:
        print("Adjusted number of data qubits from", args.min_num_qubits, "to", topo_graph.num_data_qubits)
    # topo_graph.plot("lssp-topo")
    if args.circuit == "random":
        circuit = rndcircuit.RndCircuit(args, rng, topo_graph.num_data_qubits)
    else:
        circuit = realcircuit.RealCircuit(args, rng, topo_graph.num_data_qubits)

    if "circuit" in args.plot:
        circuit.plot()
    if "freqs" in args.plot:
        circuit.plot_freqs()

    # tot_num_steps = schedule_circuit(0, 1, rng, topo_graph, circuit)
    tot_num_steps = schedule_multiprocessing(num_ranks, rng, topo_graph, circuit)
    print("Scheduled full circuit in", tot_num_steps, "(%.2f efficiency)" % (float(args.circuit_depth) / tot_num_steps))


if __name__ == "__main__":
    args = get_args()
    main()
