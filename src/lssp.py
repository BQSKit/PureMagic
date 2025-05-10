#!/usr/bin/env -S python -u

import numpy as np
import argparse
import multiprocessing as mp
import topograph
import realcircuit
import scheduler
from utils import timer


def get_args():
    parser = argparse.ArgumentParser(description="Experimental scheduler for the LSSP")
    parser.add_argument("--min-num-qubits", "-n", type=int, default=10, help="Minimum number of data qubits")
    parser.add_argument("--rseed", "-r", type=int, default=29, help="Random seed")
    path_methods = ["steiner", "bfs"]
    parser.add_argument(
        "--path-method",
        type=str,
        default="bfs",
        choices=path_methods,
        help="Method to use for finding paths: " + ", ".join(path_methods),
    )
    parser.add_argument("--threads", "-t", type=int, default=0, help="Number of processes for multiprocessing")
    parser.add_argument("--bus-ratio", "-s", type=int, default=1, help="Ratio of double qubit rows to bus rows")
    parser.add_argument("--double-bus", action="store_true", help="Double columns for bus qubits")
    plot_options = ["none", "circuit", "paths", "freqs", "topo"]
    parser.add_argument("--plot", "-p", nargs="+", type=str, default="none", choices=plot_options, help="Plotting")
    # parser.add_argument("--plot-circuit-range", type=str, default="", help="Min and max depths of circuit to plot: NN:NN")
    parser.add_argument("--circuit", "-c", type=str, required=True, default="None", help="Circuit pickle file name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    # parser.add_argument("--topbottom", action="store_true", help="Use top and bottom of double data qubits")
    parser.add_argument("--rnd-order", action="store_true", help="Randomly order the qubits")
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
        self.num_scheduled = mp.Value("i", 0)
        self.circuit = circuit
        self.scheduler = scheduler.Scheduler(args, rank, num_ranks, rng, topo_graph)

    def run(self):
        self.num_steps.value, self.num_scheduled.value = self.scheduler.schedule_circuit_barrier(self.circuit)


@timer
def schedule_multiprocessing(num_ranks, rng, topo_graph, circuit):
    proc = [None] * num_ranks
    for rank in range(num_ranks):
        proc[rank] = ScheduleProcess(rank, num_ranks, rng, topo_graph, circuit)
        proc[rank].start()
    for rank in range(num_ranks):
        proc[rank].join()
    tot_num_steps = 0
    tot_num_scheduled = 0
    for rank in range(num_ranks):
        tot_num_steps += proc[rank].num_steps.value
        tot_num_scheduled += proc[rank].num_scheduled.value
    return tot_num_steps, tot_num_scheduled


@timer
def main():
    num_ranks = mp.cpu_count() if args.threads == 0 else args.threads
    print("Running on", num_ranks, "cores")
    rng = np.random.default_rng(seed=args.rseed)
    topo_graph = topograph.TopoGraph()
    topo_graph.set_dims(args, rng)
    if topo_graph.num_data_qubits != args.min_num_qubits:
        print("Adjusted number of data qubits from", args.min_num_qubits, "to", topo_graph.num_data_qubits)
    if "topo" in args.plot:
        topo_graph.plot(".topo")
    circuit = realcircuit.RealCircuit(args)
    if "circuit" in args.plot:
        circuit.plot()
    if "freqs" in args.plot:
        circuit.plot_freqs()
    if args.barrier:
        tot_num_steps, num_scheduled = schedule_multiprocessing(num_ranks, rng, topo_graph, circuit)
    else:
        single_scheduler = scheduler.Scheduler(args, 0, 1, rng, topo_graph)
        tot_num_steps, num_scheduled = single_scheduler.schedule_circuit(circuit)
    speedup = float(num_scheduled) / tot_num_steps
    print(f"Scheduled {num_scheduled} in {tot_num_steps} time steps ({speedup:.3f} speedup)")
    num_layers = len(circuit.get_layers())
    speedup = float(num_scheduled) / num_layers
    print(f"Optimal time steps {num_layers} ({speedup:.3f} speedup)")


if __name__ == "__main__":
    args = get_args()
    main()
