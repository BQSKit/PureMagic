import sys
import os
import pickle
from re import findall
import argparse
from pathlib import Path
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import GroupSingleQuditGatePass
from bqskit.passes.rules import ZXZXZDecomposition
from bqskit.passes.control import ForEachBlockPass

# hack to ensure we find the quilt files - could also set PYTHONPATH before executing
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../quilt")

from quilt.measurement import PauliProductDAG
from timeit import default_timer as timer


def num_qubits(name: str) -> int:
    return int(findall(r"\d+", name.split("_")[1])[0])


def main():
    parser = argparse.ArgumentParser(description="Transpile .qasm circuits")
    parser.add_argument("--files", "-f", nargs="+", type=str, default="none", help="List of files to transpile")
    args = parser.parse_args()

    for file_name in args.files:
        print("Transpiling file", file_name, "with", num_qubits(file_name), "qubits")
        circuit = Circuit.from_file(file_name)
        # Decompose U3 gates into ZXZXZ
        passes = [
            GroupSingleQuditGatePass(),
            ForEachBlockPass(
                [ZXZXZDecomposition()],
                collection_filter=lambda x: x.num_qudits == 1,
            ),
        ]
        print("Decomposed")
        start_t = timer()
        with Compiler() as compiler:
            circuit.remove_all_measurements()
            print("Removed measurements")
            circuit = compiler.compile(circuit, passes)
            circuit.unfold_all()
        elapsed_t = timer() - start_t
        print(f"Compiled in {elapsed_t:.2f} s")
        dag = PauliProductDAG(circuit)
        dag_fname = Path(file_name).stem + ".dag.txt"
        print("Printing DAG to", dag_fname)
        dag.print(dag_fname)
        start_t = timer()
        dag.commute_all_cliffords()
        elapsed_t = timer() - start_t
        print(f"Transpiled in {elapsed_t:.2f}")

        # Save the transpiled DAG
        out_fname = Path(file_name).stem + ".dag"
        with open(f"{out_fname}", "wb") as f:
            pickle.dump(dag, f)
            dag.print(out_fname + ".done.csv")


if __name__ == "__main__":
    main()
