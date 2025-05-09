import sys
import os
import pickle
from re import findall
import argparse
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import GroupSingleQuditGatePass
from bqskit.passes.rules import ZXZXZDecomposition
from bqskit.passes.control import ForEachBlockPass

# hack to ensure we find the quilt files - could also set PYTHONPATH before executing
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../quilt")

from quilt.measurement import PauliProductDAG


def num_qubits(name: str) -> int:
    return int(findall(r"\d+", name)[0])


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
        with Compiler() as compiler:
            circuit = compiler.compile(circuit, passes)
            circuit.unfold_all()

        from timeit import default_timer as timer

        start_t = timer()
        dag = PauliProductDAG(circuit)
        elapsed_t = timer() - start_t
        print(f"Loading time for {file_name}: {elapsed_t:.2f} s")
        dag.commute_all_cliffords()
        stop = timer()
        elapsed_t = timer() - start_t - elapsed_t
        print(f"Transpilation time for {file_name}: {elapsed_t:.2f}")

        # Save the transpiled DAG
        with open(f"{file_name}.dag", "wb") as f:
            pickle.dump(dag, f)


if __name__ == "__main__":
    main()
