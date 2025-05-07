#!/usr/bin/env -S python -u
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../quilt")
import quilt


class PauliProduct:
    def __init__(self, num_qubits):
        self.operators = [" "] * num_qubits
        self.start_qubit = 0
        self.qubits_used = 0
        self.parents = set()
        self.children = set()
        self.angle = 0
        self.id = -1

    def set_rnd(self, rng, pauli_product_qubits, start_qubit):
        self.basis_options = ["X", "Z", "Y"]
        self.start_qubit = start_qubit
        self.qubits_used = pauli_product_qubits
        for i in range(start_qubit, start_qubit + pauli_product_qubits):
            self.operators[i] = self.basis_options[int(np.floor(rng.uniform(0, len(self.basis_options))))]

    def set(self, pp_id, quilt_pp, parents, children):
        for q, b in quilt_pp.qubit_basis.items():
            if isinstance(b, quilt.pauli.PauliX):
                self.operators[q] = "X"
            elif isinstance(b, quilt.pauli.PauliZ):
                self.operators[q] = "Z"
            elif isinstance(b, quilt.pauli.PauliY):
                self.operators[q] = "Y"
            else:
                raise RuntimeError("Operator not supported: " + str(b))
        self.angle = quilt_pp.angle
        self.parents = [parent.id for parent in parents]
        self.children = [child.id for child in children]
        self.id = pp_id

    def get_product_str(self):
        s = ""
        for i in range(len(self.operators)):
            if self.operators[i] != " ":
                s += str(i) + self.operators[i] + " "
        return s.strip()

    def get_qubits(self):
        return [i for i, o in enumerate(self.operators) if o != " "]

    def __str__(self):
        s = str(self.id) + " " + self.get_product_str()
        s += " ["
        for parent in self.parents:
            s += str(parent) + ","
        s += "] ["
        for child in self.children:
            s += str(child) + ","
        s += "]"
        return s.strip()
