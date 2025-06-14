#!/usr/bin/env -S python -u
import sys
import os
import numpy as np

# hack to ensure we find the quilt files - could also set PYTHONPATH before executing
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../quilt")
import quilt


class PauliProduct:
    def __init__(self, num_qubits):
        self.operators = [" "] * num_qubits
        self.qubits_used = 0
        self.parents = set()
        self.children = set()
        self.angle = 0
        self.id = -1

    def set(self, pp_id, quilt_pp, parents, children):
        for q, b in quilt_pp.qubit_basis.items():
            self.qubits_used += 1
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

    def is_pi_over_four(self):
        return self.angle.is_clifford()
        # return self.angle.numerator == 1 and self.angle.denominator == 4

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
