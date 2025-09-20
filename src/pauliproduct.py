#!/usr/bin/env -S python -u
import sys
import os
import numpy as np


class Operator:
    def __init__(self, qubit, basis):
        self.qubit = qubit
        self.basis = basis

    def __str__(self):
        return f"{self.qubit}{self.basis}"


class PauliProduct:
    def __init__(self):
        self.operators = []
        self.max_qubit = 0
        self.parents = []
        self.children = []
        self.id = -1
        self.num_ys = 0
        self.need_estabilizer = False
        self.need_ancilla = False
        self.is_clifford = False

    def set_from_str(self, product_id, s):
        # strings are of the form
        #   -______YX__________________<pi/8>
        self.id = product_id
        for i, c in enumerate(s):
            if i == 0:
                continue
            if c == "_":
                continue
            elif c in ["X", "Z", "Y"]:
                self.operators.append(Operator(i - 1, c))
                if c == "Y":
                    self.num_ys += 1
            elif c == "<":
                if s[i:] == "<M>":
                    self.is_clifford = True
                elif s[i:] == "<pi/8>":
                    self.is_clifford = False
                else:
                    raise RuntimeError(f"Unknown angle {s[i:]} in product {s}")
                break
            else:
                raise RuntimeError(f"Illegal character {c} at position {i} in product {s}")
        if self.num_ys % 2 == 1:
            self.need_ancilla = True
        self.max_qubit = max([op.qubit for op in self.operators])

    def get_product_str(self):
        return "".join(str(op) for op in self.operators).strip()

    def get_qubits(self):
        return [op.qubit for op in self.operators]

    def __str__(self):
        ancilla_str = "A" if self.num_ys % 2 == 1 else "-"
        es_str = "E" if self.need_estabilizer else "-"
        clifford_str = "clifford" if self.is_clifford else "non-clifford"
        s = (
            str(self.id)
            + " "
            + self.get_product_str()
            + " "
            + ancilla_str
            + " "
            + es_str
            + " "
            + clifford_str
            + " "
            + str(self.children)
            + " "
            + str(self.parents)
        )
        return s.strip()
