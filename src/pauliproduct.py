#!/usr/bin/env -S python -u
import sys
import os
import numpy as np
from ast import literal_eval

# hack to ensure we find the quilt files - could also set PYTHONPATH before executing
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../quilt")
import quilt.angle


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
        self.angle = None
        self.id = -1
        self.num_ys = 0
        self.need_estabilizer = False

    def set_vals(self, pp_id, pp_str, parents_str, children_str):
        terms = pp_str.split(".")
        self.id = pp_id
        self.parents = [int(x) for x in literal_eval(parents_str)]
        self.children = [int(x) for x in literal_eval(children_str)]
        terms = pp_str.split(".")
        for term in terms:
            if not term.startswith("Pauli"):
                raise RuntimeError("Term does not start with Pauli")
            self.operators.append(Operator(int(term.split(")")[1].split("<")[0]), term[5]))
            if self.operators[-1].basis == "Y":
                self.num_ys += 1
            # phase = term[7:0]
        self.max_qubit = max([op.qubit for op in self.operators])
        angle_parts = pp_str.split("<")[1][6:].split("/")
        if angle_parts[0] == "pi":
            numerator = 1
        else:
            numerator = int(angle_parts[0].split("p")[0])
        if len(angle_parts) == 1:
            denominator = 1
        else:
            denominator = int(angle_parts[1].split(")")[0])
        self.angle = quilt.angle.Angle(numerator, denominator)

    def is_clifford(self):
        assert self.angle is not None
        return self.angle.is_clifford()

    def get_product_str(self):
        return "".join(str(op) for op in self.operators).strip()

    def get_qubits(self):
        return [op.qubit for op in self.operators]

    def __str__(self):
        s = (
            str(self.id)
            + " "
            + self.get_product_str()
            + " "
            + str(self.angle)
            + " "
            + str(self.num_ys)
            + " "
            + str(self.need_estabilizer)
            + " "
            + str(self.children)
            + " "
            + str(self.parents)
        )
        return s.strip()
