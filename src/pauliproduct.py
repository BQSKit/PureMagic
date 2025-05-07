#!/usr/bin/env -S python -u


class PauliProduct:
    def __init__(self, rng, num_qubits, pauli_product_qubits, start_qubit):
        self.basis_options = ["X", "Z", "Y"]
        # self.basis_options = ["X", "Z"]
        self.operators = [" "] * num_qubits
        self.start_qubit = start_qubit
        self.qubits_used = pauli_product_qubits
        for i in range(start_qubit, start_qubit + pauli_product_qubits):
            self.operators[i] = self.basis_options[int(np.floor(rng.uniform(0, len(self.basis_options))))]

    def __str__(self):
        s = ""
        for i in range(len(self.operators)):
            if self.operators[i] != " ":
                s += str(i) + self.operators[i] + " "
        return s.strip()
