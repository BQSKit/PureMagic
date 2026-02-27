#!/usr/bin/env python3
"""
Script to convert quantum circuit operations from verbose format to compact format.

Input format:
    Rotate -1: XXXX
    Rotate 2: IIZX
    Measure +: IIZX
    Measure -: YZIX

Output format:
    -XXXX<pi/8>
    +__ZX<pi/4>
    +__ZX<M>
    -YZ_X<M>
"""

import sys
import argparse
import re
from pathlib import Path
from typing import Optional


def convert_operation(line):
    """
    Convert a single line from input format to output format.

    Args:
        line (str): Input line in format "Operation sign: PAULI_STRING"

    Returns:
        str: Converted line in compact format, or None if line is invalid
    """
    line = line.strip()
    if not line:
        return None

    # Parse the input line using regex
    # Matches: "Rotate -1:", "Rotate 1:", "Measure +:", "Measure -:"
    pattern = r"^(Rotate|Measure)\s+([+-]?\d*):?\s+([IXYZ]+)$"
    match = re.match(pattern, line)

    if not match:
        raise RuntimeError(f"Could not parse line: {line}")

    operation, sign_part, pauli_string = match.groups()

    # Determine the sign
    if operation == "Rotate":
        if sign_part in ["-1", "-2"]:
            sign = "-"
        elif sign_part in ["1", "2"]:
            sign = "+"
        else:
            raise RuntimeError(f"Unknown rotation sign '{sign_part}' in line: {line}")
    elif operation == "Measure":
        if sign_part == "+":
            sign = "+"
        elif sign_part == "-":
            sign = "-"
        else:
            raise RuntimeError(f"Unknown measurement sign '{sign_part}' in line: {line}")
    else:
        raise RuntimeError(f"Unknown operation '{operation}' in line: {line}")

    # Convert Pauli string: replace 'I' with '_'
    converted_pauli = pauli_string.replace("I", "_")

    # Determine the angle bracket
    if operation == "Rotate":
        if sign_part in ["1", "-1"]:
            gate_type = "T"
        else:
            gate_type = "clifford"
    elif operation == "Measure":
        gate_type = "M"
    else:
        raise RuntimeError(f"Unknown operation type {operation} in line: {line}")

    return (sign, converted_pauli, gate_type)


def get_qubits_and_terms(op_str):
    qubits = []
    terms = []
    for i, op in enumerate(op_str):
        if op != "_":
            qubits.append(i)
            terms.append(op)
    return (qubits, terms)


def get_cx_product(i, lines):
    if len(lines) <= i + 2:
        return None
    (sign, op_str, gate_type) = lines[i]
    (qubits, terms) = get_qubits_and_terms(op_str)
    # if both X and Z in the string, then it is a CNOT
    if not "X" in terms or not "Z" in terms:
        return None
    assert len(qubits) == 2 and sign == "+" and terms[0] != terms[1]
    xpos = qubits[0] if terms[0] == "X" else qubits[1]
    zpos = qubits[0] if terms[0] == "Z" else qubits[1]
    # check and clear next two lines
    for term in ["Z", "X"]:
        i += 1
        if lines[i] is None:
            return None
        (sign, op_str, gate_type) = lines[i]
        (qubits, terms) = get_qubits_and_terms(op_str)
        assert gate_type == "clifford" and len(qubits) == 1 and sign == "-"
        assert terms[0] == term and qubits[0] == (xpos if term == "X" else zpos)
    return f"cx q[{min(xpos,zpos)}], q[{max(xpos, zpos)}];"


def get_t_product(i, lines):
    (_, op_str, gate_type) = lines[i]
    assert gate_type == "T"
    (qubits, terms) = get_qubits_and_terms(op_str)
    assert len(qubits) == 1 and terms[0] == "Z"
    return f"t q[{qubits[0]}];"


def get_h_product(i, lines):
    # check for Hadamard - ZXZ over 3 timesteps
    if len(lines) <= i + 2:
        return None
    (_, op_str, gate_type) = lines[i]
    assert gate_type == "clifford"
    (qubits, terms) = get_qubits_and_terms(op_str)
    assert len(qubits) == 1
    if terms[0] != "Z":
        return None
    # could be Hadamard, check for next two lines
    for j, term in enumerate(["X", "Z"]):
        if lines[i + j + 1] is None:
            return None
        (_, next_op_str, next_gate_type) = lines[i + j + 1]
        (next_qubits, next_terms) = get_qubits_and_terms(next_op_str)
        if next_gate_type != "clifford":
            return None
        if len(next_qubits) != 1 or next_terms[0] != term or next_qubits[0] != qubits[0]:
            return None
    return f"h q[{qubits[0]}];"


def get_m_product(i, lines):
    (_, op_str, gate_type) = lines[i]
    assert gate_type == "M"
    qubits = get_qubits_and_terms(op_str)[0]
    assert len(qubits) == 1
    return f"measure q[{qubits[0]} -> meas[{qubits[0]};"


def get_s_product(i, lines):
    (_, op_str, gate_type) = lines[i]
    assert gate_type == "clifford"
    (qubits, terms) = get_qubits_and_terms(op_str)
    if len(qubits) == 1 and terms[0] == "Z":
        return f"s q[{qubits[0]}];"
    return None


def preprocess(lines):
    skips = 0
    for i in range(len(lines)):
        i = i + skips
        if i + 1 >= len(lines):
            break
        (sign, op_str, gate_type) = lines[i]
        (next_sign, next_op_str, next_gate_type) = lines[i + 1]
        if op_str != next_op_str:
            # no reduction if they don't operate on exactly the same qubits with the same terms
            continue
        if gate_type == "T" and next_gate_type == "T":
            if sign != next_sign:
                # different signs, cancel out the T gates
                # print(f"Cancel {i} {lines[i]} and {lines[i+1]}", file=sys.stderr)
                lines[i] = None
                lines[i + 1] = None
                skips += 1
                continue
            else:
                # same sign, convert to Clifford Z
                # print(f"To clifford {i} {lines[i]} and {lines[i+1]}", file=sys.stderr)
                (qubits, terms) = get_qubits_and_terms(op_str)
                assert len(qubits) == 1 and terms[0] == "Z"
                lines[i] = (sign, op_str, "clifford")
                lines[i + 1] = None
                skips += 1
                continue
    return lines


def get_converted_lines(input_path):
    lines = []
    converted_count = 0
    # Read and convert the file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for _, line in enumerate(f, 1):
                converted_line = convert_operation(line)
                if converted_line is None:
                    continue
                lines.append(converted_line)
                converted_count += 1
    except Exception as e:
        raise RuntimeError(f"Error reading input file: {e}")
    return lines


def convert_file(input_file, output_file=None):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    # Determine output file path
    if output_file is None:
        output_path = input_path.with_suffix(input_path.suffix + ".converted")
    else:
        output_path = Path(output_file)

    lines = get_converted_lines(input_path)

    num_qubits = len(lines[0][1])
    f = open(output_path, "w", encoding="utf-8")
    print("OPENQASM 2.0;", file=f)
    print('include "qelib1.inc";', file=f)
    print(f"qreg q[{num_qubits}];", file=f)
    print("gate identity1 a0", file=f)
    print("{", file=f)
    print("        U(0,0,0) a0;", file=f)
    print("}", file=f)
    print(f"creg c[{num_qubits}];", file=f)

    lines = preprocess(lines)

    num_cliffords = 0
    num_tgates = 0
    num_measurements = 0
    skips = 0
    for i in range(len(lines)):
        i = i + skips
        if i >= len(lines):
            break
        if lines[i] is None:
            continue
        gate_type = lines[i][2]
        if gate_type == "T":
            num_tgates += 1
            print(get_t_product(i, lines), file=f)
        elif gate_type == "M":
            num_measurements += 1
            print(get_m_product(i, lines), file=f)
        elif gate_type == "clifford":
            num_cliffords += 1
            cx_product = get_cx_product(i, lines)
            if cx_product is not None:
                print(cx_product, file=f)
                skips += 2
                continue
            h_product = get_h_product(i, lines)
            if h_product is not None:
                print(h_product, file=f)
                skips += 2
                continue
            s_product = get_s_product(i, lines)
            if s_product is not None:
                print(s_product, file=f)
            else:
                raise RuntimeError(f"Could not process line {lines[i]} on line {i}\n{lines[i - 1]}")
        else:
            raise RuntimeError(f"Unknown gate type {gate_type}")

    print(f"Conversion complete")
    print(f"  Number of qubits: {num_qubits}")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Lines processed: {len(lines)}")
    print(f"  Output gates: {len(lines) - skips}")
    print(f"  Number of T gates: {num_tgates}")
    print(f"  Number of Cliffords: {num_cliffords}")
    print(f"  Number of mesaurements: {num_measurements}")


def main():
    """Main function to handle command line arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert quantum circuit operations from verbose to compact format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.txt
  %(prog)s input.txt -o output.txt
  %(prog)s input.txt --output converted_circuit.txt

Input format:
  Rotate -1: XXXX
  Rotate 1: IIZX
  Measure +: IIZX
  Measure -: YZIX

Output format:
  -XXXX<pi/8>
  +__ZX<pi/8>
  +__ZX<M>
  -YZ_X<M>
        """,
    )

    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("-o", "--output", help="Output file path (default: input_file.converted)")
    args = parser.parse_args()

    try:
        convert_file(args.input_file, args.output)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
