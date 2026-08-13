"""Shared parsing for the "LSS" verbose rotation/measurement format used by
convert_lss_to_qasm.py and convert_lss_to_trans.py.

Input format (one operation per line):
    Rotate -1: XXXX
    Rotate 1: IIZX
    Measure +: IIZX
    Measure -: YZIX
"""

import re

Op = tuple[str, str, str]

_PATTERN = re.compile(r"^(Rotate|Measure)\s+([+-]?\d*):?\s+([IXYZ]+)$")

CLI_EPILOG = """
Examples:
  %(prog)s input.txt
  %(prog)s input.txt -o output.txt
  %(prog)s input.txt --output converted_circuit.txt

Input format:
  Rotate -1: XXXX
  Rotate 1: IIZX
  Measure +: IIZX
  Measure -: YZIX
"""


def parse_operation(line: str) -> Op | None:
    """Parse a line like "Rotate -1: XXXX" into (sign, pauli_string, gate_type), or
    None if the line is blank.

    gate_type is "T" for a pi/8 rotation, "clifford" for a pi/4 rotation, or "M" for a
    measurement. Raises RuntimeError on any non-blank line that doesn't parse, or that
    parses with an operation/sign this format doesn't recognize.
    """
    line = line.strip()
    if not line:
        return None

    match = _PATTERN.match(line)
    if not match:
        raise RuntimeError(f"Could not parse line: {line}")

    operation, sign_part, pauli_string = match.groups()

    if operation == "Rotate":
        if sign_part in ("-1", "-2"):
            sign = "-"
        elif sign_part in ("1", "2"):
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

    converted_pauli = pauli_string.replace("I", "_")
    gate_type = "M" if operation == "Measure" else ("T" if sign_part in ("1", "-1") else "clifford")

    return (sign, converted_pauli, gate_type)
