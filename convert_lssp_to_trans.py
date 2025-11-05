#!/usr/bin/env python3
"""
Script to convert quantum circuit operations from verbose format to compact format.

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
"""

import sys
import argparse
import re
from pathlib import Path


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
        print(f"Warning: Could not parse line: {line}", file=sys.stderr)
        return None

    operation, sign_part, pauli_string = match.groups()

    # Determine the sign
    if operation == "Rotate":
        if sign_part == "-1":
            sign = "-"
        elif sign_part == "1" or sign_part == "+1":
            sign = "+"
        else:
            print(f"Warning: Unknown rotation sign '{sign_part}' in line: {line}", file=sys.stderr)
            return None
    elif operation == "Measure":
        if sign_part == "+":
            sign = "+"
        elif sign_part == "-":
            sign = "-"
        else:
            print(
                f"Warning: Unknown measurement sign '{sign_part}' in line: {line}", file=sys.stderr
            )
            return None

    # Convert Pauli string: replace 'I' with '_'
    converted_pauli = pauli_string.replace("I", "_")

    # Determine the angle bracket
    if operation == "Rotate":
        angle = "<pi/8>"
    elif operation == "Measure":
        angle = "<M>"

    return f"{sign}{converted_pauli}{angle}"


def convert_file(input_file, output_file=None):
    """
    Convert an entire file from input format to output format.

    Args:
        input_file (str or Path): Path to input file
        output_file (str or Path, optional): Path to output file.
                                           If None, creates output file with .converted suffix
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine output file path
    if output_file is None:
        output_path = input_path.with_suffix(input_path.suffix + ".converted")
    else:
        output_path = Path(output_file)

    converted_lines = []
    total_lines = 0
    converted_count = 0

    # Read and convert the file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                converted_line = convert_operation(line)

                if converted_line is not None:
                    converted_lines.append(converted_line)
                    converted_count += 1
                elif line.strip():  # Only warn about non-empty lines
                    print(f"Warning: Skipped line {line_num}: {line.strip()}", file=sys.stderr)

    except Exception as e:
        raise RuntimeError(f"Error reading input file: {e}")

    # Write the converted file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in converted_lines:
                f.write(line + "\n")
    except Exception as e:
        raise RuntimeError(f"Error writing output file: {e}")

    print(f"Conversion complete!")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Lines processed: {total_lines}")
    print(f"  Lines converted: {converted_count}")
    print(f"  Lines skipped: {total_lines - converted_count}")


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print converted lines to stdout without writing to file",
    )

    args = parser.parse_args()

    try:
        if args.dry_run:
            # Dry run mode: print to stdout
            input_path = Path(args.input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")

            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    converted_line = convert_operation(line)
                    if converted_line is not None:
                        print(converted_line)
        else:
            # Normal mode: write to file
            convert_file(args.input_file, args.output)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
