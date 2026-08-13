#!/usr/bin/env python3
"""
Convert quantum circuit operations from the verbose "LSS" format (see
lss_common.py) directly to the `.trans` Pauli-product format: each Rotate
line becomes a `<pi/8>`- or `<pi/4>`-tagged rotation and each Measure line
becomes an `<M>`-tagged line.

Input format:
    Rotate -1: XXXX
    Rotate 1: IIZX
    Measure +: IIZX
    Measure -: YZIX
"""

import sys
import argparse
from pathlib import Path

from lss_common import CLI_EPILOG, parse_operation


def convert_operation(line):
    """Convert a single line from LSS format to the `.trans`-style compact format.

    Returns the converted line, or None if the line is blank.
    """
    parsed = parse_operation(line)
    if parsed is None:
        return None
    sign, converted_pauli, gate_type = parsed
    angle = {"T": "<pi/8>", "clifford": "<pi/4>", "M": "<M>"}[gate_type]
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

    if output_file is None:
        output_path = input_path.with_suffix(input_path.suffix + ".converted")
    else:
        output_path = Path(output_file)

    converted_lines = []
    total_lines = 0
    converted_count = 0

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                converted_line = convert_operation(line)

                if converted_line is not None:
                    converted_lines.append(converted_line)
                    converted_count += 1
                # A blank line converts to None; anything else that fails to
                # parse raises (see lss_common.parse_operation) rather than
                # being silently skipped.

    except Exception as e:
        raise RuntimeError(f"Error reading input file: {e}")

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
        description="Convert quantum circuit operations from verbose LSS format to .trans format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG + """
Output format:
  -XXXX<pi/8>
  +__ZX<pi/4>
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
            input_path = Path(args.input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")

            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    converted_line = convert_operation(line)
                    if converted_line is not None:
                        print(converted_line)
        else:
            convert_file(args.input_file, args.output)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
