"""Regex patterns shared by the scripts that parse puremagic's stdout log:
plot_puremagic.py, scheduling_table.py, and circuit_table.py.

Each of those scripts drives its own state machine over these patterns --
when a "run" starts/ends and which fields it cares about differs
deliberately per script (plot_puremagic.py wants every field for every run;
circuit_table.py only wants Layers; scheduling_table.py flushes on a new
magic_state_lambda line) -- so only the patterns that were already
byte-for-byte identical (or trivially equivalent) across two or more of
those scripts live here, not the surrounding parsing logic.
"""

import re

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# "magic_state_lambda: <value>," (from the Args debug-dump block)
MAGIC_STATE_LAMBDA = re.compile(r"magic_state_lambda:\s*([0-9.eE+\-]+),?")

# "Layers:  <n>" (from Circuit::print_statistics's "Circuit statistics:" block)
LAYERS = re.compile(r"Layers:\s+(\d+)")

# "Loaded circuit with <n> products and <q> qubits"
LOADED_CIRCUIT = re.compile(r"Loaded circuit with \d+ products and (\d+) qubits")

# "Scheduled products written to <name>.schedule"
WROTE_SCHEDULE = re.compile(r"Scheduled products written to (.+?)\.schedule")
