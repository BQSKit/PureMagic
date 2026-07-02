#!/usr/bin/bash

for f in `ls ../../all_compiled/*.cliffordt.qasm`; do
    ../../../target/release/transpile -i $f -m 1
done
