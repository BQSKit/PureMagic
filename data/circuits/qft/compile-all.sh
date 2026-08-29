#!/usr/bin/bash

for f in `ls *.qasm|grep -v clifford`; do
    ../../../target/release/compile_cliffordt $f 
done
