#!/usr/bin/bash

for f in `cat benchmarks`; do
    ../../target/release/compile_cliffordt $f.qasm
done|tee out-compile-cliffordt
