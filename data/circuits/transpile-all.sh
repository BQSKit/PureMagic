#!/usr/bin/bash

trans=../../target/release/transpile

for m in "-m 0" "-m 1" "--auto"; do
    for f in `cat benchmarks`; do
        $trans -i $f.cliffordt.qasm --defer_trailing $m
    done | tee out-$m
done

pushd heisenberg
for f in `ls *.cliffordt.qasm`; do
    ../$trans -i $f --auto
done | tee out-transauto
popd 

pushd qft
for f in `ls *.cliffordt.qasm`; do
    ../$trans -i $f --auto
done | tee out-transauto
popd 

