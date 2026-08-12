#!/usr/bin/bash

trans=../../target/release/transpile

for m in {2..20}; do
    for f in `cat benchmarks`; do
        $trans -i $f.cliffordt.qasm --defer_trailing -m $m
    done | tee out-$m
done

# for m in "-m 0" "-m 1" "--auto"; do
#     for f in `cat benchmarks`; do
#         $trans -i $f.cliffordt.qasm --defer_trailing $m
#     done | tee out-$m
# done

# for b in qv_N064_12345 qft_n160 square_heisenberg_N225; do
#     for m in {0..10} {15..50..5} {60..100..10} {120..200..20}; do
#         $trans -i $b.cliffordt.qasm -m $m --defer_trailing
#     done | tee out-$b-trans-vary
# done

# pushd heisenberg
# for f in `ls *.cliffordt.qasm`; do
#     ../$trans -i $f --auto
# done | tee out-transauto
# popd 

# pushd qft
# for f in `ls *.cliffordt.qasm`; do
#     ../$trans -i $f --auto
# done | tee out-transauto
# popd 

# pushd small
# for f in `ls *.cliffordt.qasm`; do
#     ../$trans -i $f --auto
# done | tee out-transauto
# popd 

