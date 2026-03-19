#!/usr/bin/bash

# improvement from PureMagic over Bus routing for Clifford+T
#./plot_puremagic.py -x circuit -y scheduling_efficiency --hline \
#    -f ../results/cliffordt/puremagic/out:PureMagic,../results/cliffordt/bus/out:Bus -o efficiency_puremagic_bus_ct.png

# improvement from PureMagic over Bus, for both Clifford+T and Pauli Basis
./plot_puremagic.py -x circuit -y scheduling_efficiency --hline \
    -f ../results/cliffordt/puremagic/out:PureMagic,../results/cliffordt/bus/out:"Bus Clifford+T"  \
    -f ../results/paulibasis/puremagic/out:PureMagic,../results/paulibasis/bus/out:"Bus Pauli Basis" -o efficiency_puremagic_bus_ct_pb.png

# improvement from Clifford+T over Pauli Basis for PureMagic
./plot_puremagic.py -x circuit -y timesteps --hline \
    -f ../results/paulibasis/puremagic/out:"Pauli Basis",../results/cliffordt/puremagic/out:"Clifford+T" -o timesteps_puremagic_cd_pb.png

# improvement of PureMagic over Bus for Clifford+T, also including topologies with the same ancilla counts for both PureMagic and Bus
#./plot_puremagic.py -x circuit -y scheduling_efficiency --hline -s heisenberg \
#    -f ../results/cliffordt/puremagic/out:PureMagic,../results/cliffordt/bus/out:Bus  \
#    -f ../results/cliffordt/puremagic-bus-size/out:PureMagic,../results/cliffordt/bus/out:"Bus Same Size" -o efficiency_puremagic_bus_ct_bus_size.png

# relation between scheduling efficiency and parallelism for PureMagic with Clifford+T and Pauli Basis
./plot_puremagic.py -x parallelism -y scheduling_efficiency -s heisenberg --label-data-qubits \
    -f ../results/cliffordt/puremagic/out:"Heisenberg Clifford+T" -o efficiency_v_parallelism_puremagic_ct_pb.png
#    -f ../results/cliffordt/puremagic/out:"Heisenberg Clifford+T" -f ../results/paulibasis/puremagic/out:"Heisenberg Pauli Basis" -o efficiency_v_parallelism_puremagic_ct_pb.png

# relation between scheduling efficiency and ancilla qubits for PureMagic with Clifford+T and varying topologies
#./plot_puremagic.py -x ancilla_qubits -y scheduling_efficiency --lines-with-markers \
#    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N25:heisenberg_N25 \
#    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N100:heisenberg_N100 \
#    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N225:heisenberg_N225 -o scheduling_efficiency_v_qubits_puremagic_ct.png

# relation between parallel efficiency and ancilla qubits for PureMagic with Clifford+T for varying topologies
#./plot_puremagic.py -x ancilla_qubits -y parallel_efficiency --lines-with-markers \
#    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N25:heisenberg_N25 \
#    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N100:heisenberg_N100 \
#    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N225:heisenberg_N225 -o parallel_efficiency_v_qubits_puremagic_ct.png

# relation between scheduling/parallel efficiency and ancilla qubits for PureMagic with Clifford+T for varying topologies
./plot_puremagic.py -x ancilla_qubits -y parallel_efficiency,scheduling_efficiency --lines-with-markers \
    -f ../results/cliffordt/puremagic-vary-topo/out-square_heisenberg_N64:heisenberg_N64 \
    -o efficiency_v_qubits_puremagic_ct.png

# relation between scheduling efficiency and expected cultivation times for PureMagic and Bus with Clifford+T
./plot_puremagic.py -x cultivation -y scheduling_efficiency --lines-with-markers \
    -f ../results/cliffordt/puremagic-vary-cultivation/out-square_heisenberg_N25:PureMagic,../results/cliffordt/bus-vary-cultivation/out-square_heisenberg_N25:"Bus heisenberg_N25" \
    -f ../results/cliffordt/puremagic-vary-cultivation/out-square_heisenberg_N100:PureMagic,../results/cliffordt/bus-vary-cultivation/out-square_heisenberg_N100:"Bus heisenberg_N100" \
    -f ../results/cliffordt/puremagic-vary-cultivation/out-square_heisenberg_N225:PureMagic,../results/cliffordt/bus-vary-cultivation/out-square_heisenberg_N225:"Bus heisenberg_N225" \
    -o efficiency_v_cultivation_puremagic_bus_ct.png

# relation between transpilation max weight and timesteps and number of cliffords for PureMagic
./plot_puremagic.py -x weight -y timesteps,cliffords --lines -f ../results/vary-weight/puremagic/out-square_heisenberg_N64:heisenberg_N64 -o timesteps_v_weight.png

# relation beween computation timing and parallelism for PureMagic with Clifford+T
./plot_puremagic.py -x parallelism -y timing -f ../results/cliffordt/puremagic/out:PureMagic -o timing_v_parallelism.png

