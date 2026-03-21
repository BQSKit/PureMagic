#!/usr/bin/bash

# improvement from PureMagic over Bus routing for max weight 1
#./plot_puremagic.py -x circuit -y scheduling_efficiency --hline \
#    -f ../results/max-weight-1/puremagic/out:PureMagic,../results/max-weight-1/bus/out:Bus -o efficiency_puremagic_bus_ct.png

# improvement from PureMagic over Bus, for both max weight 0 and 1
./plot_puremagic.py -x circuit -y scheduling_efficiency --hline \
    -f ../results/max-weight-1/puremagic/out:PureMagic,../results/max-weight-1/bus/out:"Bus (max weight 1)"  \
    -f ../results/max-weight-0/puremagic/out:PureMagic,../results/max-weight-0/bus/out:"Bus" -o efficiency_puremagic_bus_w0_w1.png

# improvement from max weight 1 over max weight 0 for PureMagic
./plot_puremagic.py -x circuit -y timesteps --hline \
    -f ../results/max-weight-0/puremagic/out:"No weight limit",../results/max-weight-1/puremagic/out:"Max weight 1" -o timesteps_puremagic_w0_v1.png

# relation between scheduling efficiency and parallelism for PureMagic with max weight 0 and 1
#./plot_puremagic.py -x parallelism -y scheduling_efficiency -s heisenberg --label-data-qubits \
#    -f ../results/max-weight-1/puremagic/out:"Heisenberg Clifford+T" -o efficiency_v_parallelism_puremagic_ct_pb.png

# relation between scheduling efficiency and ancilla qubits for PureMagic with max weight 1 and varying topologies
./plot_puremagic.py -x ancilla_qubits -y scheduling_efficiency --lines-with-markers \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N25:heisenberg_N25 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N100:heisenberg_N100 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N225:heisenberg_N225 -o scheduling_efficiency_v_qubits_puremagic.png

# relation between parallel efficiency and ancilla qubits for PureMagic with max weight 1 for varying topologies
./plot_puremagic.py -x ancilla_qubits -y parallel_efficiency --lines-with-markers \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N25:heisenberg_N25 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N100:heisenberg_N100 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N225:heisenberg_N225 -o parallel_efficiency_v_qubits_puremagic.png

# relation between scheduling/parallel efficiency and ancilla qubits for PureMagic with max weight 1 for varying topologies
#./plot_puremagic.py -x ancilla_qubits -y parallel_efficiency,scheduling_efficiency --lines-with-markers \
#    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N64:heisenberg_N64 \
#    -o efficiency_v_qubits_puremagic_ct.png

# relation between scheduling efficiency and expected cultivation times for PureMagic and Bus with max weight 1
./plot_puremagic.py -x cultivation -y scheduling_efficiency --lines-with-markers \
    -f ../results/max-weight-1/puremagic-vary-cultivation/out-square_heisenberg_N25:PureMagic,../results/max-weight-1/bus-vary-cultivation/out-square_heisenberg_N25:"Bus heisenberg_N25" \
    -f ../results/max-weight-1/puremagic-vary-cultivation/out-square_heisenberg_N100:PureMagic,../results/max-weight-1/bus-vary-cultivation/out-square_heisenberg_N100:"Bus heisenberg_N100" \
    -f ../results/max-weight-1/puremagic-vary-cultivation/out-square_heisenberg_N225:PureMagic,../results/max-weight-1/bus-vary-cultivation/out-square_heisenberg_N225:"Bus heisenberg_N225" \
    -o efficiency_v_cultivation_puremagic_bus_ct.png

# relation between transpilation max weight and timesteps and number of cliffords for PureMagic
#./plot_puremagic.py -x weight -y timesteps,cliffords --lines -f ../results/max-weight-vary/puremagic/out-square_heisenberg_N64:heisenberg_N64 -o timesteps_v_weight.png
./plot_puremagic.py -x weight -y timesteps --lines -f ../results/max-weight-vary/puremagic/out-square_heisenberg_N64:heisenberg_N64 -o timesteps_v_weight.png

# relation beween computation timing and parallelism for PureMagic with max weight 1
./plot_puremagic.py -x parallelism -y timing -f ../results/max-weight-1/puremagic/out:PureMagic -o timing_v_parallelism.png

# ratio of timesteps for given data qubits comparing bus to puremagic -
#./plot_puremagic.py -x data_qubits -y timesteps --hline \
#    -f ../results/max-weight-1/bus/out:Bus,../results/max-weight-1/puremagic/out:PureMagic -o timesteps_v_data_qubits_puremagic_bus.png

# for heisenberg, showin relation of data qubits (circuit width) to both volume and timesteps for bus v puremagic
./plot_puremagic.py -x data_qubits -y timesteps,volume --hline \
    -f ../results/max-weight-1/bus/out:Bus,../results/max-weight-1/puremagic/out:PureMagic --ylim 0,3 --y2lim 0,3 --lines-with-markers -s heisenberg -o timesteps_v_data_qubits_puremagic_bus.png
