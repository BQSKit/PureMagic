#!/usr/bin/bash

# improvement from PureMagic over Bus routing for max weight 1
#./plot_puremagic.py -x circuit -y scheduling_efficiency --hline \
#    -f ../results/max-weight-1/puremagic/out:PureMagic,../results/max-weight-1/bus/out:Bus -o efficiency_puremagic_bus_ct.png

# improvement from PureMagic over Bus, for both max weight 0 and 1
./plot_puremagic.py -x circuit -y scheduling_efficiency --hline \
    -c benchmarks \
    -f ../results/max-weight-1/puremagic/out:PureMagic,../results/max-weight-1/bus/out:"Bus (max weight 1)"  \
    -f ../results/max-weight-0/puremagic/out:PureMagic,../results/max-weight-0/bus/out:"Bus" -o efficiency_puremagic_bus_w0_w1.png
./plot_puremagic.py -x circuit -y volume \
    -c benchmarks \
    -f ../results/max-weight-1/puremagic/out,../results/max-weight-1/bus/out:"Max weight 1"  \
    -f ../results/max-weight-0/puremagic/out,../results/max-weight-0/bus/out:"No max weight" -o volume_puremagic_bus_w0_w1.png

# improvement from max weight 1 over max weight 0 for PureMagic
./plot_puremagic.py -x circuit -y lcycles --hline \
    -c benchmarks \
    -f ../results/max-weight-0/puremagic/out:"No weight limit",../results/max-weight-1/puremagic/out:"Max weight 1" -o lcycles_puremagic_w0_v1.png

# relation between scheduling efficiency and parallelism for PureMagic with max weight 0 and 1
#./plot_puremagic.py -x parallelism -y scheduling_efficiency -s heisenberg --label-data-qubits \
#    -c benchmarks \
#    -f ../results/max-weight-1/puremagic/out:"Heisenberg Clifford+T" -o efficiency_v_parallelism_puremagic_ct_pb.png

# relation between scheduling efficiency and ancilla qubits for PureMagic with max weight 1 and varying topologies
./plot_puremagic.py -x ancilla_qubits -y scheduling_efficiency --lines-with-markers \
    -c benchmarks \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N25:heisenberg_N25 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N100:heisenberg_N100 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N225:heisenberg_N225 \
    --ylim 0,0.7 -o scheduling_efficiency_v_qubits_puremagic.png

# relation between parallel efficiency and ancilla qubits for PureMagic with max weight 1 for varying topologies
./plot_puremagic.py -x ancilla_qubits -y parallel_efficiency --lines-with-markers \
    -c benchmarks \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N25:heisenberg_N25 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N100:heisenberg_N100 \
    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N225:heisenberg_N225 \
    --ylim 0,1 -o parallel_efficiency_v_qubits_puremagic.png

# relation between scheduling/parallel efficiency and ancilla qubits for PureMagic with max weight 1 for varying topologies
#./plot_puremagic.py -x ancilla_qubits -y parallel_efficiency,scheduling_efficiency --lines-with-markers \
#    -c benchmarks \
#    -f ../results/max-weight-1/puremagic-vary-topo/out-square_heisenberg_N64:heisenberg_N64 \
#    -o efficiency_v_qubits_puremagic_ct.png

# relation between scheduling efficiency and expected cultivation times for PureMagic and Bus with max weight 1
./plot_puremagic.py -x cultivation -y scheduling_efficiency --lines-with-markers \
    -c benchmarks \
    -f ../results/max-weight-1/puremagic-vary-cultivation/out-square_heisenberg_N25:PureMagic,../results/max-weight-1/bus-vary-cultivation/out-square_heisenberg_N25:"Bus heisenberg_N25" \
    -f ../results/max-weight-1/puremagic-vary-cultivation/out-square_heisenberg_N100:PureMagic,../results/max-weight-1/bus-vary-cultivation/out-square_heisenberg_N100:"Bus heisenberg_N100" \
    -f ../results/max-weight-1/puremagic-vary-cultivation/out-square_heisenberg_N225:PureMagic,../results/max-weight-1/bus-vary-cultivation/out-square_heisenberg_N225:"Bus heisenberg_N225" \
    --ylim 0,5 -o efficiency_v_cultivation_puremagic_bus_ct.png --hline

# relation between transpilation max weight and lcycles and number of cliffords for PureMagic
./plot_puremagic.py -x weight -y lcycles,cliffords --lines -c benchmarks -f ../results/max-weight-vary/puremagic/out-square_heisenberg_N64:heisenberg_N64 -o lcycles_v_weight.png
#./plot_puremagic.py -x weight -y lcycles --lines \
#    -c benchmarks \
#    -f ../results/max-weight-vary/puremagic/out-square_heisenberg_N25:heisenberg_N25 \
#    -f ../results/max-weight-vary/puremagic/out-square_heisenberg_N100:heisenberg_N100 \
#    -f ../results/max-weight-vary/puremagic/out-square_heisenberg_N225:heisenberg_N225 \
#    -o lcycles_v_weight.png

# relation beween computation timing and parallelism for PureMagic with max weight 1
./plot_puremagic.py -x parallelism -y timing -c benchmarks -f ../results/max-weight-1/puremagic/out:PureMagic -o timing_v_parallelism.png --ylim 0,14

# ratio of lcycles for given data qubits comparing bus to puremagic -
#./plot_puremagic.py -x data_qubits -y lcycles --hline \
#    -c benchmarks \
#    -f ../results/max-weight-1/bus/out:Bus,../results/max-weight-1/puremagic/out:PureMagic -o lcycles_v_data_qubits_puremagic_bus.png

# for heisenberg, show relation of data qubits (circuit width) to both volume and lcycles for bus v puremagic
./plot_puremagic.py -x data_qubits -y lcycles/volume --hline --ylabel Ratio \
    -c benchmarks \
    -f ../results/max-weight-1/bus/out:Bus,../results/max-weight-1/puremagic/out:PureMagic \
    --ylabel Ratio --ylim 0,3 --stackedbar -s heisenberg -o lcycles_v_data_qubits_puremagic_bus_barchart.png

# for heisenberg, show relation of data qubits (circuit width) to both volume and lcycles for bus v puremagic
./plot_puremagic.py -x data_qubits -y lcycles/volume --hline --ylabel Ratio \
    -c benchmarks \
    -f ../results/max-weight-1/bus/out:Bus,../results/max-weight-1/puremagic/out:PureMagic \
    --ylim 0,3 --lines-with-markers -s heisenberg -o lcycles_v_data_qubits_puremagic_bus.png

# impact of incorporating T failures
./plot_puremagic.py -x circuit -y lcycles --hline --ylabel Ratio \
    -c benchmarks \
    -f ../results/max-weight-0/puremagic/out:PureMagic-w0,../results/no-t-fail/max-weight-0/puremagic/out:"No Fail" \
    -f ../results/max-weight-1/puremagic/out:PureMagic-w1,../results/no-t-fail/max-weight-1/puremagic/out:"No Fail" \
    -o efficiency_w0_w1_pm_no_t_fail.png


