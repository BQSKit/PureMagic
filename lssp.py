#!/usr/bin/env -S python -u

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

def build_topo(num_cols, num_rows):
    topo_graph = nx.Graph()
    node_pos = {}
    for col in range(num_cols):
        if col % 2 == 0:
            node_label = "m" + str(col) + "-0"
            node_pos[node_label] = [col, num_rows - 1]
            topo_graph.add_edge(node_label, "b" + str(col) + "-1")
            for row in range(1, num_rows - 2):
                node_label = "b" + str(col) + "-" + str(row)
                node_pos[node_label] = [col, num_rows - 1 - row]
                topo_graph.add_edge(node_label, "b" + str(col) + "-" + str(row + 1))
            prev_node_label = "b" + str(col) + "-" + str(num_rows - 2)
            node_pos[prev_node_label] = [col, 1]
            node_label = "m" + str(col) + "-" + str(num_rows - 1)
            node_pos[node_label] = [col, 0]
            topo_graph.add_edge(node_label, prev_node_label)

    #print(topo_graph.nodes)
    #print(topo_graph.edges)
    plt.rc("figure", figsize=[num_cols, num_rows])
    nx.draw_networkx(topo_graph, pos=node_pos, node_color="lightgrey", node_size=1200)
    plt.show()

if __name__ == "__main__":
    build_topo(15, 9)

