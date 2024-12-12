""" Network """

import numpy as np
import pandas as pd
from basic_types import NodeID, Ping
from node import Node
import random
from position import CoordinateSystemPoint, Euclidean2D
from matplotlib import pyplot as plt

class Network:
    """Network"""

    def __init__(self, nodes: list[Node], pings: dict[NodeID, dict[NodeID,Ping]] = None):
        self.nodes = nodes
        self.pings = pings

    @classmethod
    def randomize(cls, num_nodes: int, grid_size: int):
        nodes: list[Node] = []
        for i in range(num_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            node = Node(i, pos=Euclidean2D(x, y))
            nodes.append(node)
        pings = {node1.node_id: {node2.node_id: Ping(float("inf"), 0) for node2 in nodes} for node1 in nodes}
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                csp: CoordinateSystemPoint = (nodes[i].pos - nodes[j].pos)
                pings[i][j] = Ping(csp.norm(), 0.1)
        return cls(nodes, pings)

    @classmethod
    def from_dicts(cls, data: dict[int, dict[int, tuple[float, float]]], coords: dict[int, tuple[float, float]], fraction: float = 1):
        nodes: list[Node] = []

        node_ids: set[NodeID] = set()

        coords_keys = list(coords.keys())
        if fraction < 1:
            random.shuffle(coords_keys)
            num_keys = int(len(coords_keys) * fraction)
            coords_keys = coords_keys[:num_keys]

        for node_id in coords_keys:
            nodes.append(Node(node_id, pos=Euclidean2D(*coords[node_id])))
            node_ids.add(node_id)

        pings = {node1.node_id: {node2.node_id: Ping(float("inf"), 0) for node2 in nodes} for node1 in nodes}

        for source in data:
            if source not in node_ids:
                continue
            pings[source][source] = Ping(0, 0)

            for destination in data[source]:
                if destination not in node_ids:
                    continue

                ping_avg, ping_std_dev = data[source][destination]
                pings[source][destination] = Ping(ping_avg, ping_std_dev)

        return cls(nodes, pings)

    def get_base_delay(self, node_1: NodeID, node_2: NodeID) -> float:
        """Returns the latency between two nodes"""

        return self.pings[node_1][node_2].base

    def get_delay(self, node_1: NodeID, node_2: NodeID) -> float:
        """Returns the latency between two nodes"""

        ping = self.pings[node_1][node_2]
        return ping.base + abs(np.random.normal(0, ping.std_dev))

    def show_network(self) -> None:
        """Plots the 2D network in a grid"""
        x = []
        y = []
        for node in self.nodes:
            pos: Euclidean2D = node.pos
            x.append(pos.x)
            y.append(pos.y)

        plt.figure(figsize=(10, 8))
        plt.scatter(y, x)

        # for node1 in self.nodes:
        #     for node2 in self.nodes:
        #         if node1.node_id == node2.node_id:
        #             continue
        #         pos1: Euclidean2D = node1.pos
        #         pos2: Euclidean2D = node2.pos
        #         plt.plot([pos1.x, pos2.x],[pos1.y, pos2.y], alpha = 0.8)
        #         plt.text(x = (pos1.x + pos2.x)/2, y = (pos1.y + pos2.y)/2, s = f"{self.pings[node1.node_id][node2.node_id].base:.2f}")

        plt.grid()
        plt.savefig("network.png", dpi = 300)
        plt.show()



def servers_csv_to_dict(filename: str) -> dict[int, tuple[float, float]]:
    df = pd.read_csv(filename)
    node_coordinates = df.set_index('id')[['latitude', 'longitude']].T.to_dict()

    keys = list(node_coordinates.keys())
    # random.shuffle(keys)
    # keys = keys[:int(len(keys)/3)]

    node_coordinates = {k: (float(node_coordinates[k]['latitude']), float(node_coordinates[k]['longitude'])) for k in keys}
    return node_coordinates

def pings_csv_to_dict(filename: str) -> dict[str, dict[str, list[str]]]:
    data = {}
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            parts = [part.strip() for part in line.split(',')]
            source = int(parts[0].replace("\"", ""))
            destination = int(parts[1].replace("\"", ""))
            ping_avg = float(parts[4].replace("\"", ""))
            ping_std_dev = float(parts[6].replace("\"", ""))
            if source not in data:
                data[source] = {}
            data[source][destination] = (ping_avg, ping_std_dev)
    return data
