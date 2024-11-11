""" Network """

from node import Node
import random
from position import Euclidean2D
from matplotlib import pyplot as plt


class Network:
    """Network"""

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    @classmethod
    def randomize(cls, num_nodes: int, grid_size: int):
        nodes: list[Node] = []
        for i in range(num_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            node = Node(i, pos=Euclidean2D(x, y))
            nodes.append(node)
        return cls(nodes)

    def show_network(self) -> None:
        """Plots the 2D network in a grid"""
        x = []
        y = []
        for node in self.nodes:
            pos: Euclidean2D = node.pos
            x.append(pos.x)
            y.append(pos.y)

        plt.figure(figsize=(10, 8))
        plt.scatter(x, y)
        plt.grid()
        plt.show()
