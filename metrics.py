""" Metrics """

import numpy as np
from basic_types import NodeID
from dataclasses import dataclass
from node import Node
from network import Network
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Metric:
    """Any metric"""
    values: list

    def mean(self):
        """Returns the mean"""
        return np.mean(self.values)
    def median(self):
        """Returns the median"""
        return np.median(self.values)
    def max(self):
        """Returns the max"""
        return max(self.values)
    def min(self):
        """Returns the min"""
        return min(self.values)

    def __repr__(self):
        return f"Mean: {self.mean()} | Median: {self.median()} | Max: {self.max()} | Min: {self.min()}"

    def plot_histogram(self, xlabel: str, save: bool, fname: str = "histogram.png", bins: int = 10, color: str = 'blue'):
        """ Plots a histogram"""
        values = np.array(self.values)
        counts, bin_edges = np.histogram(values, bins=bins)
        percentages = 100 * counts / len(values)

        plt.figure(figsize=(8, 6))
        plt.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), color=color, edgecolor='black', align='edge')

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Frequency (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        if save:
            plt.savefig(fname, dpi = 300)
        plt.show()

class Metrics:
    """ Metrics """

    def __init__(self, network: Network, source: Node, arrival_times: dict[NodeID, float]):
        self.source = source
        self.arrival_times = arrival_times
        self.network = network

    def get_stretch(self) -> Metric:
        """Computes the stretch"""
        print("stretch debug")
        stretches: list[float] = []
        for node, time in self.arrival_times.items():
            if node == self.source.node_id:
                continue
            stretches.append(time / self.network.get_base_delay(self.source.node_id, node))
            print(time, self.network.get_base_delay(self.source.node_id, node))
        return Metric(stretches)
