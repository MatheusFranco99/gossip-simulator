"""Gossip Algorithm"""


import abc
import random

import numpy as np
from basic_types import NodeID
from clustering import create_cluster_nodes
from network import Network
from node import Node
from position import CoordinateSystemPoint
from probability import select_from_group, select_samples_from_group_without_replacement


class GossipAlgorithm(abc.ABC):
    """GossipAlgorithm"""

    def __init__(self, network: Network):
        self.network = network
        self.node_ids: list[NodeID] = [node.node_id for node in self.network.nodes]

    @abc.abstractmethod
    def select_targets(self, node_id: NodeID) -> list[NodeID]:
        """Given that [node_id] just received a message, returns a list of nodes for it to propagate the message """

class RandomWalk(GossipAlgorithm):
    """RandomWalk"""

    def __init__(self, network):
        super().__init__(network)

    def select_targets(self, node_id: NodeID) -> list[NodeID]:
        return [np.random.choice(self.node_ids)]

class CobraWalk(GossipAlgorithm):
    """CobraWalk"""

    def __init__(self, network, rho: float):
        super().__init__(network)
        self.rho = rho

    def get_random_target(self) -> NodeID:
        """Returns a random Node ID"""
        return np.random.choice(self.node_ids)

    def select_targets(self, node_id: NodeID) -> list[NodeID]:
        target1 = self.get_random_target()

        random_value = random.random()
        cobra_partition = random_value <= self.rho
        if cobra_partition:
            target2 = self.get_random_target()
            if target1 != target2:
                return [target1, target2]

        return [target1]

class HierarchialGossip(GossipAlgorithm):
    """HierarchialGossip"""

    def __init__(self, network, fanout_intra: int, fanout_inter: int, num_clusters: int):
        super().__init__(network)
        self.clusters, self.node_cluster = create_cluster_nodes(self.network, n_clusters = num_clusters)
        self.fanout_intra = fanout_intra
        self.fanout_inter = fanout_inter

    def select_targets(self, node_id: NodeID) -> list[NodeID]:

        node_cluster_id = self.node_cluster[node_id]

        intra_targets = select_samples_from_group_without_replacement(self.clusters[node_cluster_id], k = self.fanout_intra)

        outside_nodes = [node_id for node_id in self.node_ids if node_id not in self.clusters[node_cluster_id]]
        inter_targets = select_samples_from_group_without_replacement(outside_nodes, k = self.fanout_inter)
        return intra_targets + inter_targets

class SpatialGossip(GossipAlgorithm):
    """SpatialGossip"""

    def __init__(self, network, dimension: int, rho: float):
        super().__init__(network)
        self.dimension: int = dimension
        self.rho: float = rho

        # Pre-compute probabilities
        self.spatial_gossip_vectors: dict[NodeID, dict[NodeID, float]] = {}
        for node in self.network.nodes:
            self.spatial_gossip_vectors[node.node_id] = (
                self.get_spatial_gossip_probability_vector(
                    self.network, node, dimension=self.dimension, rho=self.rho
                )
            )

    def select_targets(self, node_id: NodeID) -> list[NodeID]:
        return [np.random.choice(
                    list(self.spatial_gossip_vectors[node_id].keys()),
                    p=list(self.spatial_gossip_vectors[node_id].values()),
                )]


    def calculate_probability(self, node: Node, other_node: Node, dimension: int, rho: float) -> float:
        """Computes the probability associated to the distance of two nodes"""
        distance_vector: CoordinateSystemPoint = node.pos - other_node.pos
        distance = distance_vector.norm()
        return (distance + 1) ** (-dimension * rho)


    def get_spatial_gossip_probability_vector(self, network: Network, node: Node, dimension: int, rho: float) -> dict[NodeID, float]:
        """Computes the probability of choosing a neighbour for all nodes with [node] as source"""
        probs: dict[NodeID, float] = {}
        for target in network.nodes:
            if target.node_id == node.node_id:
                continue
            probs[target.node_id] = self.calculate_probability(node, target, dimension, rho)

        # Normalize
        total_sum = sum(list(probs.values()))
        for key, value in probs.items():
            probs[key] = value / total_sum

        return probs


class SpatialGossipWithCobraWalk(SpatialGossip):
    """SpatialGossip"""

    def __init__(self, network, dimension: int, rho: float, cobra_walk_rho):
        super().__init__(network, dimension, rho)
        self.cobra_walk_rho: float = cobra_walk_rho


    def select_targets(self, node_id: NodeID) -> list[NodeID]:
        target1: list[NodeID] = super().select_targets(node_id)

        random_value = random.random()
        cobra_partition = random_value <= self.cobra_walk_rho
        if cobra_partition:
            target2: list[NodeID] = super().select_targets(node_id)
            if target1[0] != target2[0]:
                return target1 + target2

        return target1
