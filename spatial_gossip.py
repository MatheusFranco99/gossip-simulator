""" Spatial gssip algorithm"""

from basic_types import NodeID
from network import Network
from node import Node
from position import CoordinateSystemPoint


def calculate_probability(
    node: Node, other_node: Node, dimension: int, rho: float
) -> float:
    """Computes the probability associated to the distance of two nodes"""
    distance_vector: CoordinateSystemPoint = node.pos - other_node.pos
    distance = distance_vector.norm()
    return (distance + 1) ** (-dimension * rho)


def get_spatial_gossip_probability_vector(
    network: Network, node: Node, dimension: int, rho: float
) -> dict[NodeID, float]:
    """Computes the probability of choosing a neighbour for all nodes with [node] as source"""
    probs: dict[NodeID, float] = {}
    for target in network.nodes:
        if target.node_id == node.node_id:
            continue
        probs[target.node_id] = calculate_probability(node, target, dimension, rho)

    # Normalize
    total_sum = sum(list(probs.values()))
    for key, value in probs.items():
        probs[key] = value / total_sum

    return probs
