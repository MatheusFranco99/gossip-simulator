""" Clustering """

from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from utils.basic_types import NodeID
from core.network import Network


def create_cluster_nodes(
    network: Network, n_clusters: int = 9
) -> tuple[dict[int, list[NodeID]], dict[NodeID, int]]:
    """
    Clusters nodes in the given network using KMeans.

    Args:
        network (Network): The network to be clustered.
        eps (float): The maximum distance between two samples for them to be considered in the same neighborhood.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        dict[int, list[Node]]: A dictionary mapping cluster IDs to lists of nodes in that cluster.
    """
    # Extract positions as a 2D array for clustering
    positions = np.array([[node.pos.x, node.pos.y] for node in network.nodes])

    kmeans = KMeans(n_clusters=n_clusters)
    clustering = kmeans.fit(positions)

    cluster_map: dict[int, list[NodeID]] = defaultdict(list)
    node_to_cluster_map: dict[NodeID, int] = {}
    for node, cluster_label in zip(network.nodes, clustering.labels_):
        cluster_map[cluster_label].append(node)
        node_to_cluster_map[node.node_id] = cluster_label

    # y = [node.pos.x for node in network.nodes]
    # x = [node.pos.y for node in network.nodes]
    # plt.scatter(x, y, c = kmeans.labels_)
    # plt.show()

    # Remove noise cluster (label == -1)
    if -1 in cluster_map:
        del cluster_map[-1]

    return cluster_map, node_to_cluster_map
