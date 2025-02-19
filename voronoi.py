""" Voronoi tessellation of clusters """

import numpy as np
from scipy.spatial import Voronoi


def create_voronoi(cluster_map):

    # Calculate clusters centroids
    centroids = list()
    for cluster_nodes in cluster_map.values():
        positions = np.array(
            [[x, y] for x, y in [(node.pos.x, node.pos.y) for node in cluster_nodes]]
        )
        centroids.append(np.mean(positions, axis=0))

    # Voronoi tessellation
    centroids = np.array(centroids)[:, [1, 0]]
    # Extend centroids for wrapping at x = ±180
    extended_centroids = np.vstack(
        [
            centroids,  # Original
            centroids + np.array([360, 0]),  # Shifted right (wrap-around)
            centroids - np.array([360, 0]),  # Shifted left (wrap-around)
        ]
    )
    vor = Voronoi(extended_centroids)

    x_min, x_max = -180, 180
    y_min, y_max = -90, 90

    neighbors = {int(cluster_id): set() for cluster_id in range(len(centroids))}

    # Iterate through ridge_points to find neighbors
    for ridge, point_pair in zip(vor.ridge_vertices, vor.ridge_points):
        inside = False
        vertex_coords = [vor.vertices[i] for i in ridge if i != -1]
        for v in vertex_coords:
            if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max:
                inside = True
                break
        if not inside:
            continue

        cluster_a = point_pair[0] % len(centroids)
        cluster_b = point_pair[1] % len(centroids)
        neighbors[cluster_a].add(int(cluster_b))
        neighbors[cluster_b].add(int(cluster_a))

    return vor, neighbors
