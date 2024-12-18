""" Voronoi tessellation of clusters """

import numpy as np
from scipy.spatial import Voronoi

def create_voronoi(cluster_map):

    # Calculate clusters centroids
    centroids = list()
    for cluster_nodes in cluster_map.values():
        positions = np.array([[x, y] for x, y in [(node.pos.x, node.pos.y) for node in cluster_nodes]])
        centroids.append(np.mean(positions, axis=0))

    # Voronoi tessellation
    centroids = np.array(centroids)[:, [1, 0]]
    vor = Voronoi(centroids)

    x_min, x_max = -170, 190
    y_min, y_max = -50, 70

    neighbors = {int(cluster_id): set() for cluster_id in cluster_map}

    # Iterate through ridge_points to find neighbors
    for ridge, point_pair in zip(vor.ridge_vertices, vor.ridge_points):
        inside = False
        vertex_coords = [vor.vertices[i] for i in ridge if i != -1]
        for v in vertex_coords:
            if (x_min <= v[0] <= x_max and y_min <= v[1] <= y_max):
                inside = True
                break
        if not inside:
            continue

        cluster_a, cluster_b = point_pair
        neighbors[cluster_a].add(int(cluster_b))
        neighbors[cluster_b].add(int(cluster_a))

    return vor, neighbors
