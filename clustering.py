from typing import List
from dataclasses import dataclass
from kmedoids import KMedoids
from sklearn.cluster import KMeans
import numpy as np


# def compute_clustering(
#     elem_to_cluster: np.array, method: str, clustering_args: dict
# ):
#     if method == "Kmeans":
#         cluster_obj = KMeans(**clustering_args)
#         cluster_obj.fit(elem_to_cluster)
#         res = cluster_obj.labels_
#     elif method == "Kmedoids":
#         cluster_obj = KMedoids(**clustering_args)
#         cluster_obj.fit(elem_to_cluster)
#         res = cluster_obj.labels_
#     return res


def compute_clustering(
    elem_to_cluster: np.array, clustering_recipe: List
) -> List:
    clust_obj = clustering_recipe[0](**clustering_recipe[1])
    clust_obj.fit(elem_to_cluster)
    return clust_obj.labels_
