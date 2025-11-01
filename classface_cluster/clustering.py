"""
Clustering of face embeddings into student identities.

This module constructs a similarity graph over embeddings using FAISS for
nearest neighbour search and then clusters the graph using either simple
connected components or HDBSCAN.  A post‑processing step merges small
clusters into larger ones based on centroid similarity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable, Optional

import numpy as np
import faiss


def build_knn_graph(embeddings: np.ndarray, k: int, sim_threshold: float) -> List[Tuple[int, int]]:
    """Construct a mutual k‑NN similarity graph.

    Parameters
    ----------
    embeddings: ndarray, shape (n_samples, dim)
        L2‑normalised embedding vectors.
    k: int
        Number of nearest neighbours to retrieve for each sample (including self).
    sim_threshold: float
        Minimum cosine similarity for retaining an edge.

    Returns
    -------
    list of (int, int)
        List of undirected edges (i, j) representing mutual neighbours above
        the similarity threshold.  Self edges are not included.
    """
    n, d = embeddings.shape
    # Build FAISS index with HNSW for approximate search
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efSearch = max(64, k * 2)
    index.add(embeddings)
    # Query k+1 neighbours (the first neighbour is self)
    distances, indices = index.search(embeddings, k + 1)
    # Convert distances to cosine similarity (since embeddings are L2‑normed, dot = 1 - 0.5*L2^2)
    # Actually FAISS returns L2 distances by default; we can compute cosine from dot directly
    # Instead compute dot product via matrix multiply; but easier to compute dot = 1 - 0.5*dist**2 if embeddings have unit norm and metric L2.
    edges = []
    # Build adjacency sets for mutual filtering
    neigh_dict = {i: set() for i in range(n)}
    for i in range(n):
        for dist, j in zip(distances[i][1:], indices[i][1:]):
            if j < 0 or j == i:
                continue
            # compute cosine similarity (approx): dot = 1 - 0.5 * (dist**2)
            sim = 1.0 - 0.5 * float(dist)
            if sim >= sim_threshold:
                neigh_dict[i].add(j)
    # Keep only mutual edges
    for i in range(n):
        for j in neigh_dict[i]:
            if i < j and i in neigh_dict.get(j, set()):
                edges.append((i, j))
    return edges


def connected_components(n_nodes: int, edges: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Compute connected components using a union–find structure.

    Returns a mapping from component representative to a list of member indices.
    """
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    # Group by root
    comp: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_nodes):
        comp[find(i)].append(i)
    return comp


def cluster_embeddings(embeddings: np.ndarray, image_ids: List[int],
                       sim_threshold: float, merge_threshold: float,
                       min_cluster_size: int, knn_k: int,
                       use_hdbscan: bool = False,
                       hdbscan_min_cluster_size: int = 8,
                       hdbscan_min_samples: int = 2) -> List[List[int]]:
    """Cluster embeddings into identities.

    Parameters
    ----------
    embeddings: ndarray, shape (n_samples, dim)
        L2‑normalised embedding vectors.
    image_ids: list of int
        Image ID for each embedding; used to forbid edges between faces in the same image.
    sim_threshold: float
        Threshold for creating edges between mutual nearest neighbours.
    merge_threshold: float
        Threshold for merging small clusters by centroid similarity.
    min_cluster_size: int
        Minimum cluster size to consider as a valid identity.
    knn_k: int
        Number of neighbours for k‑NN graph.
    use_hdbscan: bool
        Whether to use HDBSCAN instead of connected components.
    hdbscan_min_cluster_size: int
        Parameter for HDBSCAN; ignored if use_hdbscan is False.
    hdbscan_min_samples: int
        Parameter for HDBSCAN; ignored if use_hdbscan is False.

    Returns
    -------
    list of list of int
        A list of clusters, each being a list of face indices.
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    if use_hdbscan:
        try:
            import hdbscan
        except ImportError:
            raise RuntimeError("HDBSCAN is not installed; install hdbscan or run without --hdbscan")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            if lab >= 0:
                clusters[int(lab)].append(idx)
        # HDBSCAN may already filter out small clusters; we handle unknown later
    else:
        # Build similarity graph and prune cannot‑link edges
        edges = build_knn_graph(embeddings, knn_k, sim_threshold)
        # Remove edges connecting faces from the same image
        cannot = set()
        # Build mapping from image_id to face indices
        img_to_faces: Dict[int, List[int]] = defaultdict(list)
        for idx, img_id in enumerate(image_ids):
            img_to_faces[img_id].append(idx)
        for faces in img_to_faces.values():
            if len(faces) > 1:
                # Add all pairs (i,j) in this image to cannot
                for i in range(len(faces)):
                    for j in range(i + 1, len(faces)):
                        a, b = faces[i], faces[j]
                        if a < b:
                            cannot.add((a, b))
                        else:
                            cannot.add((b, a))
        pruned_edges = [e for e in edges if e not in cannot]
        comps = connected_components(n, pruned_edges)
        # Build clusters dictionary; keys are component representatives
        clusters = comps
    # Convert clusters mapping to list of lists
    cluster_list: List[List[int]] = list(clusters.values())
    # Merge small clusters into nearest larger ones
    # Compute centroids for each cluster
    centroids = []
    for cluster in cluster_list:
        emb = embeddings[cluster]
        centroids.append(np.mean(emb, axis=0))
    # Normalise centroids
    centroids = np.stack(centroids)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    # Determine which clusters are small
    large_mask = np.array([len(c) >= min_cluster_size for c in cluster_list])
    merged_clusters: List[List[int]] = []
    cluster_to_output: Dict[int, int] = {}
    large_indices_all = list(np.nonzero(large_mask)[0])
    for i, cluster in enumerate(cluster_list):
        if large_mask[i]:
            cluster_to_output[i] = len(merged_clusters)
            merged_clusters.append(cluster.copy())
        else:
            # Small cluster: try to merge with the most similar large cluster
            available_targets = [idx for idx in large_indices_all if idx in cluster_to_output]
            if not available_targets:
                # No processed large cluster exists; keep as standalone
                cluster_to_output[i] = len(merged_clusters)
                merged_clusters.append(cluster.copy())
                continue
            sims = [float(centroids[idx] @ centroids[i]) for idx in available_targets]
            best_pos = int(np.argmax(sims))
            if sims[best_pos] >= merge_threshold:
                target_cluster_idx = cluster_to_output[available_targets[best_pos]]
                merged_clusters[target_cluster_idx].extend(cluster)
            else:
                cluster_to_output[i] = len(merged_clusters)
                merged_clusters.append(cluster.copy())
    # Filter out empty entries
    merged_clusters = [c for c in merged_clusters if c]
    return merged_clusters


@dataclass
class ClusterResult:
    """Result of clustering, along with metadata for database insertion."""
    clusters: List[List[int]]
    labels: List[str]
    cluster_sizes: List[int]


def assign_labels(clusters: List[List[int]]) -> List[str]:
    """Assign canonical string labels (Student_0000, Student_0001, …).

    Returns a list of the same length as ``clusters`` with labels sorted by
    descending cluster size.  Unknown clusters are also labelled but not
    distinguished; they can be further processed downstream.
    """
    # Sort cluster indices by size descending
    order = sorted(range(len(clusters)), key=lambda i: len(clusters[i]), reverse=True)
    labels = [None] * len(clusters)
    for idx, cluster_idx in enumerate(order):
        labels[cluster_idx] = f"Student_{idx:04d}"
    return labels
