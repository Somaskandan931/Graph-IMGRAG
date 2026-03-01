"""
src/graph/build_graph.py
Build a KNN similarity graph from image embeddings.

Graph design (report §7):
    Node = image
    Edge = cosine similarity; each node → K nearest neighbours

Run standalone:
    python src/graph/build_graph.py
"""

import os, sys, pickle
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.helpers import (
    load_config, get_logger, ensure_dirs,
    load_pickle, get_category,
)

log = get_logger("Graph")

try:
    from pyvis.network import Network
    PYVIS = True
except ImportError:
    PYVIS = False


def build(image_paths: list, embeddings: np.ndarray, cfg: dict):
    """
    Construct KNN graph.

    Returns:
        G          : nx.Graph
        sim_matrix : np.ndarray  (N x N)
        labels     : list[str]   filenames

    Saves:
        outputs/graphs/similarity_graph.gpickle
        outputs/graphs/similarity_graph.gexf
    """
    k        = cfg["graph"]["knn_k"]
    out_gpkl = cfg["graph"]["output_file"]
    out_gexf = cfg["graph"]["output_gexf"]

    ensure_dirs("outputs/graphs")

    sim_matrix = cosine_similarity(embeddings)
    n          = len(image_paths)
    labels     = [os.path.basename(p) for p in image_paths]

    # Resume
    if os.path.exists(out_gpkl):
        log.info(f"Resuming — loading cached graph from '{out_gpkl}'")
        with open(out_gpkl, "rb") as f:
            G = pickle.load(f)
        return G, sim_matrix, labels

    log.info(f"Building KNN graph  k={k}, n={n} ...")
    G = nx.Graph()
    for i, lbl in enumerate(labels):
        G.add_node(i, label=lbl, path=image_paths[i],
                   category=get_category(image_paths[i]))

    edge_count = 0
    for i in range(n):
        neighbours = np.argsort(sim_matrix[i])[::-1][1: k + 1]
        for j in neighbours:
            if not G.has_edge(i, int(j)):
                G.add_edge(i, int(j), weight=float(sim_matrix[i, j]))
                edge_count += 1

    avg_deg = sum(d for _, d in G.degree()) / n
    log.info(f"Graph: {G.number_of_nodes()} nodes | {edge_count} edges | "
             f"avg degree={avg_deg:.2f}")

    with open(out_gpkl, "wb") as f:
        pickle.dump(G, f)
    nx.write_gexf(G, out_gexf)
    log.info(f"Saved → '{out_gpkl}' & '{out_gexf}'")
    return G, sim_matrix, labels


def visualize(G: nx.Graph, labels: list, cfg: dict):
    """
    Draw the graph with community colouring.
    Saves static PNG + optional interactive HTML.
    """
    from networkx.algorithms.community import greedy_modularity_communities
    communities    = list(greedy_modularity_communities(G))
    palette        = list(plt.cm.Set2.colors) + list(plt.cm.Set1.colors)
    node_color_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_color_map[node] = palette[idx % len(palette)]

    colors  = [node_color_map.get(n, (0.7, 0.7, 0.7)) for n in G.nodes()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, seed=42, k=0.6)

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=280, alpha=0.92)
    nx.draw_networkx_edges(G, pos, width=[w * 2.5 for w in weights],
                           alpha=0.35, edge_color="#888888")
    nx.draw_networkx_labels(G, pos,
                            labels={n: labels[n][:15] for n in G.nodes()},
                            font_size=5.5)

    patches = [
        mpatches.Patch(color=palette[i % len(palette)],
                       label=f"Cluster {i+1}  ({len(c)} images)")
        for i, c in enumerate(communities)
    ]
    plt.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)
    plt.title("Text-Based Image Similarity Graph  (KNN, k=5)",
              fontsize=14, fontweight="bold")
    plt.axis("off")

    out = cfg["graph"]["output_png"]
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Graph image saved → '{out}'")

    if PYVIS:
        _interactive(G, labels, node_color_map, palette)
    else:
        log.info("Install pyvis for interactive HTML graph.")


def _interactive(G, labels, node_color_map, palette):
    net = Network(height="800px", width="100%",
                  bgcolor="#0d1117", font_color="#e6edf3",
                  select_menu=True, filter_menu=True)
    net.set_options(
        '{"physics":{"barnesHut":{"gravitationalConstant":-8000,"springLength":150}}}'
    )
    for node in G.nodes():
        rgb = node_color_map.get(node, (0.6, 0.6, 0.6))
        hx  = "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in rgb[:3]])
        cat = G.nodes[node].get("category", "?")
        net.add_node(node, label=labels[node][:20], color=hx,
                     title=f"{labels[node]}<br>{cat}")
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d["weight"],
                     title=f"sim={d['weight']:.3f}")
    out = "outputs/graphs/interactive_graph.html"
    net.save_graph(out)
    log.info(f"Interactive graph saved → '{out}'")


if __name__ == "__main__":
    cfg = load_config()
    pkl = cfg["embeddings"]["output_file"]
    if not os.path.exists(pkl):
        log.error("Run generate_embeddings.py first.")
        sys.exit(1)
    data        = load_pickle(pkl)
    image_paths = data["image_paths"]
    embeddings  = data["embeddings"]
    G, sim, lbl = build(image_paths, embeddings, cfg)
    visualize(G, lbl, cfg)