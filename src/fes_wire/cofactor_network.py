from itertools import combinations
import numpy as np
import networkx as nx
from .structures import cofactors, coords


def analyze(structure, chain_map, cutoff):
    graph = nx.Graph()
    clusters = cofactors(structure)
    for index, residue in enumerate(clusters):
        graph.add_node(
            index,
            chain=residue.parent.id,
            **chain_map[residue.parent.id],
            centroid_A=coords(residue).mean(0).tolist(),
        )
    distances = []
    for i, j in combinations(range(len(clusters)), 2):
        a, b = coords(clusters[i]), coords(clusters[j])
        center = float(np.linalg.norm(a.mean(0) - b.mean(0)))
        edge = float(np.linalg.norm(a[:, None] - b[None, :], axis=-1).min())
        u, v = graph.nodes[i], graph.nodes[j]
        delta = abs(u["layer"] - v["layer"])
        kind = (
            "intraring" if delta == 0 else ("axial" if u["subunit"] == v["subunit"] else "diagonal")
        )
        row = dict(
            i=i,
            j=j,
            centroid_distance_A=center,
            minimum_heavy_atom_distance_A=edge,
            kind=kind,
            layer_separation=delta,
        )
        distances.append(row)
        if delta <= 1 and edge <= cutoff:
            graph.add_edge(i, j, **row)
    components = list(nx.connected_components(graph))
    last = max((v["layer"] for _, v in graph.nodes(data=True)), default=0)
    spanning = any(
        {graph.nodes[n]["layer"] for n in comp} == set(range(last + 1)) for comp in components
    )
    return dict(
        label="Geometrical proxy for potential electron-transfer pathways",
        cutoff_A=cutoff,
        nodes=[dict(id=i, **d) for i, d in graph.nodes(data=True)],
        edges=[d for _, _, d in graph.edges(data=True)],
        distances=distances,
        components=[sorted(c) for c in components],
        spans_layers=spanning,
    )


def plot_network(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    positions = {}
    for node in result["nodes"]:
        graph.add_node(node["id"])
        x, y, z = node["centroid_A"]
        positions[node["id"]] = (float(np.degrees(np.arctan2(y, x))), z)
    graph.add_edges_from((e["i"], e["j"]) for e in result["edges"])
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    nx.draw_networkx(graph, pos=positions, ax=axis, node_size=160, font_size=7)
    axis.set_axis_on()
    axis.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    axis.set_xlabel("Cofactor azimuth (degrees)")
    axis.set_ylabel("Axial position (Å)")
    axis.set_title("Geometrical cofactor network")
    figure.savefig(path, dpi=160)
    plt.close(figure)
