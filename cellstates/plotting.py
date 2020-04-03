import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from ete3 import Tree, TreeStyle, NodeStyle
from scipy.cluster.hierarchy import dendrogram
from .cluster import Cluster
from .helpers import get_scipy_hierarchy, hierarchy_to_newick, clusters_from_hierarchy


def plot_hierarchy_scipy(hierarchy_df, n_groups, **kwargs):
    Z = get_scipy_hierarchy(hierarchy_df)
    clusters = clusters_from_hierarchy(hierarchy_df, steps=-n_groups+1)

    colors = plt.cm.hsv(np.linspace(0, 1, n_groups+1))
    cluster_colors = {i: matplotlib.colors.to_hex(c) for i, c in zip(np.unique(clusters), colors)}
    # Color mapping
    dflt_col = "#808080"   # Unclustered gray
    D_leaf_colors = {i:cluster_colors[c] for i, c in enumerate(clusters)}

    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    link_cols = dict()
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x] for x in i12)
        link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col

    fig, ax = plt.subplots(1,1, figsize=(9, 6))
    R = dendrogram(Z, color_threshold=None, #Z[-n_groups+1, 2])
                   link_color_func=lambda x: link_cols[x],
                   ax=ax, **kwargs)
    ax.set_yscale('symlog', linthreshy=1e3)
    fig.subplots_adjust(left=0.05, right=0.99)
    return fig, ax, R


def plot_hierarchy_ete3(hierarchy_df, clusters, n_groups=3, file_path=None):
    """
    Parameters
    ----------
    hierarchy_df
    clusters
    n_groups : int, default=3
        number of groups to color
    file_path : str
        if given, tree will be rendered as pdf

    Returns
    -------
    t : formatted ete3.Tree object
    """
    newick_string = hierarchy_to_newick(hierarchy_df, clusters, cell_leaves=True)
    t = Tree(newick_string, format=1)
    colors = plt.cm.hsv(np.linspace(0, 1, n_groups+1))
    cluster_colors = [matplotlib.colors.to_hex(c) for c in colors]

    ts = TreeStyle()
    ts.show_leaf_name=False
    ts.scale = 1e-4
    ts.rotation = 90

    base_style = NodeStyle()
    base_style['size'] = 0
    base_style['bgcolor'] = "#ffffff"
    base_style['hz_line_width'] = 5
    base_style['vt_line_width'] = 5

    root_node_names = [f'I{i}' for i in range(0, n_groups-1)]
    color_index = 0

    for i in range(0, n_groups-1):
        root_node = t.search_nodes(name=f'I{i}')[0]
        root_node.set_style(base_style)
        root_node.img_style['size'] = 0
        for child in root_node.children:
            if child.name in root_node_names:
                continue
            else:
                color = cluster_colors[color_index]
                color_index += 1
                style = NodeStyle()
                style['size'] = 0
                style['vt_line_color'] = color
                style['hz_line_color'] = color
                style['vt_line_width'] = 3
                style['hz_line_width'] = 1

                child.set_style(style)
                for node in child.iter_descendants():
                    node.set_style(style)
    if file_path:
        t.render(file_path, tree_style=ts)

    return t, ts
