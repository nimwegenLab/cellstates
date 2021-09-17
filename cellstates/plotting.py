import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import dendrogram
from .cluster import Cluster
from .helpers import get_scipy_hierarchy, hierarchy_to_newick, clusters_from_hierarchy

USE_ETE = True
try:
    from ete3 import Tree, TreeStyle, NodeStyle
except ImportError:
    USE_ETE = False


def plot_hierarchy_scipy(hierarchy_df, n_groups=2,
                         dflt_color="#808080", colors=None,
                         **kwargs):
    """
    function to plot cellstate hierarchy with scipy,
    colouring of branches into n_groups.
    Leaf-order can be different than in plot_hierarchy_ete3

    Parameters
    ----------
    hierarchy_df
    n_groups : int, default=2
        number of groups to color
    dflt_color : color understood by matplotlib, default '#808080'
        color or root and lower branches
    colors : list of colors understood by matplotlib
        colors of grouped upper branches
    **kwargs :
        passed on to scipy.cluster.hierarchy.dendrogram
        e.g. pass ax keyword to set matplotlib axis

    Returns
    -------
    R : dict
        dictionary of data structures returned by
        scipy.cluster.hierarchy.dendrogram
    """
    Z, labels = get_scipy_hierarchy(hierarchy_df, return_labels=True)
    clusters = clusters_from_hierarchy(hierarchy_df, cluster_init=labels, steps=-n_groups+1)
    if colors is None:
        colors = plt.cm.hsv(np.linspace(0, 1, n_groups+1))[:-1]

    cluster_colors = {i: matplotlib.colors.to_hex(c)
                      for i, c in zip(np.unique(clusters), colors)}
    # Color mapping
    D_leaf_colors = {i:cluster_colors[c] for i, c in enumerate(clusters)}

    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    link_cols = dict()
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x] for x in i12)
        link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_color

    R = dendrogram(Z, color_threshold=None, #Z[-n_groups+1, 2])
                   link_color_func=lambda x: link_cols[x],
                   labels=list(labels),
                   **kwargs)
    return R


if USE_ETE:
    def plot_hierarchy_ete3(hierarchy_df, clusters, n_groups=2,
                            colors=None, linewidth = 2,
                            show_cells=False, leaf_scale=1.,
                            file_path=None):
        """
        Parameters
        ----------
        hierarchy_df
        clusters
        n_groups : int, default=2
            number of groups to color
        colors : list of colors understood by ete3
            (RGB hex code or SVG color name)
        linewidth : float, default=2
        show_cells : bool, default=False
            whether to have cells or clusters as leaves.
            If False, leaf node size is proportional to number of cells in
            cluster
        leaf_scale : float, default=0.2
            global scale of leaf node sizes
        file_path : str
            if given, tree will be rendered as pdf
        Returns
        -------
        t : formatted ete3.Tree object
        ts : ete3.TreeStyle object
        """
        newick_string = hierarchy_to_newick(hierarchy_df, clusters, cell_leaves=show_cells)
        t = Tree(newick_string, format=1)

        cellstate_names, cellstate_sizes = np.unique(clusters, return_counts=True)
        size_dict = dict(zip(cellstate_names, cellstate_sizes))
        all_leaf_names = np.array([f'C{c}' for c in cellstate_names])
        h_clusters_cellstates = clusters_from_hierarchy(hierarchy_df,
                                                        cluster_init=cellstate_names,
                                                        steps=-n_groups+1)
        cluster_names = np.unique(h_clusters_cellstates)

        if colors is None:
            colors = plt.cm.hsv(np.linspace(0, 1, n_groups+1))[:-1]
        color_map = {cn:matplotlib.colors.to_hex(cl) for cn, cl in zip(cluster_names, colors)}

        ts = TreeStyle()
        ts.show_leaf_name=False
        ts.scale = 3e-5
        ts.rotation = 90

        base_color='black'
        base_style = NodeStyle()
        base_style['vt_line_width'] = linewidth
        base_style['hz_line_width'] = linewidth
        base_style['size'] = 0
        base_style["vt_line_color"] = base_color
        base_style["hz_line_color"] = base_color

        t.set_style(base_style)
        for n in t.traverse():
            n.set_style(base_style)

        # color subbranches of tree in their respective colors
        for cn in cluster_names:
            color = color_map[cn]
            style = NodeStyle(**base_style)
            style["vt_line_color"] = color
            style["hz_line_color"] = color
            style['fgcolor'] = color

            leaf_names = all_leaf_names[(h_clusters_cellstates==cn)]
            if len(leaf_names) == 1:
                node = t.search_nodes(name=leaf_names[0])[0]
                leaf_style = NodeStyle(**style)
                if not show_cells:
                    cellstate_id = int(node.name[1:])
                    leaf_style['size'] = np.sqrt(size_dict[cellstate_id])*leaf_scale
                node.set_style(leaf_style)
            else:
                ancestor = t.get_common_ancestor([str(l) for l in leaf_names])
                ancestor.set_style(style)

                for node in ancestor.iter_descendants():
                    if node.is_leaf():
                        leaf_style = NodeStyle(**style)
                        if not show_cells:
                            cellstate_id = int(node.name[1:])
                            leaf_style['size'] = np.sqrt(size_dict[cellstate_id])*leaf_scale
                        node.set_style(leaf_style)
                    else:
                        node.set_style(style)
        if file_path:
            t.render(file_path, tree_style=ts)

        return t, ts
