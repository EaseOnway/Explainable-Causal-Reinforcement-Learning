from typing import Dict, Iterable, Literal
import graphviz as gv


def plot_digraph(nodes: Iterable[str], edges: Dict[str, Iterable[str]],
                 edgemode: Literal['pa', 'ch'] = 'pa', format='png'):
    g = gv.Digraph(format=format)

    for node in nodes:
        g.node(node)

    if edgemode == 'pa':
        for i, js in edges.items():
            for j in js:
                g.edge(j, i)
    elif edgemode == 'ch':
        for i, js in edges.items():
            for j in js:
                g.edge(i, j)
    else:
        raise ValueError(f"unknown edgemode '{edgemode}'")

    return g