import networkx as nx

def read_graph(infile, weighted):
    """
    Reads the input network in networkx.
    """
    if weighted:
        G = nx.read_edgelist(
            infile,
            nodetype=int,
            data=(("weight", float),),
            create_using=nx.DiGraph(),
            delimiter=" ",
            comments="?",
        )
    else:
        G = nx.read_edgelist(
            infile,
            nodetype=int,
            data=(("weight", float),),
            create_using=nx.DiGraph(),
            delimiter=" ",
            comments="?",
        )
        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1
    G = G.to_undirected()
    return G


