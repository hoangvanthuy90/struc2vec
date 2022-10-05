from struct2vec import Struc2Vec
import networkx as nx

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    G = nx.read_edgelist('data/flight/brazil-airports.edgelist', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])  # read graph

    model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )  # init model
    model.train(window_size=5, iter=3)  # train model
    embeddings = model.get_embeddings()  # get embedding vectors
    print(embeddings)
