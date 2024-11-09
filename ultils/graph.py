import torch
from torch_geometric.data import HeteroData


def save_graph(graph, filename='graphData.pt'):
    try:
        torch.save(graph, filename)
    except:
        print("Kiểm tra lại file")


def load_from_file(filename='graphData.pt'):
    g = torch.load(filename)
    if not isinstance(g, HeteroData):
        print("Dữ liệu tải về không phải kiểu HeteroData.")
    else:
        return g


def add_node_to_graph(graph, feature_node, edges=None):
    graph['user'].x = feature_node
    if edges:
        edge_index = graph['user', 'rates', 'movie'].edge_index
        edge_index = torch.cat((edge_index, edges), dim=1)
        edge_index, _ = torch.unique(edge_index, dim=1, return_inverse=True)

        
        edge_weight = graph['user', 'rates', 'movie'].edge_attr
        edge_weight 

        graph['user', 'rates', 'movie'].edge_index = edge_index
    
