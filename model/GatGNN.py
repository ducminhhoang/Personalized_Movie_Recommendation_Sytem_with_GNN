import torch
from torch_geometric.nn import GATConv, to_hetero
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from torch import Tensor


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(hidden_channels, hidden_channels,heads = 8, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels,heads = 1, add_self_loops=False)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.movie_lin = torch.nn.Linear(18, hidden_channels)
        self.user_emb = torch.nn.Embedding(6040, hidden_channels)
        self.movie_emb = torch.nn.Embedding(6040, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        if "edge_label_index" in data["user", "rates", "movie"]:
            edge_label_index = data["user", "rates", "movie"].edge_label_index
        else:
            edge_label_index = data["user", "rates", "movie"].edge_index

        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            edge_label_index,
        )

        return pred