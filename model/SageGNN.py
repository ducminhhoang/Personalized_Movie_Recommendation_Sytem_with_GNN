from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import nn
from torch import Tensor
import torch

#
class GNN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(nn.Module):
    def __init__(self, data, hidden_channels=64):
        super().__init__()
        self.user_lin = nn.Linear(3, hidden_channels)
        self.movie_lin = nn.Linear(18, hidden_channels)
        self.user_emb = nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = nn.Embedding(data["movie"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    
    def forward(self, data) -> Tensor:
        x_dict = {
          "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
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

    def recommend(self, user_id, data) -> Tensor:
        user_embeddings, movie_embeddings = self.get_embeddings(data)

        user_embedding = user_embeddings[user_id]

        scores = (user_embedding * movie_embeddings).sum(dim=-1)

        return scores
    
    def get_embeddings(self, data):
        x_dict = {
          "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        user_embeddings = x_dict["user"]
        movie_embeddings = x_dict["movie"]
        return user_embeddings, movie_embeddings