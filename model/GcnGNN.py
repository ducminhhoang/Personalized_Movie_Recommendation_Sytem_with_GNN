import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNRecommendationModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(GNNRecommendationModel, self).__init__()

        # GCN layers
        self.gcn_user = GCNConv(3, hidden_dim)
        self.gcn_movie = GCNConv(18, hidden_dim)

        # MLP for prediction (output is the predicted rating)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        # Đặc trưng của người dùng và phim
        user_x = data['user'].x
        movie_x = data['movie'].x

        # Edge indices và edge weights
        edge_index = data['user', 'rates', 'movie'].edge_index
        edge_attr = data['user', 'rates', 'movie'].edge_attr


        # GCN cho người dùng
        user_out = self.gcn_user(user_x, edge_index, edge_attr)
        # GCN cho phim
        movie_out = self.gcn_movie(movie_x, edge_index, edge_attr)
        print(edge_attr)
        # Ghép đặc trưng người dùng và phim cho mỗi cạnh
        out = torch.cat([user_out, movie_out], dim=-1)

        # Dự đoán rating từ MLP
        rating_pred = self.mlp(out)

        return rating_pred