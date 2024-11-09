import torch
from ultils.graph import load_from_file
from torch_geometric.data import HeteroData
from model.SageGNN import Model


def infer(user_id, k=10):
    # initial
    data = load_from_file()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(data)
    model.load_state_dict(torch.load("model_gnn.pth", map_location=torch.device(device)))
    model.eval()
    
    # set up data
    scores = model.recommend(user_id, data)
    _, top_k_movie_indices = torch.topk(scores, k)
    return top_k_movie_indices


if __name__ == '__main__':
    user_id = int(input('Enter user_id: '))
    recommended_movie = infer(user_id)
    print(f"The recommended movie for user {user_id} is movie {recommended_movie.item()}.")