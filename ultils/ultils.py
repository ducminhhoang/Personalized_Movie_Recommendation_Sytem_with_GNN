import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from ultils.graph import save_graph
from ultils.preprocessing import process_movie, process_user
from model.SageGNN import Model
from ultils.vector import setup_vectordatabase


def reset_db(users, movies, ratings):
    if 'mapped_id' in users.columns:
        users = users.drop(columns=['mapped_id'])
    if 'mapped_id' in movies.columns:
        movies = movies.drop(columns=['mapped_id'])

    # Mapping of user and movie IDs to consecutive values
    unique_user_id = users['user_id'].unique()
    unique_user_id = pd.DataFrame(data={
        'user_id': unique_user_id,
        'mapped_id': pd.RangeIndex(len(unique_user_id)),
    })
    users = pd.merge(users, unique_user_id, on='user_id', how='inner')
    
    unique_movie_id = movies['movie_id'].unique()
    unique_movie_id = pd.DataFrame(data={
        'movie_id': unique_movie_id,
        'mapped_id': pd.RangeIndex(len(unique_movie_id)),
    })
    movies = pd.merge(movies, unique_movie_id, on='movie_id', how='inner')

    users.to_csv('resources/users.csv', index=False)
    movies.to_csv('resources/movies.csv', index=False)
    ratings.to_csv('resources/ratings.csv', index=False)
    print("Load xong data")

    # Set up graph from original data
    data = HeteroData()

    # preprocessing
    users = process_user(users)
    movies = process_movie(movies)

    data["user"].node_id = torch.arange(len(unique_user_id))
    data["movie"].node_id = torch.arange(len(unique_movie_id))

    # feature node
    data['user'].x = torch.tensor(users[['gender', 'age', 'occupation']].values, dtype=torch.float)
    data['movie'].x = torch.tensor(movies.drop(['title', 'movie_id', 'mapped_id'], axis=1).values, dtype=torch.float)

    # Tạo edges dựa trên ratings
    merged_user = ratings.merge(unique_user_id, on='user_id', how='left')
    merged_movie = ratings.merge(unique_movie_id, on='movie_id', how='left')
    ratings_user_id = torch.from_numpy(merged_user['mapped_id'].values)
    ratings_movie_id = torch.tensor(merged_movie['mapped_id'])
    edge_index = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
    data['user', 'rates', 'movie'].edge_index = edge_index

    # Tạo trọng số cho các cạnh (dùng rating làm trọng số)
    edge_weight = torch.tensor(ratings['rating'].values, dtype=torch.float)
    data['user', 'rates', 'movie'].edge_attr = edge_weight  # Gắn trọng số vào các cạnh
    data = T.ToUndirected()(data)
    save_graph(data)
    print('Load xong graph')

    # set up vector embedding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(data)
    model.load_state_dict(torch.load("model_gnn.pth", map_location=torch.device(device)))
    model.eval()
    user_embeddings, movies_embeddings = model.get_embeddings(data)
    setup_vectordatabase(user_embeddings, movies_embeddings)
    print("Load xong vector database")