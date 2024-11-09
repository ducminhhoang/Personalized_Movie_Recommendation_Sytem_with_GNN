import sys
import os
# Thêm thư mục gốc vào sys.path để Python tìm được package 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import sqlite3
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from ultils.preprocessing import process_movie, process_user
from ultils.graph import save_graph


# Load datasets
users_df = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python',
                    names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python',
                     names=['movie_id', 'title', 'genres'], encoding='latin1')
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

users = users_df.copy()
movies = movies_df.copy()
ratings = ratings_df.copy()

# Mapping of user IDs to consecutive values
unique_user_id = users['user_id'].unique()
unique_user_id = pd.DataFrame(data={
    'user_id': unique_user_id,
    'mapped_id': pd.RangeIndex(len(unique_user_id)),
})
users_df = pd.merge(users_df, unique_user_id, on='user_id', how='inner')
assert len(users_df) == len(users)

# Mapping of movie IDs to consecutive values
unique_movie_id = movies['movie_id'].unique()
unique_movie_id = pd.DataFrame(data={
    'movie_id': unique_movie_id,
    'mapped_id': pd.RangeIndex(len(unique_movie_id)),
})
movies_df = pd.merge(movies_df, unique_movie_id, on='movie_id', how='inner')
assert len(movies_df) == len(movies)

users_df.to_csv('resources/users.csv', index=False)
movies_df.to_csv('resources/movies.csv', index=False)
ratings_df.to_csv('resources/ratings.csv', index=False)
print("Load xong data")
# Kết nối với cơ sở dữ liệu SQLite
# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()

# # Tạo bảng Users
# cursor.execute('''CREATE TABLE IF NOT EXISTS Users (
#                     user_id INTEGER PRIMARY KEY,
#                     gender TEXT,
#                     age INTEGER,
#                     occupation TEXT,
#                     zip_code INTEGER,
#                     mapped_id INTEGER)''')

# # Tạo bảng Movies
# cursor.execute('''CREATE TABLE IF NOT EXISTS Movies (
#                     movie_id INTEGER PRIMARY KEY,
#                     title TEXT,
#                     genres TEXT,
#                     mapped_id INTEGER)''')

# # Tạo bảng Ratings
# cursor.execute('''CREATE TABLE IF NOT EXISTS Ratings (
#                     user_id INTEGER,
#                     movie_id INTEGER,
#                     rating INTEGER,
#                     timestamp INTEGER,
#                     PRIMARY KEY (user_id, movie_id))''')

# users.to_sql('Users', conn, if_exists='append', index=False)


# # Lưu thay đổi và đóng kết nối
# conn.commit()
# conn.close()


# Set up graph from original data
data = HeteroData()

# preprocessing
users = process_user(users)
movies = process_movie(movies)

data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(unique_movie_id))

# feature node
data['user'].x = torch.tensor(users[['gender', 'age', 'occupation']].values, dtype=torch.float)
data['movie'].x = torch.tensor(movies.drop(['title', 'movie_id'], axis=1).values, dtype=torch.float)

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