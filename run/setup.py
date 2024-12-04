import sys
import os
# Thêm thư mục gốc vào sys.path để Python tìm được package 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import pandas as pd
from ultils.ultils import reset_db

# Load datasets
users_df = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python',
                    names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python',
                     names=['movie_id', 'title', 'genres'], encoding='latin1')
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

reset_db(users_df, movies_df, ratings_df)
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