import streamlit as st
import sqlite3
import pandas as pd
from run.infer import infer
import time
from ultils.ultils import reset_db

user_age_dict = {1:  "Dưới 18",
             18:  "18-24",
             25:  "25-34",
             35:  "35-44",
             45:  "45-49",
             50:  "50-55",
             56:  "56+"}
user_job_dict = {
    0:  "Khác",
    1:  "học thuật/giáo dục",
    2:  "nghệ sĩ",
    3:  "nhân viên văn phòng/hành chính",
    4:  "sinh viên/học sinh",
    5:  "dịch vụ khách hàng",
    6:  "bác sĩ/chăm sóc sức khỏe",
    7:  "quản lý/ CEO",
    8:  "nông dân",
    9:  "nội trợ",
    10:  "học sinh",
    11:  "luật sư",
    12:  "lập trình viên",
    13:  "nghỉ hưu",
    14:  "sales/marketing",
    15:  "nhà khoa học",
    16:  "tự chủ",
    17:  "kỹ sư/kỹ thuật viên",
    18:  "thợ thủ công, thợ buôn",
    19:  "thất nghiệp",
    20:  "nhà văn"
}

# Giả sử bạn đã huấn luyện mô hình GNN của mình và có hàm để đề xuất phim
# Hàm này trả về danh sách các movie_id được đề xuất
def get_recommended_movies(user_id, k=10):
    recommended_movie_ids = infer(user_id)
    return recommended_movie_ids

# Kết nối đến cơ sở dữ liệu SQLite
def get_db_connection():
    conn = sqlite3.connect('movie_recommendation.db')
    conn.row_factory = sqlite3.Row
    return conn

# Streamlit UI
def main():
    st.title("Movie Recommendation System")

    # Lấy danh sách người dùng từ cơ sở dữ liệu
    # conn = get_db_connection()
    # users_df = pd.read_sql_query('SELECT * FROM Users', conn)  # Dữ liệu người dùng
    # conn.close()

    users_df = pd.read_csv(r'resources/users.csv')
    ratings_df = pd.read_csv(r'resources/ratings.csv')
    movies_df = pd.read_csv(r'resources/movies.csv')
    merged_data = pd.merge(ratings_df, movies_df, on='movie_id')
    # Chọn người dùng từ dropdown
    user_id = st.selectbox("Chọn id người dùng", [None] + users_df['user_id'].tolist())

    # Khi chọn người dùng, hiển thị thông tin người dùng và đề xuất phim
    if user_id:
        st.subheader(f"User Information - ID: {user_id}")
        user_info = users_df[users_df['user_id'] == user_id].iloc[0]
        st.write(f"Gender: {'Nam' if user_info['gender'] == 'M' else 'Nữ'}")
        st.write(f"Age: {user_age_dict[user_info['age']]}")
        st.write(f"Occupation: {user_job_dict[user_info['occupation']]}")
        

        col1, col2 = st.columns(2)

        # Cột 1: History Films
        with col1:
            st.subheader('History Films')
            # History film
            user_movies_his = merged_data[merged_data['user_id'] == user_id][['title', 'rating', 'genres']]
            st.dataframe(user_movies_his)

            # static
            

        # Cột 2: Recommend Films (Giả sử là phim chưa xem của user)
        with col2:
            st.subheader('Phim gợi ý cho bạn')
            # Lấy các bộ phim đề xuất
            recommended_movie_ids = get_recommended_movies(user_info['mapped_id']).tolist()
            recommended_movies = movies_df[movies_df['mapped_id'].isin(recommended_movie_ids)]
            st.dataframe(recommended_movies[['title', 'genres']])
            title = st.selectbox('Chọn title:', [None] + recommended_movies['title'].tolist())
            if title:
                choice_movie = recommended_movies[recommended_movies['title'] == title].iloc[0].to_dict()
                new_rating = pd.DataFrame({"user_id": [user_id], "movie_id": [choice_movie['movie_id']], "rating": [5], "timestamp": [int(time.time())]})
                ratings_df = pd.concat([ratings_df, new_rating], ignore_index=False)
                reset_db(users_df, movies_df, ratings_df)
        
        
        # Lấy thông tin các bộ phim từ cơ sở dữ liệu
        # conn = get_db_connection()
        # movies_df = pd.read_sql_query('SELECT * FROM Movies WHERE movie_id IN ({})'.format(','.join('?' for _ in recommended_movie_ids)), conn, params=recommended_movie_ids)
        # conn.close()
        

        # for _, movie in movies_df.iterrows():
        #     st.write(f"**{movie['title']}** - Genres: {movie['genres']}")

if __name__ == "__main__":
    main()
