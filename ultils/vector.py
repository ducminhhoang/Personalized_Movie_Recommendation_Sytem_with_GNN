import faiss
import numpy as np
import torch



def setup_vectordatabase(user_embeddings, movie_embeddings):
    # Chuyển embeddings sang NumPy array để đưa vào Faiss
    user_embeddings_np = user_embeddings.detach().cpu().numpy()
    movie_embeddings_np = movie_embeddings.detach().cpu().numpy()

    # Kích thước của mỗi embedding (dimension)
    dim = user_embeddings_np.shape[1]

    # Tạo chỉ mục Faiss (IndexFlatL2 là chỉ mục đơn giản sử dụng khoảng cách Euclidean)
    user_index = faiss.IndexFlatIP(dim)  # Chỉ mục cho người dùng
    movie_index = faiss.IndexFlatIP(dim)  # Chỉ mục cho phim

    # Thêm embeddings người dùng vào chỉ mục Faiss
    user_index.add(user_embeddings_np)

    # Thêm embeddings phim vào chỉ mục Faiss
    movie_index.add(movie_embeddings_np)

    # Lưu chỉ mục Faiss vào file
    faiss.write_index(user_index, "resources/user_index.faiss")
    faiss.write_index(movie_index, "resources/movie_index.faiss")


def add_user_embed(user_embeddings):
    # Tải lại chỉ mục Faiss từ file
    user_index = faiss.read_index("resources/user_index.faiss")

    user_embeddings_np = user_embeddings.detach().cpu().numpy()

    # Kích thước của mỗi embedding (dimension)
    dim = user_embeddings_np.shape[1]

    # Tạo chỉ mục Faiss cho người dùng
    user_index = faiss.IndexFlatIP(dim)  # Chỉ mục sử dụng khoảng cách Euclidean

    # Thêm các embedding người dùng vào chỉ mục Faiss
    user_index.add(user_embeddings_np)

    # Lưu chỉ mục Faiss vào file (optional, để lưu lại chỉ mục cho lần sau)
    faiss.write_index(user_index, "user_index.faiss")
    print("Đã add user embeddings vào vector database")


def vector_search(user_id, k=10):
    # Giả sử bạn đã tải lại chỉ mục Faiss
    user_index = faiss.read_index("user_index.faiss")
    movie_index = faiss.read_index("movie_index.faiss")

    # Lấy embedding của người dùng với ID tương ứng từ Faiss
    embedding = user_index.reconstruct(user_id)  # Faiss dùng chỉ mục bắt đầu từ 0, nên cần giảm 1

    # Chuyển đổi embedding về dạng tensor nếu cần
    user_embedding_tensor = torch.tensor(embedding)
    distances, indices = movie_index.search(user_embedding_tensor, k=k)
    return indices