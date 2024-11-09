import pandas as pd

def get_unique(df):
    unique_user_id = df[0].unique()

    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    unique_user_id.head()


def process_user(user):
    user['gender'] = user['gender'].apply(lambda x: 1 if x == 'M' else 0)
    return user

def process_movie(movies):
    all_genres = set(g for genres in movies['genres'].str.split('|') for g in genres)
    for genre in all_genres:
        movies[genre] = movies['genres'].apply(lambda x: int(genre in x.split('|')))
    movies = movies.drop('genres', axis=1)
    return movies