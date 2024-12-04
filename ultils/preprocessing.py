import pandas as pd

def process_user(user):
    user['gender'] = user['gender'].apply(lambda x: 1 if x == 'M' else 0)
    return user

def process_movie(movies):
    all_genres = set(g for genres in movies['genres'].str.split('|') for g in genres)
    for genre in all_genres:
        movies[genre] = movies['genres'].apply(lambda x: int(genre in x.split('|')))
    movies = movies.drop('genres', axis=1)
    return movies