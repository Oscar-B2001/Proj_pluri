import pandas as pd
import numpy as np

def create_profiles(num_users=10):
    categories = [
        'health', 'news', 'sports', 'weather', 'entertainment', 'autos', 'lifestyle',
        'travel', 'foodanddrink', 'tv', 'finance', 'movies', 'video', 'music',
        'kids', 'middleeast', 'northamerica'
    ]

    reader_ids = [f"READER{i}" for i in range(1, num_users + 1)]

    user_profiles = pd.DataFrame({
        "pref": [list(np.random.choice(categories, 3, replace=False)) for _ in range(num_users)]
    }, index=reader_ids)

    user_liked = pd.DataFrame({
        "liked": [[] for _ in range(num_users)]
    }, index=reader_ids)

    user_watched = pd.DataFrame({
        "watched": [[] for _ in range(num_users)]
    }, index=reader_ids)

    return user_profiles, user_liked, user_watched
