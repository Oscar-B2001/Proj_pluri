import pandas as pd
import numpy as np

def create_profiles(num_users):
    categories = [
        'health', 'news', 'sports', 'weather', 'entertainment', 'autos', 'lifestyle',
        'travel', 'foodanddrink', 'tv', 'finance', 'movies', 'video', 'music', 
        'kids', 'middleeast', 'northamerica'
    ]

    reader_ids = [f"READER{i}" for i in range(1, num_users + 1)]
    
    user_profiles = {
        "Reader ID": reader_ids,
        "Likes": [list(np.random.choice(categories, 3, replace=False)) for _ in range(num_users)]
    }

    return pd.DataFrame(user_profiles)

