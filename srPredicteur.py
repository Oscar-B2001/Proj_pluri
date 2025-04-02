import random
import pandas as pd

def get_raw(pref_list, watched_list, train_news_reduced):
    movies_bundel = []
    for pref in pref_list:
        news_pref = train_news_reduced[
            (train_news_reduced["category"] == pref) &
            (~train_news_reduced["article_id"].isin(watched_list))
        ].head(2)
        movies_bundel.extend(news_pref["article_id"].tolist())
    return movies_bundel


def get_filtered(pref_list, liked_list, watched_list, train_news_reduced, cosine_df):
    result_articles = []
    train_news_to_get = train_news_reduced[
        ~train_news_reduced["article_id"].isin(watched_list)
    ]

    for pref in pref_list:
        news_pref = train_news_to_get[train_news_to_get["category"] == pref]
        article_similar_liked = train_news_reduced[
            (train_news_reduced["article_id"].isin(liked_list)) &
            (train_news_reduced["category"] == pref)
        ]["article_id"]

        scored_articles = []

        for new_article in news_pref["article_id"]:
            new_cosine = 0
            n = len(article_similar_liked)
            if n == 0:
                continue

            for liked_article in article_similar_liked:
                # Utilise .get() pour éviter des KeyError
                sim = cosine_df.get(liked_article, {}).get(new_article, 0)
                new_cosine += sim / n

            scored_articles.append((new_article, new_cosine))

        # Tri des articles par score décroissant
        top_articles = sorted(scored_articles, key=lambda x: x[1], reverse=True)[:2]
        result_articles.extend([art[0] for art in top_articles])

    return result_articles


def pred_article(user_profiles, user_liked, user_watched, train_news_reduced, cosine_df):
    # Initialiser un dictionnaire pour stocker les articles générés par utilisateur
    news_dict = {}

    for user_id in user_profiles.index:
        prefs = user_profiles.loc[user_id, "pref"]
        liked = user_liked.loc[user_id, "liked"]
        watched = user_watched.loc[user_id, "watched"]

        # Obtenir 2 articles par catégorie (max 6)
        if not liked:
            news = get_raw(prefs, watched, train_news_reduced)
        else:
            news = get_filtered(prefs, liked, watched, train_news_reduced, cosine_df)

        # Limiter à 6 articles uniques
        news = list(set(news))[:6]

        # Mettre à jour watched sans doublons
        watched = list(set(watched + news))
        user_watched.loc[user_id, "watched"] = watched

        # Sélectionner 2 articles à liker au max
        to_like = random.sample(news, min(2, len(news)))
        liked = list(set(liked + to_like))
        user_liked.loc[user_id, "liked"] = liked

        # Ajouter les articles générés pour cet utilisateur dans le dictionnaire
        news_dict[user_id] = news

    # Convertir le dictionnaire en DataFrame
    all_news = pd.DataFrame(list(news_dict.items()), columns=['user_id', 'recommended_news'])

    # Retourner les DataFrames mis à jour et le DataFrame all_news
    return user_liked, user_watched, all_news





