import random
import ollama
import pandas as pd


def build_prompt(user_id, prefs, liked, watched, train_news_reduced):
    prompt = f"""Tu es un assistant intelligent qui recommande des articles à lire.
Voici les préférences de l'utilisateur {user_id} : {prefs}.
Voici les articles qu'il a déjà lus : {watched}.
Voici ceux qu'il a aimés : {liked}.

Voici une liste d'articles candidats. Pour chaque article, tu verras son ID, titre et résumé.
Sélectionne jusqu'à 2 articles par préférence de l'utilisateur qui pourraient plaire à l'utilisateur en question, en fonction de son historique.

Retourne uniquement une liste d'IDs d'articles séparés par des virgules.

Articles candidats :
"""

    candidates = train_news_reduced[
        (~train_news_reduced["article_id"].isin(watched))
    ].sample(n=min(20, len(train_news_reduced)))  # échantillon aléatoire

    for _, row in candidates.iterrows():
        prompt += f"\nID: {row['article_id']} | Titre: {row['title']} | Résumé: {row['abstract']}"

    return prompt, candidates["article_id"].tolist()


def ask_llm(prompt, model="mistral"):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print("Erreur LLM :", e)
        return ""


import re


def parse_llm_response(response, valid_ids):
    # Utiliser une regex pour ne récupérer que les IDs corrects
    ids = re.findall(r'N\d+', response)  # Capturer uniquement les IDs au format N12345

    # Filtrer pour ne garder que les IDs qui existent dans les articles candidats
    ids = [item for item in ids if item in valid_ids]

    return ids[:6]  # Limiter à 6 recommandations


def pred_article_llm(user_profiles, user_liked, user_watched, train_news_reduced, model="mistral", n_iter=5):
    news_dict = {}
    for user_id in user_profiles.index:
        prefs = user_profiles.loc[user_id, "pref"]
        liked = user_liked.loc[user_id, "liked"]
        watched = user_watched.loc[user_id, "watched"]

        prompt, valid_ids = build_prompt(user_id, prefs, liked, watched, train_news_reduced)
        llm_response = ask_llm(prompt, model)
        recommended_ids = parse_llm_response(llm_response, valid_ids)
        news_dict[user_id] = recommended_ids
        # Mettre à jour watched et liked
        watched = list(set(watched + recommended_ids))
        user_watched.loc[user_id, "watched"] = watched

        to_like = random.sample(recommended_ids, min(2, len(recommended_ids)))
        liked = list(set(liked + to_like))
        user_liked.loc[user_id, "liked"] = liked

    all_news = pd.DataFrame(list(news_dict.items()), columns=['user_id', 'recommended_news'])
    return user_liked, user_watched, all_news