from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3],
    'movie_id': [101, 102, 103, 101, 104, 102, 103],
    'rating': [5, 3, 4, 4, 5, 2, 5]
}
df = pd.DataFrame(data)


ratings_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)


item_similarity = cosine_similarity(ratings_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# Recommendation function
def recommend_movies(user_id, top_n=3):
    if user_id not in ratings_matrix.index:
        return []

    user_ratings = ratings_matrix.loc[user_id]
    scores = item_similarity_df.dot(user_ratings).sort_values(ascending=False)

    
    watched = user_ratings[user_ratings > 0].index
    scores = scores.drop(watched, errors='ignore')

    return scores.head(top_n).index.tolist()


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    top_n = int(request.args.get('top_n', 3))
    recommendations = recommend_movies(user_id, top_n)
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
