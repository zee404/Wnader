import pickle
import pandas as pd
from bing_image_downloader import downloader
import re
from urllib.parse import quote
import glob
from models.utils import Util


class get_recomendation:
    def __init__(self, categories) -> None:
        weight_path = "models/hybrid_model/weight/"
        # load the model from a file
        with open(weight_path + 'knn_weights.pkl', 'rb') as f:
            self.knn_model = pickle.load(f)

        # load the model weights from a file
        with open(weight_path + 'kmeans_weights.pkl', 'rb') as f:
            self.kmeans_model = pickle.load(f)
        self.categories = categories
        self.reviews_data = pd.read_csv('models/hybrid_model/data/final_data_train.csv')
        self.reviews_data_kmeans = pd.read_csv('models/hybrid_model/data/final_data_reviews_kmeans.csv')
        self.reviews_data_kmeans['cluster'] = self.kmeans_model.predict(
            self.reviews_data_kmeans[['review_star', 'sentiment_score', 'state', 'city']])
        self.df_train_knn_final = \
            self.reviews_data[['user_id', 'business_id', 'review_star']].groupby(['user_id', 'business_id'])[
                'review_star'].mean().unstack().fillna(0)
        self.business_data = pd.read_csv('models/hybrid_model/data/business_restaurant_clean.csv')

    def get_image(self, name):
        # print(name)
        util = Util()
        name = name.replace("_", " ")

        dir_path = "media/downloads"
        try:
            cont = util.check_string(name)
            if cont:
                name = util.remove_special_characters(name)
            downloader.download(name, limit=1, output_dir=dir_path, adult_filter_off=True, force_replace=False,
                                timeout=60)

            for filename in glob.glob("media/downloads/{name}/*jpg".format(name=name)) + glob.glob(
                    "media/downloads/{name}/*png".format(name=name)):
                return filename
        except Exception as e:
            # print(e)
            # for filename in glob.glob("downloads/*jpg"):
            #     return filename
            return None

    def get_kmeans_recc(self, user_id):
        reviews_data_kmeans = self.reviews_data_kmeans.loc[self.reviews_data_kmeans['user_id'] == user_id]
        # print(reviews_data_kmeans)

        kmeans_rec = self.kmeans_model.predict(reviews_data_kmeans[['review_star', 'sentiment_score', 'state', 'city']])
        kmeans_rec = list(dict.fromkeys(kmeans_rec))
        # print(kmeans_rec)
        data = self.reviews_data_kmeans[self.reviews_data_kmeans['cluster'].isin(kmeans_rec)]

        data = data[['review_star', 'business_id']]
        grouped_data = data.groupby(['business_id']).agg({'review_star': 'mean'})

        df_merged = pd.merge(grouped_data, self.business_data, on='business_id', how='left')
        filtered_df = df_merged[df_merged['categories'].str.contains('|'.join(self.categories))]

        df_result = filtered_df[['business_id', 'review_star']]

        sorted_data = df_result.sort_values(by='review_star', ascending=False)
        sorted_data.set_index('business_id', inplace=True)
        kmeans_recs = sorted_data.head(100).to_dict()['review_star']

        return kmeans_recs

    def get_knn_recc(self, user_id):
        self.reviews_data_knn = pd.read_csv('models/hybrid_model/data/final_data_reviews_knn.csv')
        # Create empty dataframe with 0 values
        input_dict = self.reviews_data_knn.to_dict()
        reviews_data_user = self.reviews_data.loc[self.reviews_data['user_id'] == user_id]
        reviews_data_user = reviews_data_user[["business_id", "review_star"]].groupby('business_id')[
            'review_star'].mean().reset_index()
        # print(reviews_data_user)

        # print(input_df)
        # Create a dictionary to store the mapping of business_id to review_star
        mapping = dict(zip(reviews_data_user['business_id'], reviews_data_user['review_star']))

        # Map the ratings from table A to the corresponding columns in table B
        for col in self.reviews_data_knn.columns:
            if col in mapping:
                # print("insert: ", col, mapping[col])
                input_dict[col] = mapping[col]
            else:
                input_dict[col] = 0

        # print(input_dict)
        # convert dictionary to dataframe
        input_df = pd.DataFrame(input_dict, index=[0])

        # print(input_df)

        # Get the K nearest neighbors of the user
        _, indices = self.knn_model.kneighbors(input_df, n_neighbors=100)
        # print(indices)
        # Get the top recommended restaurants from the neighbors
        recommended_restaurants = []
        for neighbor_index in indices[0]:
            neighbor_ratings = self.df_train_knn_final.iloc[neighbor_index]
            recommended_restaurants += list(neighbor_ratings[neighbor_ratings > 1].index)

        recommended_restaurants = list(set(recommended_restaurants))
        # print(recommended_restaurants)
        # Filter the reviews dataframe to only include the business IDs in the business_ids array
        # print(self.df_train_knn_final)
        filtered_reviews = self.reviews_data[self.reviews_data['business_id'].isin(recommended_restaurants)]

        # Group the filtered reviews dataframe by business_id and calculate the mean review_star for each group
        review_stars = filtered_reviews.groupby('business_id')['review_star'].mean().reset_index()
        # Perform a left join on the 'key' column
        df_merged = pd.merge(review_stars, self.business_data, on='business_id', how='left')
        filtered_df = df_merged[df_merged['categories'].str.contains('|'.join(self.categories))]

        df_result = filtered_df[['business_id', 'review_star']]

        sorted_data = df_result.sort_values('review_star', ascending=False)
        # print(sorted_data)
        sorted_data.set_index('business_id', inplace=True)
        knn_recs = sorted_data.head(100).to_dict()['review_star']

        return knn_recs

    def recc(self, user_id, date_range, top_n=10):
        # Assuming you have the KNN and K-means recommendation dictionaries stored as knn_recs and kmeans_recs respectively

        user = self.reviews_data.sample(n=1, replace=True)
        user_id = user[['user_id']].values.tolist()[0][0]
        # print(user_id)
        kmeans_recs = self.get_kmeans_recc(user_id)
        knn_recs = self.get_knn_recc(user_id)

        print(kmeans_recs)
        print(knn_recs)

        # Combine the two dictionaries into a single dictionary with the restaurant names as keys and the weighted average scores as values
        combined_recs = []
        for business_id in set(knn_recs.keys()) | set(kmeans_recs.keys()):
            knn_score = knn_recs.get(business_id, 0)
            kmeans_score = kmeans_recs.get(business_id, 0)
            weighted_avg = (knn_score + kmeans_score) / 2
            combined_recs.append({
                'business_id': business_id,
                'review_star': weighted_avg,
            })

        # Deduplicate the recommendations and sort by the weighted average score in descending order
        sorted_recs = sorted(combined_recs, key=lambda x: x['review_star'], reverse=True)

        # create an empty list to store the results
        results = []

        # loop through the ratings array
        # print(sorted_recs)
        for rating in sorted_recs:
            # look up the location and name for the business in the business dataframe
            row = self.business_data.loc[self.business_data['business_id'] == rating['business_id']]
            address = row['address'].iloc[0]
            name = row['name'].iloc[0]
            city = row['city'].iloc[0]
            state = row['state'].iloc[0]
            categories = row['categories'].iloc[0]
            stars = row['stars'].iloc[0]
            lat = row['latitude'].iloc[0]
            long = row['longitude'].iloc[0]

            # add the business_id, rating_star, location, and name to the results list
            results.append({
                'business_id': rating['business_id'],
                'review_star': rating['review_star'],
                'name': name,
                'address': str(address) + ' ' + str(city) + ' ' + str(state),
                'categories': categories,
                'stars': stars,
                'image': None,
                'location': str(lat) + ', ' + str(long)
            })

        # display the result
        # print(results)
        # Create a dictionary to hold the final recommendations for each meal
        all_of_recc = []
        for i in range(1, (date_range).days + 2):
            final_recs = {"breakfast": [], "lunch": [], "dinner": []}

            # Iterate over the sorted recommendations and assign them to the appropriate meal, up to a maximum of 2 recommendations per meal
            for i in range(len(results) - 1, -1, -1):
                # for rest in results:
                rest = results.pop(i)
                print(rest["name"])
                rest["image"] = [self.get_image(rest["name"])]

                if len(final_recs["breakfast"]) < 2:
                    final_recs["breakfast"].append(rest)
                elif len(final_recs["lunch"]) < 2:
                    final_recs["lunch"].append(rest)
                elif len(final_recs["dinner"]) < 2:
                    final_recs["dinner"].append(rest)
                else:
                    break

            all_of_recc.append(final_recs)
        print(all_of_recc)

        return all_of_recc


def check_images(list_images):
    # print(list_images)
    final_images = []
    default_image = "etl/attractions.png"

    if list_images == None:
        final_images.append(default_image)
    else:
        for im in list_images:
            if im is not None:
                final_images.append(im)
            else:
                final_images.append(default_image)

    # print(final_images)
    first_images = [open(i, "rb").read() for i in final_images]
    return first_images


def final_output(days, final):
    time = ['breakfast', 'lunch', 'dinner']
    fields = ['NAME', 'CATEGORY', 'LOCATION', 'RATING']
    recommendations = ['Recommendation 1:', 'Recommendation 2:']

    result = []

    result.append({'name': final})
    return result
