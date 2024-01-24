import glob
import math
import os
import re

import nltk as nl
import pandas as pd
import tensorflow as tf
from bing_image_downloader import downloader
from nltk.corpus import wordnet as wn
from sklearn import preprocessing

from models.rbm.rbm_model import RBM
from models.utils import Util

nl.download('wordnet')


def f(row):
    avg_cat_rat = dict()
    for i in range(len(row['category'])):
        if row['category'][i] not in avg_cat_rat:
            avg_cat_rat[row['category'][i]] = [row['rating'][i]]
        else:
            avg_cat_rat[row['category'][i]].append(row['rating'][i])
    for key, value in avg_cat_rat.items():
        avg_cat_rat[key] = sum(value) / len(value)
    return avg_cat_rat


def sim_score(row):
    score = 0.0
    match = 0
    col1 = row['cat_rat']
    col2 = row['user_data']
    for key, value in col2.items():
        if key in col1:
            match += 1
            score += (value - col1[key]) ** 2
    if match != 0:
        return ((math.sqrt(score) / match) + (len(col2) - match))
    else:
        return 100


def calculate_scores(ratings, attractions, rec, user):
    '''
    Function to obtain recommendation scores for a user
    using the trained weights
    '''
    # print(rec.tolist()[0][0])
    # Creating recommendation score for books in our data
    ratings["Recommendation Score"] = rec.tolist()[0][0]

    """ Recommend User what books he has not read yet """
    # Find the mock user's user_id from the data
    #         cur_user_id = ratings[ratings['user_id']

    # Find all books the mock user has read before
    visited_places = ratings[ratings['user_id'] == user]['attraction_id']
    visited_places

    # converting the pandas series object into a list
    places_id = visited_places.tolist()

    # getting the book names and authors for the books already read by the user
    places_names = []
    places_categories = []
    places_prices = []
    for place in places_id:
        places_names.append(
            attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
        places_categories.append(
            attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
        places_prices.append(
            attractions[attractions['attraction_id'] == place]['price'].tolist()[0])

    # Find all books the mock user has 'not' read before using the to_read data
    unvisited = attractions[~attractions['attraction_id'].isin(places_id)]['attraction_id']
    unvisited_id = unvisited.tolist()

    # extract the ratings of all the unread books from ratings dataframe
    unseen_with_score = ratings[ratings['attraction_id'].isin(unvisited_id)]

    # grouping the unread data on book id and taking the mean of the recommendation scores for each book_id
    grouped_unseen = unseen_with_score.groupby('attraction_id', as_index=False)['Recommendation Score'].max()

    # getting the names and authors of the unread books
    unseen_places_names = []
    unseen_places_categories = []
    unseen_places_prices = []
    unseen_places_scores = []
    for place in grouped_unseen['attraction_id'].tolist():
        unseen_places_names.append(
            attractions[attractions['attraction_id'] == place]['name'].tolist()[0])
        unseen_places_categories.append(
            attractions[attractions['attraction_id'] == place]['category'].tolist()[0])
        unseen_places_prices.append(
            attractions[attractions['attraction_id'] == place]['price'].tolist()[0])
        unseen_places_scores.append(
            grouped_unseen[grouped_unseen['attraction_id'] == place]['Recommendation Score'].tolist()[0])

    # creating a data frame for unread books with their names, authors and recommendation scores
    unseen_places = pd.DataFrame({
        'att_id': grouped_unseen['attraction_id'].tolist(),
        'att_name': unseen_places_names,
        'att_cat': unseen_places_categories,
        'att_price': unseen_places_prices,
        'score': unseen_places_scores
    })

    # creating a data frame for read books with the names and authors
    seen_places = pd.DataFrame({
        'att_id': places_id,
        'att_name': places_names,
        'att_cat': places_categories,
        'att_price': places_prices
    })

    return unseen_places, seen_places


def export(unseen, seen, filename, user):
    '''
    Function to export the final result for a user into csv format
    '''
    # sort the result in descending order of the recommendation score
    sorted_result = unseen.sort_values(
        by='score', ascending=False)

    x = sorted_result[['score']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler((0, 5))
    x_scaled = min_max_scaler.fit_transform(x)

    sorted_result['score'] = x_scaled

    # exporting the read and unread books  with scores to csv files
    if not os.path.exists(filename):
        os.mkdir(filename)

    seen.to_csv(filename + '/user' + user + '_seen.csv')
    sorted_result.to_csv(filename + '/user' + user + '_unseen.csv')


def get_recc(att_df, cat_rating):
    util = Util()
    epochs = 50
    rows = 40000
    alpha = 0.01
    num_hid = 100
    batch_size = 16
    dir = 'models/etl/'
    ratings, attractions = util.read_data(dir)
    ratings = util.clean_subset(ratings, rows)
    rbm_att, train = util.preprocess(ratings)
    num_vis = len(ratings)

    joined = ratings.set_index('attraction_id').join(
        attractions[["attraction_id", "category"]].set_index("attraction_id")).reset_index('attraction_id')
    grouped = joined.groupby('user_id')
    category_df = grouped['category'].apply(list).reset_index()
    rating_df = grouped['rating'].apply(list).reset_index()
    cat_rat_df = category_df.set_index('user_id').join(rating_df.set_index('user_id'))
    cat_rat_df['cat_rat'] = cat_rat_df.apply(f, axis=1)
    cat_rat_df = cat_rat_df.reset_index()[['user_id', 'cat_rat']]

    cat_rat_df['user_data'] = [cat_rating for i in range(len(cat_rat_df))]
    cat_rat_df['sim_score'] = cat_rat_df.apply(sim_score, axis=1)
    user = cat_rat_df.sort_values(['sim_score']).values[0][0]

    print("Similar User: {u}".format(u=user))
    filename = "e" + str(epochs) + "_r" + str(rows) + "_lr" + str(alpha) + "_hu" + "_bs" + str(batch_size)

    rbm_model = RBM(num_vis, num_hid)

    checkpoint_path = "models/rbm/weight/rbm_weight_model"
    checkpoint = tf.train.Checkpoint(model=rbm_model)
    checkpoint.write(checkpoint_path)

    # You can then load the model using the following code:
    loaded_checkpoint = tf.train.Checkpoint(model=RBM(num_vis, num_hid))
    loaded_checkpoint.restore(checkpoint_path)
    rbm_model = loaded_checkpoint.model

    inputUser = [train[user]]

    reco = rbm_model.predict(tf.constant([inputUser], dtype=tf.float32))
    unseen, seen = calculate_scores(ratings, attractions, reco, user)
    export(unseen, seen, 'models/rbm/final_data/' + filename, str(user))
    return filename, user, rbm_att


def filter_df(filename, user, low, high, country, att_df):
    recc_df = pd.read_csv('models/rbm/final_data/' + filename + '/user{u}_unseen.csv'.format(u=user), index_col=0)
    recc_df.columns = ['attraction_id', 'att_name', 'att_cat', 'att_price', 'score']
    recommendation = att_df[
        ['attraction_id', 'name', 'category', 'city', 'latitude', 'longitude', 'price', 'country', 'rating']].set_index(
        'attraction_id').join(recc_df[['attraction_id', 'score']].set_index('attraction_id'),
                              how="inner").reset_index().sort_values("score", ascending=False)

    filtered = recommendation[
        (recommendation.country == country) & (recommendation.price >= low) & (recommendation.price >= low)]

    url = pd.read_json('models/etl/attractions_cat.json', orient='records')
    url['id'] = url.index
    with_url = filtered.set_index('attraction_id').join(url[['id', 'attraction']].set_index('id'), how="inner")
    return with_url


def get_image(name):
    # Assuming the Util class has the methods used below
    util = Util()
    name = name.replace("_", " ")
    # Sanitize 'name' to remove characters not allowed in Windows file/directory names
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "_")

    dir_path = "media/downloads"
    file_path = os.path.join(dir_path, name)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path) # Construct the file path where you expect the image to be

    # Try to find the file with the expected name in the directory
    file_exists = any(os.path.isfile(os.path.join(file_path, f)) for f in os.listdir(file_path) if f.endswith(('.png', '.jpg', '.jpeg')))

    # If the file does not exist, download it
    if not file_exists:
        try:
            cont = util.check_string(name)
            if cont:
                name = util.remove_special_characters(name)

            downloader.download(name, limit=1, output_dir=dir_path, adult_filter_off=True, force_replace=False, timeout=60)

            for filename in glob.glob(f"{file_path}/*jpg") + glob.glob(f"{file_path}/*png"):
                return filename
        except Exception as e:
            print(e)
            return None
    else:
        # Return the existing file path
        existing_files = glob.glob(f"{file_path}/*jpg") + glob.glob(f"{file_path}/*png")
        return existing_files[0] if existing_files else None

def top_recc(with_url, final):
    # print(with_url)
    i = 0
    while (1):
        first_recc = with_url.iloc[[i]]

        if (first_recc['name'].values.T[0] not in final['name']):
            final['name'].append(first_recc['name'].values.T[0])
            final['location'].append(first_recc[['latitude', 'longitude']].values.tolist()[0])
            final['price'].append(first_recc['price'].values.T[0])
            final['rating'].append(first_recc['rating'].values.T[0])
            final['image'].append(get_image(first_recc['name'].values.T[0]))
            final['category'].append(first_recc['category'].values.T[0])
            return final
        else:
            i += 1


def find_closest(with_url, loc, tod, final):
    syns1 = wn.synsets("evening")
    syns2 = wn.synsets("night")
    evening = [word.lemmas()[0].name() for word in syns1] + [word.lemmas()[0].name() for word in syns2]
    distance = list()
    for i in with_url[['latitude', 'longitude']].values.tolist():
        distance.append(math.sqrt((loc[0] - i[0]) ** 2 + (loc[1] - i[1]) ** 2))
    with_dist = with_url
    with_dist["distance"] = distance
    sorted_d = with_dist.sort_values(['distance', 'price'], ascending=['True', 'False'])
    if tod == "Evening":
        mask = sorted_d.name.apply(lambda x: any(j in x for j in evening))
    else:
        mask = sorted_d.name.apply(lambda x: all(j not in x for j in evening))
    # print(sorted_d[mask], final)
    final = top_recc(sorted_d[mask], final)
    return final


def final_output(days, final):
    time = ['Morning', 'Evening']
    fields = ['Name: ', 'Category: ', 'Location: ', 'Price: ', 'Rating: ']
    recommendations = ['Recommendation 1:', 'Recommendation 2:']

    result = []
    for i in range(days):

        # print(final['image'][i*4:(i+1)*4])

        # print(final['category'])
        # print(final['location'])
        # print(final['price'])
        # print(final['rating'])
        start_idx = i * 4
        end_idx = (i + 1) * 4
        images = final['image'][start_idx:end_idx]
        # print(images)

        final_images = []
        for i in images:
            image = "models/etl/attractions.png"
            if i is not None:
                image = i
            final_images.append(image)

        # images = [open(i, "rb").read() for i in final_images]

        name = [re.sub('_', ' ', i).capitalize() for i in final['name'][start_idx:end_idx]]

        category = [re.sub('_', ' ', i).capitalize() for i in final['category'][start_idx:end_idx]]
        location = [str(i[0]) + "," + str(i[1]) for i in final['location'][start_idx:end_idx]]
        price = [str(i) for i in final['price'][start_idx:end_idx]]
        rating = [str(i) for i in final['rating'][start_idx:end_idx]]

        result.append({'name': name, 'images': images, 'category': category, 'location': location, 'price': price,
                       'rating': rating})

    return result
