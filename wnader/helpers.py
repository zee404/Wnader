import pandas as pd

from models.attractions_recc import get_recc, filter_df, top_recc, find_closest, final_output
from models.hotel_recc import amenities_rating, model_train, get_user_data, get_hotel_recc, get_image
from models.res_recc import get_recomendation
from models.utils import Util


def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not any(char.isupper() for char in password):
        return "Password must contain at least one uppercase letter."
    if not any(char.islower() for char in password):
        return "Password must contain at least one lowercase letter."
    if not any(char.isdigit() for char in password):
        return "Password must contain at least one digit."
    return False


def get_res_recommendation(name, begin_date, end_date, preferences):
    gr = get_recomendation(preferences)
    result = gr.recc(None, date_range=end_date - begin_date)

    return result


def get_attr_recommendation(begin_date, end_date, low, high, destination, attraction_ratings):
    att_df = pd.read_json('models/etl/attractions.json', orient='records')
    filename, user, rbm_att = get_recc(att_df, attraction_ratings)
    with_url = filter_df(filename, user, low, high, destination, att_df)
    final = dict()
    final['timeofday'] = []
    final['image'] = []
    final['name'] = []
    final['location'] = []
    final['price'] = []
    final['rating'] = []
    final['category'] = []
    for i in range(1, (end_date - begin_date).days + 2):
        for j in range(2):
            final['timeofday'].append('Morning')
        for j in range(2):
            final['timeofday'].append('Evening')
    for i in range(len(final['timeofday'])):
        if i % 4 == 0:
            final = top_recc(with_url, final)
        else:
            final = find_closest(with_url, final['location'][-1], final['timeofday'][i], final)
    days = (end_date - begin_date).days + 1
    result = final_output(days, final)

    return result


def get_hot_recommendation(name, destination, begin_date, end_date, amenities):
    utils = Util()
    ## Reading file containing hotel details after removing duplicates
    del_dup = utils.read_newline_json("models/etl/del_dup/")

    ## Reading file containing hotel details after removing duplicates and exploding amenities
    newh_df = utils.read_newline_json("models/etl/newh_df/")

    usr_rating = amenities_rating(amenities, newh_df)
    _, _, hotel_ids = model_train(usr_rating, train=False)

    user_matrix, usrid_s2 = get_user_data(usr_rating)
    # print(usrid_s2)
    u_tempdf = get_hotel_recc(user_matrix, usrid_s2, hotel_ids)
    # u_tempdf.show()
    print(u_tempdf.head())
    hotel_df = pd.merge(del_dup, u_tempdf, on='id')
    hotel_df['address'] = hotel_df['address'].str.lower()

    user_location = destination

    hotel_sugg = hotel_df[hotel_df['country'].str.lower().str.contains(user_location)]
    recc = hotel_sugg.dropna()

    days = (end_date - begin_date).days + 1
    final = dict()
    final['address'] = recc[:days]['address'].values.tolist()
    final['amenities'] = recc[:days]['amenities'].values.T.tolist()
    final['experience'] = recc[:days]['hotel_experience'].values.tolist()
    final['name'] = recc[:days]['hotel_name'].values.tolist()
    final['rating'] = recc[:days]['hotel_rating'].values.tolist()
    final['location'] = [i[1:-1] for i in recc[:days]['location'].values.tolist()]
    final['price'] = recc[:days]['price'].values.tolist()
    final['image'] = [get_image(i) for i in recc[:days]['hotel_name'].values.tolist()]

    result = []
    # CREATING A JSON FOR EACH DAY FOR THE PLAN DAYS THAT CONTAIN INFORMATION FOR HOTEL
    for i in range(days):
        recom = {}
        recom['name'] = final['name'][i]
        recom['address'] = final['address'][i]
        recom['amenities'] = final['amenities'][i]
        recom['experience'] = final['experience'][i]
        recom['rating'] = final['rating'][i]
        recom['location'] = final['location'][i]
        recom['price'] = final['price'][i]
        recom['image'] = final['image'][i]
        result.append(recom)

    return result
