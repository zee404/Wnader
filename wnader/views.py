import json
import os
import random
import threading
from datetime import datetime, date

import django
import pandas as pd
import pyautogui
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404

from wnader.models import Customer, AttractionCategory, Cuisine, CustomerPlan, Amenities, Hotel, Restaurant, Attraction
from .helpers import validate_password, get_attr_recommendation, get_res_recommendation, get_hot_recommendation

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wnader.settings")
django.setup()


def index(request):
    attractions = AttractionCategory.objects.all()
    attraction_categories = []
    for attraction in attractions:
        name = attraction.name.replace('_', ' ')
        attraction_categories.append({'id': attraction.name, 'name': name})
    cuisines = Cuisine.objects.all()
    amenities = Amenities.objects.all()

    # TOP HOTELS
    top_hotels = Hotel.objects.order_by('-rating')[:4]
    top_restaurant = Restaurant.objects.order_by('-review_star')[:4]
    top_attractions = Attraction.objects.order_by('-rating')[:4]

    context = {'attraction_categories': attraction_categories, 'cuisines': cuisines, 'amenities': amenities,
               'top_hotels': top_hotels, 'top_restaurant': top_restaurant, 'top_attractions': top_attractions}

    return render(request, 'home.html', context=context)



@login_required(login_url='/signin/')
def save_plan(request):
    user = get_user(request)
    if request.method == "POST":
        # ---------------------- GETTING ALL THE REQUIRED VALUES FROM FORM
        destination = request.POST.get('destination')
        begin_date_str = request.POST.get('begin_date')
        end_date_str = request.POST.get('end_date')
        people = request.POST.get('people')
        min_budget = float(request.POST.get('min_budget'))
        max_budget = float(request.POST.get('max_budget'))
        selected_attractions = request.POST.get('selected_attractions')
        selected_cuisines = request.POST.get('selected_cuisines')
        selected_amenities = request.POST.get('selected_amenities')
        plan_attractions = request.POST.get('plan_attractions')
        plan_restaurants = json.loads(request.POST.get('plan_restaurants'))
        plan_hotels = json.loads(request.POST.get('plan_hotels'))

        begin_date = datetime.strptime(begin_date_str, '%b. %d, %Y').date()
        end_date = datetime.strptime(end_date_str, '%b. %d, %Y').date()

        # ----------------------- GETTING THE CURRENT LOGGED IN CUSTOMER
        customer = Customer.objects.get(user=user)
        # ----------------------- CREATE A PLAN OBJECT AND PASS THE REQUIRED VALUES
        customer_plan = CustomerPlan.objects.create(customer=customer, destination=destination, start_date=begin_date,
                                                    end_date=end_date, min_budget=min_budget, max_budget=max_budget,
                                                    people=people, attractions=selected_attractions,
                                                    cuisines=selected_cuisines,
                                                    amenities=selected_amenities,
                                                    attraction_detail=plan_attractions,
                                                    restaurant_detail=json.dumps(plan_restaurants),
                                                    hotel_detail=json.dumps(plan_hotels)
                                                    )
        customer_plan.save()

    context = {'destination': destination, 'begin_date': begin_date, 'end_date': end_date, 'people': people,
               'min_budget': min_budget, 'max_budget': max_budget, plan_attractions
               : 'plan_attractions', 'plan_restaurants': plan_restaurants, 'selected_cuisines': selected_cuisines,
               'selected_attractions': selected_attractions}

    return redirect('profile')


def remove_plan(request, plan_id):
    plan = get_object_or_404(CustomerPlan, id=plan_id, customer=request.user.customer)
    if request.method == 'POST':
        plan.delete()
    return redirect('profile')


def signup(request):
    if request.method == "POST":
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('user_email')
        password = request.POST.get('user_password')
        if User.objects.filter(username__iexact=email).exists():
            messages.error(request, 'This email address is already registered.')
            # return render(request, 'signup.html', )
            return redirect('signup')
        else:
            pass_validated = validate_password(password)
            if pass_validated:
                messages.error(request, pass_validated)
                return redirect('signup')
            else:
                user = User.objects.create_user(username=email, first_name=first_name, last_name=last_name, email=email,
                                                password=password)
                user.save()
                customer = Customer.objects.create(user=user, email=email)
                customer.save()
                loguser = authenticate(username=email, password=password)

                if loguser is not None:
                    login(request, loguser)
                    return redirect('index')

    userdata = Customer.objects.last()
    context = {'userdata': userdata}

    return render(request, 'signup.html', context=context)


def signin(request):

    if request.method == "POST":
        username = request.POST['login_email']
        password = request.POST['login_password']
        print("inside post in  login form", username, password)
        user = authenticate(username=username, password=password)

        print(user)
        if user is not None:
            login(request, user)
            if customer_exist(user):
                return redirect('index')
        else:
            messages.error(request, 'Invalid Credentials.')
            return redirect('signin')

    return render(request, 'signin.html')


def logout_user(request):
    print("inside logout ")
    logout(request)
    return redirect('signin')

@login_required(login_url='/signin/')
def profile(request):
    user = get_user(request)
    if user.is_authenticated:
        if user.is_superuser:
            return redirect('logout')
        else:
            customer = Customer.objects.get(user=user)
            plans = CustomerPlan.objects.filter(customer=customer)

            last_plan = CustomerPlan.objects.filter(customer=customer).order_by('-created_at').first()
            attractions = json.loads(last_plan.attractions)
            cuisines = last_plan.cuisines.split(';')
            print(cuisines)
            context = {'plans': plans, 'last_plan': last_plan, 'attractions': attractions, 'cuisines': cuisines}
            print(context)

    return render(request, 'profile.html', context=context)


def import_data(request):
    att_df = pd.read_json('models/etl/attractions.json', orient='records')
    category_df = att_df.groupby('category').size().reset_index().sort_values([0], ascending=False)[:20]
    categories = list(category_df.category)
    # for c in categories:
    #     category = c.replace('_', ' ')
    #     att_ctg = AttractionCategory(
    #         name=category,
    #     )
    #     att_ctg.save()

    context = {'categories': categories}
    return JsonResponse(context)

    # Open the Parquet file
    # parquet_file = pq.ParquetFile('data5.parquet')
    #
    # dataframe = parquet_file.read().to_pandas()

    # for index, row in dataframe.iterrows():
    #     attraction = Attraction(
    #         name=row['name'],
    #         city=row['city'],
    #         province=row['province'],
    #         country=row['country'],
    #         location=row['location'],
    #         rating=row['rating'],
    #         price=row['price'],
    #         category=row['category'],
    #     )
    #     attraction.save()


def customer_exist(user):
    return Customer.objects.filter(user__exact=user).exists()


def generate_plan(request):
    isuser = get_user(request)
    if isuser:
        user = isuser.first_name
    else:
        user = 'guest'
    if request.method == "POST":
        destination = request.POST.get('destination')
        plan_date = request.POST.get('daterange')
        people = request.POST.get('people')
        budget = request.POST.get('budget')
        attractions = json.loads(request.POST.get('attraction_dict'))
        preferences = request.POST.getlist('cuisines')
        amenities = request.POST.getlist('amenities')
        start_date_str, end_date_str = plan_date.split(" - ")
        begin_date = datetime.strptime(start_date_str, "%m/%d/%Y").date()
        end_date = datetime.strptime(end_date_str, "%m/%d/%Y").date()

        low, high = map(float, budget.split(';'))

        attractions_with_rating = {}
        for item in attractions:
            attraction = item['attraction']
            rating = float(item['rating'])
            attractions_with_rating[attraction] = rating
        attraction_recommendation = []
        restaurant_recommendation = []
        hotel_recommendation = []

        # Define two functions to run the models in separate threads
        def get_attractions():
            nonlocal attraction_recommendation
            attraction_recommendation = get_attr_recommendation(begin_date, end_date, low, high, destination,
                                                                attractions_with_rating)

        def get_restaurants():
            nonlocal restaurant_recommendation
            restaurant_recommendation = get_res_recommendation(user, begin_date, end_date, preferences)

        def get_hotels():
            nonlocal hotel_recommendation
            hotel_recommendation = get_hot_recommendation(user, destination, begin_date, end_date, amenities)

        # Create two threads and start them
        t1 = threading.Thread(target=get_attractions)
        t2 = threading.Thread(target=get_restaurants)
        t3 = threading.Thread(target=get_hotels)
        t1.start()
        t2.start()
        t3.start()

        # Wait for the threads to finish before rendering the template
        t1.join()
        t2.join()
        t3.join()

    context = {'destination': destination, 'begin_date': begin_date, 'end_date': end_date,
               'people': people, 'attractions': attractions_with_rating, 'attractions_json': json.dumps(attractions),
               'cuisines': preferences,
               'amenities': amenities,
               'cuisines_json': '; '.join(preferences),
               'amenities_json': '; '.join(amenities),
               'res_json': json.dumps(restaurant_recommendation),
               'attr_json': json.dumps(attraction_recommendation), 'hotel_json': json.dumps(hotel_recommendation),
               'restaurant_recc': restaurant_recommendation,
               'hotel_recc': hotel_recommendation,
               'attraction_recc': attraction_recommendation, 'low': low, 'high': high}
    return render(request, 'plan_detail.html', context=context)



