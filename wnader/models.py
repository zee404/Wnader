from django.contrib.auth.models import User
from django.db import models


class Customer(models.Model):
    email = models.EmailField(max_length=30)
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.user.first_name

    class Meta:
        verbose_name = "Customer"
        verbose_name_plural = "Customers"


class Amenities(models.Model):
    name = models.CharField(max_length=20)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Amenity"
        verbose_name_plural = "Amenities"


class AttractionCategory(models.Model):
    name = models.TextField(null=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "AttractionCategory"
        verbose_name_plural = "AttractionCategories"


class Cuisine(models.Model):
    name = models.TextField(null=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Cuisine"
        verbose_name_plural = "Cuisines"


class Attraction(models.Model):
    name = models.CharField(max_length=100)
    image = models.CharField(max_length=255)
    category = models.TextField(null=True)
    location = models.CharField(max_length=100)
    price = models.FloatField()
    rating = models.FloatField()

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Attraction"
        verbose_name_plural = "Attractions"


class Restaurant(models.Model):
    name = models.CharField(max_length=100)
    image = models.CharField(max_length=255)
    review_star = models.FloatField()
    address = models.CharField(max_length=100)
    categories = models.TextField()
    location = models.CharField(max_length=100)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Restaurant"
        verbose_name_plural = "Restaurants"


class Hotel(models.Model):
    name = models.CharField(max_length=100)
    image = models.CharField(max_length=255)
    address = models.CharField(max_length=100)
    amenities = models.TextField()
    location = models.CharField(max_length=100)
    price = models.FloatField()
    rating = models.FloatField()

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Hotel"
        verbose_name_plural = "Hotels"


class CustomerPlan(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    destination = models.CharField(max_length=50)
    attractions = models.JSONField()
    cuisines = models.TextField()
    amenities = models.TextField(null=True)
    attraction_detail = models.JSONField(null=True)
    restaurant_detail = models.JSONField(null=True)
    hotel_detail = models.JSONField(null=True)
    start_date = models.DateField()
    end_date = models.DateField()
    people = models.IntegerField(default=1)
    min_budget = models.FloatField(default=25)
    max_budget = models.FloatField(default=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.destination

    class Meta:
        verbose_name = "CustomerPlan"
        verbose_name_plural = "CustomerPlans"
