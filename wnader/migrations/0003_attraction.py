# Generated by Django 3.2.18 on 2023-04-04 06:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wnader', '0002_remove_customer_full_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Attraction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(null=True)),
                ('city', models.CharField(max_length=50, null=True)),
                ('province', models.CharField(max_length=50, null=True)),
                ('country', models.CharField(max_length=50, null=True)),
                ('location', models.TextField(null=True)),
                ('price', models.FloatField(null=True)),
                ('rating', models.FloatField(null=True)),
                ('category', models.CharField(max_length=50, null=True)),
            ],
            options={
                'verbose_name': 'Attraction',
                'verbose_name_plural': 'Attractions',
            },
        ),
    ]
