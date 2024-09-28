from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address = models.CharField(max_length=3000)
    gender = models.CharField(max_length=300)

class predict_firearms_monitoring(models.Model):

    Flowid= models.CharField(max_length=300)
    accused_name= models.CharField(max_length=300)
    date_of_incident= models.CharField(max_length=300)
    manner_of_killed= models.CharField(max_length=300)
    age= models.CharField(max_length=300)
    gender= models.CharField(max_length=300)
    race= models.CharField(max_length=300)
    licensename= models.CharField(max_length=300)
    street= models.CharField(max_length=300)
    city= models.CharField(max_length=300)
    flee= models.CharField(max_length=300)
    detected_by= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



