from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, User
from django.db import models
from django.utils import timezone
from django.urls.base import reverse
import uuid
import uuid
from datetime import datetime

class Upload(models.Model):
    name=models.CharField(max_length=1000)
    audio=models.FileField(upload_to='audio/')
juz_CHOICES = (
    ('1','1'),
    ('2', '2'),
('3', '3'),
('4', '4'),
('5', '5'),
('6', '6'),
('7', '7'),
('8', '8'),
('9', '9'),
('10', '10'),
('11', '11'),
)
surah_choices=(('Albqara',"Albqara"),)
shiekh_choices=(("mohamend","mohamend"),)
class Record(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    juza=models.CharField(max_length=100,blank=True,null=True)
    surah= models.CharField(max_length=100,blank=True,null=True)
    reader= models.CharField(max_length=100,blank=True,null=True)
    voice_record = models.FileField(upload_to="records")

    class Meta:
        verbose_name = "Record"
        verbose_name_plural = "Records"

    def __str__(self):
        return str(self.id)

    def get_absolute_url(self):
        return reverse("record_detail", kwargs={"id": str(self.id)})


class MyAccountManager (BaseUserManager):
    def create_user(self, email, username, password=None):
        if not email:
            raise ValueError("Users must have an email address.")
        if not username:
            raise ValueError("Users must have a username.")
        user = self.model(
            email=self.normalize_email(email),
            username=username),

        user.set_password(password)
        user.save(using = self._db)
        return user


class Account(AbstractBaseUser):
    email = models.EmailField(verbose_name="email", max_length=60, unique=True)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')
    username = models.CharField(max_length= 30, unique = True)
    date_joined = models.DateTimeField(verbose_name="date joined", auto_now_add = True)
    last_login = models.DateTimeField(verbose_name="last login", auto_now = True)
    is_admin = models.BooleanField(default = False)
    is_active = models.BooleanField(default = True)
    is_staff = models.BooleanField(default = False)
    is_superuser = models.BooleanField(default = False)
    hide_email=models.BooleanField(default = True)
    USERNAME_FIELD='email'
    REQUIRED_FIELDS = ['username']
    objects=MyAccountManager()

    def __str__(self):
        return self.username

    def __str__(self):
        return f'{self.user.username} Profile'

    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_Label):
        return True
class Follower(models.Model):
    user = models.ForeignKey(User, related_name='followers', on_delete=models.CASCADE)
    follower = models.ForeignKey(User, related_name='following', on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'follower')

def get_profile_image_filepath(self, filename):
    return f'profile_images/{self.pk}/'
class PastFile(models.Model):
    title = models.CharField(max_length=21)
    content = models.TextField()
    date_posted = models.DateTimeField(default = timezone.now)
    result=models.ImageField(max_length=255, upload_to=get_profile_image_filepath, null=False, blank=False)

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


class Post(models.Model):
    juza=models.CharField(max_length=100,blank=True,null=True)
    surah= models.CharField(max_length=100,blank=True,null=True)
    reader= models.CharField(max_length=100,blank=True,null=True)
    file = models.FileField(upload_to="posts")
    content = models.TextField(max_length=1000)
    date_posted = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)


    # def __str__(self):
    #     return self.juza,self.surah,self.reader
    
    def like_count(self):
        return self.likes_dislikes.filter(like=True).count()
    
    def dislike_count(self):
        return self.likes_dislikes.filter(dislike=True).count()
    
    # def get_absolute_url(self):
    #     return reverse('post-detail', kwargs={'pk': self.pk})

class LikeDislike(models.Model):
    post = models.ForeignKey('Post', related_name='likes_dislikes', on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    like = models.BooleanField(default=False)
    dislike = models.BooleanField(default=False)

class LikePost(models.Model):
    post_id = models.CharField(max_length=500)
    username = models.CharField(max_length=100)

    def __str__(self):
        return self.username

class FollowersCount(models.Model):
    follower = models.CharField(max_length=100)
    user = models.CharField(max_length=100)

    def __str__(self):
        return self.user


class Item(models.Model):
    name = models.CharField(max_length=100)
    # Other fields in your model

    def __str__(self):
        return self.name
