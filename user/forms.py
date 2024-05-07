from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from user.models import Post
class UploadForm(forms.ModelForm):
    class Meta:
        model=Post
        fields = ('juza', 'file', 'content')


class UploadFileForms(forms.Form):
    file = forms.FileField()

class UserRegisterForm(UserCreationForm):
    email=forms.EmailField()
    class Meta:
        model=User
        fields= ['username','email','password1','password2']


class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email']


class UploadPostForm(forms.ModelForm):
    class Meta:
        model=Post
        fields = ('juza', 'file', 'content')