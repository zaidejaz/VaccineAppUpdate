from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.urls import reverse_lazy
from .models import Account,LikeDislike
from .forms import UploadForm, UserRegisterForm, UserUpdateForm
from django.views.generic import TemplateView
from django.core.exceptions import ValidationError
from django.contrib.auth.decorators import login_required
import requests
from bs4 import BeautifulSoup as bs
from django.contrib import messages
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render
from .forms import UploadPostForm
from .models import Record
# views.py
from .models import Post
from user import models
from django.contrib.auth.models import User


from .models import *

# Create your views here.
def upload_form(request):
    if request.method == 'POST':
        form=UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        else:
            context ={'form': form}
            return render(request, 'user/uploadaudio.html',context)
    context={'form':UploadForm()}
    return  render(request, 'user/uploadaudio.html',context)
def upload_profile_pic(request):
    if request.method == 'POST':
        # Get the account object for the current user
        account = Account.objects.get(id=request.user.id)

        # Get the uploaded image from the request
        uploaded_image = request.FILES.get('image')

        # Update the image field of the account object
        account.image = uploaded_image
        account.save()

        # Redirect to some success page or view after successful upload
        return redirect('success_page')  # Replace 'success_page' with the name of your success page or view
    else:
        # Render the HTML form
        return render(request, 'user/accounts/upload_profile_pic.html')
    
def profile(request, id):
    # Get the user data
    user_data = User.objects.get(id=id)

    # Get the user's posts
    user_posts = Post.objects.filter(author=user_data)

    # Calculate the follower count and following count
    follower_count = Follower.objects.filter(user=user_data).count()
    following_count = Follower.objects.filter(follower=user_data).count()

    # Get the followers and follows (if needed, for future use)
    followers = Follower.objects.filter(user=user_data).select_related('follower')
    follows = Follower.objects.filter(follower=user_data).select_related('user')

    context = {
        'user_data': user_data,
        'user_posts': user_posts,
        'followers': followers,
        'follows': follows,
        'follower_count': follower_count,
        'following_count': following_count,
    }

    return render(request, 'user/accounts/profile.html', context)

@login_required
def follow_user(request, user_id):
    # Get the user to follow
    user_to_follow = get_object_or_404(User, id=user_id)

    # Check if the current user is already following the user to follow
    existing_follower = Follower.objects.filter(user=user_to_follow, follower=request.user).exists()

    if not existing_follower:
        # Create a new Follower object if not already following
        Follower.objects.create(user=user_to_follow, follower=request.user)
        # Optionally, you can add a message indicating the follow action was successful

    # Redirect back to the previous page
    # Retrieve the referrer URL from the request headers
    referrer_url = request.META.get('HTTP_REFERER')

    # Redirect the user to the referrer URL if available; otherwise, fallback to a default URL
    if referrer_url:
        return redirect(referrer_url)
    else:
        # Fallback to a default URL if HTTP_REFERER is not available
        return redirect('profile', id=user_id)

from django.shortcuts import render
from haystack.query import SearchQuerySet
from .models import Post
from django.views.generic import (
    ListView,DetailView,
    CreateView,
    UpdateView,
    DeleteView)


class PostListView(ListView):
    model = Post
    template_name = 'user/homepage.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']

class PostDetailView(DetailView):
    model = Post


class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['title', 'file']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def validate_file_extension(value):
        import os
        ext = os.path.splitext(value.name)[1]
        valid_extensions = ['.wav', '.mp3', '.docx']
        if not ext in valid_extensions:
            raise ValidationError(u'File not supported!')


class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    form_class = UploadPostForm  # Specify the form class for updating the Post
    template_name = 'user/post_update.html'  # Specify the template name

    def form_valid(self, form):
        # Automatically set the author to the current user
        form.instance.author = self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        # Only allow the author of the post to update it
        return self.request.user == post.author

    def get_success_url(self):
        # Get the referring URL from the request's headers
        referrer_url = f'/profile/{self.request.user.id}'
        return referrer_url

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    template_name = 'user/post_confirm_delete.html'  # Path to the confirmation template

    def get_success_url(self):
        # Get the referring URL from the request's headers
        referrer_url = f'/profile/{self.request.user.id}'
        return referrer_url

    def test_func(self):
        post = self.get_object()
        # Check if the request user is the author of the post
        return self.request.user == post.author
    

@login_required
def record(request):
    if request.method == "POST":
        # Ensure audio data is correctly received
        audio_file = request.FILES.get('audio')
        juza = request.POST.get('juz')
        surah = request.POST.get('surah')
        reader = request.POST.get('reader')    
        content = request.POST.get('content')    
        author = request.user  # Assuming you have user authentication
        
        # Create a new Post object
        post = Post(juza=juza, surah=surah, reader=reader, content=content, file=audio_file, author=author)
        post.save()
        # Optionally, return a success response
        return JsonResponse({'url': '/homepage/'})
    context = {"page_title": "Record audio"}
    return render(request, "user/record.html", context)


def record_detail(request, id):
    record = get_object_or_404(Record, id=id)
    context = {
        "page_title": "Recorded audio detail",
        "record": record,
    }
    return render(request, "user/record_detail.html", context)


def audio(request):
    records = Record.objects.all()
    context = {"page_title": "Voice records", "records": records}
    return render(request, "user/audio.html", context)

def userfeed(request):
    return render(request, 'user/userfeed.html', {'title': '  Home Page'})
def userfeed(request):
    return render(request, 'accounts/userfeed.html', {'title': '  Home Page'})


def start(request):
    return render(request, 'user/start.html', {'title': ' Start Page'})

def about(request):
    return render(request, 'user/about.html', {'title': 'About'})
def login(request):
    return render(request, 'user/login.html', {'title': 'Login'})
def indexsocial(request):
    return render(request, 'user/indexsocial.html', {'title': 'Login'})
def upload(request):
    return HttpResponse('<h1>Upload View </h1>')
class Home(TemplateView):
    template_name = "home.html"

def index(request):
    context = {'redirect_to': request.path}

    return render(request, 'index3.html',context), {'title': 'Search'}

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Your Account Has Been Created!')
            return redirect('site-user-login')



    else:
        form = UserRegisterForm()
    return render(request, 'user/register.html', {'form': form})



posts = [
    {
        'author': 'CoreyMS',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'August 27, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'August 28, 2018'
    }
]


@login_required
def homepage(request):
    # Query all posts and annotate with like and dislike counts
    all_posts = Post.objects.annotate(
        like_count=models.Count('likes_dislikes', filter=models.Q(likes_dislikes__like=True)),
        dislike_count=models.Count('likes_dislikes', filter=models.Q(likes_dislikes__dislike=True))
    )
    # Prepare context to pass to the template
    context = {
        'title': 'Homepage',
        'posts': all_posts, 
    }
    
    # Render the template with the context data
    return render(request, 'user/homepage.html', context)
    

@login_required
def editacc(request):
    if request.method == 'POST':

     u_form= UserUpdateForm(request.POST,instance=request.user)
     if u_form.is_valid():
         u_form.save()
         messages.success(request, f'Your Account Has Been Updated!')
         return redirect('site-homepage')
    else:
        u_form = UserUpdateForm( instance=request.user)

    context={
        'u_form':u_form
    }

    return render(request, 'user/editacc.html', {'u_form':u_form})



@login_required
def logout(request):
    return render(request, 'user/logout.html', {'title': 'Logout'})


def passreset(request):
    return render(request, 'user/passreset.html', {'title': 'Password Reset'})


def passreset2(request):
    return render(request, 'user/passreset2.html', {'title': 'Password Reset'})

def search(request):
    if request.method == 'POST':
        search = request.POST['search']
        url = 'https://www.ask.com/web?q='+search
        # url = 'https://www.google.com/search?q='+search
        res = requests.get(url)
        soup = bs(res.text, 'lxml')

        result_listings = soup.find_all('div', {'class': 'PartialSearchResults-item'})
        # result_listings = soup.find_all('div', {'class': 'PartialSearchResults-item'})

        final_result = []

        for result in result_listings:
            result_title = result.find(class_='PartialSearchResults-item-title').text
            result_url = result.find('a').get('href')
            result_desc = result.find(class_='PartialSearchResults-item-abstract').text

            final_result.append((result_title, result_url, result_desc))

        context = {
            'final_result': final_result
        }

        return render(request, 'user/search.html', context)
    else:
        return render(request, 'user/search.html')
    
@login_required
def post_form(request):
    if request.method == 'POST':
        juza = request.POST.get('juz')
        surah = request.POST.get('surah')
        reader = request.POST.get('reader')
        audio_file = request.FILES.get('audio')
        content = request.POST.get('content')
        author = request.user  # Assuming you have user authentication
        
        # Create a new Post object
        post = Post(juza=juza, surah=surah, reader=reader, file=audio_file, content=content, author=author)
        post.save()
        return redirect('site-homepage')
    else:
        form = UploadPostForm()
    return render(request, 'user/post_form.html', {'form': form})


@login_required
def searchpage(request):
    query = request.GET.get('q', '')
    results = Post.objects.filter(surah__icontains=query) | Post.objects.filter(reader__icontains=query) | Post.objects.filter(author__username__icontains=query)
    return render(request, 'user/searchpage.html', {'results': results, 'query': query, 'user': request.user})

@login_required
def like_post(request, post_id):
    if request.method == 'POST':
        post = get_object_or_404(Post, pk=post_id)
        # Check if the user has already liked the post
        existing_like = LikeDislike.objects.filter(post=post, user=request.user, like=True).exists()
        if not existing_like:
            # Create a LikeDislike object only if the user hasn't already liked the post
            LikeDislike.objects.create(post=post, user=request.user, like=True)
            like_count = LikeDislike.objects.filter(post=post, like=True).count()
            return JsonResponse({'success': 'Post liked successfully', 'like_count': like_count})
        else:
            return JsonResponse({'error': 'You have already liked this post'})
    else:
        return JsonResponse({'error': 'Invalid request'})

@login_required
def dislike_post(request, post_id):
    if request.method == 'POST':
        post = get_object_or_404(Post, pk=post_id)
        # Check if the user has already disliked the post
        existing_dislike = LikeDislike.objects.filter(post=post, user=request.user, dislike=True).exists()
        if not existing_dislike:
            # Create a LikeDislike object only if the user hasn't already disliked the post
            LikeDislike.objects.create(post=post, user=request.user, dislike=True)
            dislike_count = LikeDislike.objects.filter(post=post, dislike=True).count()
            return JsonResponse({'success': 'Post Disliked successfully', 'dislike_count': dislike_count})
        else:
            return JsonResponse({'error': 'You have already Disliked this post'})
    else:
        return JsonResponse({'error': 'Invalid request'})

# 2. load Recommendations
from django.shortcuts import render
import h5py
from user.models import Post
import random

@login_required
def recommendation_view(request):
    user_id = request.GET.get('user_id', None)
    items = Post.objects.all()

    # Path to your HDF5 file containing the recommendations
    file_path = 'user/models/hybrid_recommendations.h5'

    try:
        # Open the HDF5 file and load the dataset
        with h5py.File(file_path, 'r') as file:
            # Check if the user is in the dataset and the item data exists
            if 'user_data' in file and user_id in file['user_data'] and 'item_data' in file and items in file['item_data']:
                user_predictions = file['user_data'][user_id][:]
                item_related = file['item_data'][items][:]
                recommended_items = process_predictions(user_predictions)
            else:
                # Fallback to random items for new users or if the item_id is not found
                recommended_items = get_random_items()

        # Render a template with the recommended items
        return render(request, 'user/recommendations.html', {'recommended_items': recommended_items})

    except Exception as e:
        # Handle exceptions or errors in loading or processing
        return render(request, 'error.html', {'message': str(e)})


def process_predictions(predictions):
    # Example: Return the top 10 items; this is highly simplified
    top_indices = predictions.argsort()[-10:][::-1]
    return top_indices  # Replace with your method to map indices to actual items

  

def get_random_items():
    all_posts = list(Post.objects.all())
    random.shuffle(all_posts)
    # Select the top 10 randomly shuffled posts
    top_random_posts = all_posts[:10]
    return top_random_posts

#Following posts
@login_required
def my_following(request):
    # Get the current user
    current_user = request.user
    
    # Query the user IDs the current user is following
    following_users_ids = Follower.objects.filter(follower=current_user).values_list('user', flat=True)
    
    # Check if following_users_ids is not empty before using it in a queryset
    if following_users_ids:
        # Query the User objects that the current user is following
        following_users = User.objects.filter(id__in=following_users_ids)
        
        # Filter posts where the author is in the list of followed users
        posts = Post.objects.filter(author__in=following_users)
    else:
        # If following_users_ids is empty, set posts to an empty queryset
        posts = Post.objects.none()
    
    return render(request, 'user/my_following.html', {'posts': posts})

def allposts(request):
    {'user': Account.objects.all(),
     'posts': posts
     }
    allPosts = Post.objects.annotate(
        like_count=models.Count('likes_dislikes', filter=models.Q(likes_dislikes__like=True)),
        dislike_count=models.Count('likes_dislikes', filter=models.Q(likes_dislikes__dislike=True))
    )
    return render(request, 'user/allposts.html', {'title': 'All Posts', "posts": allPosts})