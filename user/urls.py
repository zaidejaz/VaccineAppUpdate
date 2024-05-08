from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib import admin
from django.conf.urls.static import static
from .views import Home, PostListView, PostDetailView, PostCreateView, PostUpdateView, PostDeleteView, allposts, my_following, recommendation_view
from django.conf import settings
urlpatterns =[
    path('', views.search, name="search"),
    path('profile/<int:id>/', views.profile, name="profile"),
    
    path('follow/<int:user_id>/', views.follow_user, name='follow_user'),

    path('upload_profile_pic/', views.upload_profile_pic, name="upload_profile_pic"),
    path('searchpage/', views.searchpage, name="searchpage"),
    path("post/",views.upload_form, name="post"),
    path("post_form/",views.post_form, name="post_form"),
    path("indexsocial/",views.indexsocial, name="index"),
    path('search/', views.search, name='search'),
    path("feed/",views.userfeed, name="userfeed"),
    path("record/", views.record, name="record"),
    path("record/detail/<uuid:id>/", views.record_detail, name="record_detail"),
    path("audio/", views.audio, name="audio"),
    path("about/", views.about, name="about"),
    path('account/', views.start, name='account-base'),
    path("accounts/", include("allauth.urls")),  # new
    path("github", Home.as_view(), name="home"), # new
    path('register/', views.register, name='site-register'),
    path('upload', views.upload, name='upload'),
    path('homepage/', PostListView.as_view(), name='site-homepage'),
    path('post/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('post/new/', PostCreateView.as_view(), name='post-create'),
    path('post/<int:pk>/update/', PostUpdateView.as_view(), name='post-update'),
    path('post/<int:pk>/delete/', PostDeleteView.as_view(), name='post-delete'),
    path('homepage/editaccount/', views.editacc, name='site-editacc'),
    path('login/', auth_views.LoginView.as_view(template_name='user/login.html'), name='site-user-login'),
    path('logout/',  auth_views.LogoutView.as_view(template_name='user/logout.html'), name='site-user-logout'),
    path('password-reset/enter_email/', auth_views.PasswordResetView.as_view(template_name='user/passreset.html'), name='site-user-passwordreset'),
    path('password-reset/link_sent/', auth_views.PasswordResetDoneView.as_view(template_name='user/passreset2.html'), name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='user/password_reset_confirm.html'),
        name='password_reset_confirm'),
    path('password-reset-complete/', auth_views.PasswordResetCompleteView.as_view(template_name='user/passreset_complete.html'),
        name='password_reset_complete'),
    path('like-post/<int:post_id>/', views.like_post, name="likePost"),
    path('dislike-post/<int:post_id>/', views.dislike_post, name="disLikePost"),
    path('recommendations/', recommendation_view, name='recommendations'),
    path('following/', my_following, name='following'),
    path('allposts/', allposts, name='allposts'),
]


urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


