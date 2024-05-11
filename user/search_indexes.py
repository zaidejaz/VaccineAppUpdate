from haystack import indexes
from .models import Post

class PostIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    juza = indexes.CharField(model_attr='juza')
    surah = indexes.CharField(model_attr='surah')
    reader = indexes.CharField(model_attr='reader')
    author = indexes.CharField(model_attr='author')

    def get_model(self):
        return Post

    def index_queryset(self, using=None):
        return self.get_model().objects.all()

post = PostIndex()

