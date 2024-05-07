from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('user', '0005_account_image'),
    ]

    operations = [
        migrations.RenameField(
            model_name='post',
            old_name='caption',
            new_name='content',
        ),
        migrations.RenameField(
            model_name='post',
            old_name='user',
            new_name='title',
        ),
        migrations.RemoveField(
            model_name='post',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='post',
            name='image',
        ),
        migrations.RemoveField(
            model_name='post',
            name='no_of_likes',
        ),
        migrations.AddField(
            model_name='post',
            name='author',
            field=models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='post',
            name='date_posted',
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='post',
            name='file',
            field=models.FileField(upload_to=''),  # Ensure you provide a proper upload_to path
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='post',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False),
        ),
    ]
