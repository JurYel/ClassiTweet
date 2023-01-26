from django import forms
from .models import SentimentModel

class SentimentForm(forms.Form):
    review = forms.CharField(max_length = 5000, widget=forms.Textarea(
        attrs = {
            'id': 'input-text',
            'placeholder': 'Input comment...',
        }
    ))

    class Meta:
        model = SentimentModel
        fields = [
            'review'
        ]