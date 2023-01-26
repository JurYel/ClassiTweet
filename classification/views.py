from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from .forms import SentimentForm

import torch
from .apps import ClassificationConfig
from .utils import text_preprocess
from decimal import Decimal, ROUND_HALF_EVEN

# Create your views here.
def index(request):
    template = loader.get_template("index.html")
    
    context = {}
    return HttpResponse(template.render(context, request))

def analyze_page(request):
    template = loader.get_template("analyze.html")

    context = {}

    return HttpResponse(template.render(context, request))

def get_prediction(request):
    
    """ This function handles the review input and prediction of result """
    
    form = SentimentForm(request.POST)
    context = {}

    if request.method == "POST" :
        
        if form.is_valid():
            final_review = form['review'].value() #extracts the review to be analyzed'
        
            # Text preprocessing
            cleaned_tweet = text_preprocess(final_review)

            # Using Our TfidfVectorizer to transform the tokens from our Training Data
            tfidf_tweet = ClassificationConfig.tfidf.transform(cleaned_tweet).toarray()

            # Converting our vector into tensor
            tfidf_tweet = torch.from_numpy(tfidf_tweet).type(torch.float)
            
            # Using our model to predict the sentiment
            prediction = ClassificationConfig.model_2.forward(tfidf_tweet)
            predictions = torch.softmax(prediction, dim=1)
            prediction = torch.argmax(predictions, dim=1)
            prediction = prediction.detach().cpu().numpy()
            
            # Convert to a list rounded integers
            predictions = predictions[0].detach().cpu().numpy()
            predictions = list(predictions * 100)
            # predictions = [Decimal(x).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN) for x in predictions]
            predictions = ['{:.2f}'.format(round(x, 2)) for x in predictions] # Convert to string to keep 0s.
            classes = ['Negative', 'Neutral', 'Positive']

            predictions = zip(classes, predictions)

            if prediction[0] == 0:
                sentiment = "Negative"
            elif prediction[0] == 1:
                sentiment = "Neutral"
            else:
                sentiment = "Positive"
            
            context = {'result': sentiment, 'input': final_review, 'predictions': predictions}
            

    else:
        form = SentimentForm()

    context['form'] = form
    return render(request, 'analyze.html', context=context)



