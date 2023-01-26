from django.apps import AppConfig
import torch
import joblib
from torch import nn
import os

NUM_CLASSES = 3
NUM_FEATURES = 5000

class BalancedNeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BalancedNeuralNetwork, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.hidden_layer_1 = nn.Linear(self.in_dim, 50) # input to first hidden layer, produces 50 features
        self.output_layer = nn.Linear(50, self.out_dim) # takes in 50 features, produces 3 feature (y)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer_1(x))
        y = self.output_layer(x)

        return y
    
class ClassificationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classification'

    cwd = os.getcwd()

    # load the torch model
    model_2 = BalancedNeuralNetwork(NUM_FEATURES, NUM_CLASSES)
    state = torch.load(os.path.join(cwd, 'classification\\imports\\model_2.pth'))
    model_2.load_state_dict(state['state_dict'])

    # Load the vectorizer
    tfidf = joblib.load(os.path.join(cwd, 'classification\\imports\\tfidf.sav'))
