import re
import string
import spacy
import re
import numpy as np

def text_preprocess(message):
    
    result = message.lower()
    result = re.sub(r'\d+','',result)
    result = re.sub(r"won't", "will not",result)
    result = re.sub(r"would't", "would not",result)
    result = re.sub(r"could't", "could not",result)
    result = re.sub(r"\'d", " would",result)
    result = re.sub(r"can\'t", "can not",result)
    result = re.sub(r"n\'t", " not", result)
    result = re.sub(r"\'re", " are", result)
    result = re.sub(r"\'s", " is", result)
    result = re.sub(r"\'ll", " will", result)
    result = re.sub(r"\'t", " not", result)
    result = re.sub(r"\'ve", " have", result)
    result = re.sub(r"\'m", " am", result)

    result = result.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    
    result = result.strip()
    result = re.sub(' +',' ',result)
    result = result.replace('\n','')

    nlp = spacy.load("en_core_web_sm", disable = ['ner', 'parser', 'textcat'])

    text = nlp(result)
    lemmatized_text = " ".join([token.lemma_.lower() for token in text])
    text_array = np.array([lemmatized_text])

    return text_array