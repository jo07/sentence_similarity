

# =============================================================================
# Objective : Find sentence similarity or inference 
# Requirements
# Glove data used : https://www.kaggle.com/devjyotichandra/glove6b50dtxt
# 
# Spacy packages
#   python -m spacy download en_core_web_sm (Not used)
#   python -m spacy download en_core_web_lg
# =============================================================================

# =============================================================================
# Cosine similarity
# =============================================================================
def cosine_distance_countvectorizer_method(s1, s2):
    
    # sentences to list
    allsentences = [s1 , s2]
    
    # packages
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.spatial import distance
    
    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1-cosine)*100,2)

# =============================================================================
# Glove embedding
# =============================================================================

gloveFile = "glove.6B.50d.txt"
import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

import re
from nltk.corpus import stopwords
import pandas as pd

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words



def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    vector_1 = np.mean([model[word] for word in preprocess(s1) if word in model],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2) if word in model],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return round((1-cosine)*100,2)

# =============================================================================
#  Spacy - Universal sentence encoders
# =============================================================================
    
import spacy
# this loads the wrapper
nlp_sg = spacy.load('en_core_web_sm')


nlp_lg = spacy.load('en_core_web_lg')

def encoder_similarity(s1, s2, encoder = nlp_sg):
    s1_doc = encoder(s1)
    s2_doc = encoder(s2)
    return s1_doc.similarity(s2_doc)

# =============================================================================
# Model ensembles similarity scores
# =============================================================================



def fetch_ensemble_scores(train_data):
    train_data['glove_score'] = train_data.apply(lambda x: cosine_distance_wordembedding_method(x['sentence1'], x['sentence2']) , axis=1)
    train_data['cosine_score'] = train_data.apply(lambda x: cosine_distance_countvectorizer_method(x['sentence1'], x['sentence2']) , axis=1)
    
    #train_data['en_core_web_sm_score'] = train_data.apply(lambda x: encoder_similarity(x['sentence1'], x['sentence2']) , axis=1)
    
    train_data['en_core_web_lg_score'] = train_data.apply(lambda x: encoder_similarity(x['sentence1'], x['sentence2'], nlp_lg) , axis=1)

    train_data = train_data.drop(['sentence1', 'sentence2'], axis = 1) 
    return train_data



# =============================================================================
# Train model
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load glove model
model = loadGloveModel(gloveFile)

train_file = './assignment_data_set/train.csv'
train_data = pd.read_csv(train_file)

data  =  fetch_ensemble_scores(train_data)
#data = data.drop(['sentence1', 'sentence2'], axis = 1) 

clf_model = RandomForestClassifier( n_estimators=100 )

labels = data['gold_label']
features = data.drop(['gold_label'], axis = 1) 




# Accuracy
print (np.mean(cross_val_score(clf_model, features, labels, cv=10)))


clf_model.fit(features, labels)


# =============================================================================
# Predicting Lables
# =============================================================================

test_file = './assignment_data_set/test.csv'
test_data = pd.read_csv(test_file)

test_data = fetch_ensemble_scores(test_data)

predicted_labels = clf_model.predict(test_data)
predicted_labels = list(predicted_labels)

# =============================================================================
# Save results
# =============================================================================
df = pd.DataFrame(data={"gold_label": predicted_labels})
df.to_csv("./predicted.csv", sep=',',index=False)
