import pandas as pd
import re
import json
import spacy 
from gensim.parsing.preprocessing import strip_tags, strip_non_alphanum, strip_multiple_whitespaces, strip_short

pd.set_option('display.max_colwidth', None)
nlp = spacy.load('en', disable=['ner', 'parser'])


""" load data """
dataset_review_meta = pd.read_csv('../data/amazon-cell-phone-review-meta.csv')
dataset_review = pd.read_csv('../data/amazon-cell-phone-reviews.csv')
dataset_review.columns = map(str.lower, dataset_review.columns)

with open('../data/contractions.json') as file:
    contraction_dict = {key.lower(): value.lower() for key, value in json.load(file).items()}


""" stage data """
dataset_review = dataset_review.dropna(subset=['body']) # drop empty or null rows
dataset_review = dataset_review.drop_duplicates('body') # drop duplicate rows
dataset_review = dataset_review.groupby('asin').filter(lambda x:len(x) > 780) # get items length over n amount
dataset_review = dataset_review.query('verified == True') # consider ony verified reviews
dataset_review = dataset_review[dataset_review['body'].str.split().str.len() > 5] # remove review length less than 5
df_cleansed_review = dataset_review[{'body', 'asin'}].rename(columns={'asin': 'item_id', 'body': 'review_text'}, errors='raise').copy().reset_index(drop=True)


""" transform data """
re_pattern = re.compile('({})'.format('|'.join(contraction_dict.keys())), flags=re.IGNORECASE)
def expand_contractions(text, pattern=re_pattern, contraction_map=contraction_dict):
    '''
    this function will expands the contraction of provided text by matching the pattern given
        text - sentence, phrase or word for expansion
        patter - regex pattern
        contraction_map - contraction mapping dictionary
    '''
    def replace(match):
        return contraction_map[match.group(0)]
    return pattern.sub(replace, text)

def preprocess(text):
    ''' 
    this function does simple text pre-processing such as, 
        - remove html tags
        - remove non-alphabetic 
        - remove punctuation
    '''
    step_process_text = strip_tags(str(text))
    step_process_text = strip_non_alphanum(step_process_text)
    # step_process_text = remove_stopwords(step_process_text)
    # step_process_text = strip_multiple_whitespaces(step_process_text)
    step_process_text = strip_short(step_process_text, minsize=5)
    processed_text = step_process_text.strip()
    return processed_text

def lemmatize(text):
    ''' 
    this function does text lemmatization on pre-processed text; 
        - remove stop words
        - lemmatizaation
    '''
    lemmatized_text = " ".join([token.lemma_ for token in nlp(text) if not token.is_stop])
    return lemmatized_text

df_cleansed_review['review_text_cont'] = df_cleansed_review['review_text'].str.lower().apply(lambda x: expand_contractions(x)) # expand contraction
preprocess_text = (preprocess(row) for row in df_cleansed_review['review_text_cont']) # preprocessing
lemmatize_text = [lemmatize(str(doc)) for doc in nlp.pipe(preprocess_text, batch_size=100, n_threads=-1)] # stop word and lemmatize