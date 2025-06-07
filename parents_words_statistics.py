import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from collections import Counter
import contractions
from textblob import TextBlob

def clean_words(post):
    tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
    lemmatizer = WordNetLemmatizer()
    text = contractions.fix(post.lower()) #разворачиваю слова по типу won't, can't, etc.
    tokens = tokenizer.tokenize(text)
    pos_tags = pos_tag(tokens)
    nouns = []
    
    #учитываю только существительные, так как они должны показать то, о чем именно жалеют родители
    for word, pos in pos_tags:
        if pos.startswith('NN'):
            lemmatized_words = lemmatizer.lemmatize(word, pos='n')
            nouns.append(lemmatized_words)
    return nouns

#чтобы уменьшить количество нерелевантных слов, я учитываю только предложения с негативными коннотациями
def negative_words(text):
    sentences = re.split(r'[\n\.\!\?]', str(text))
    negative_words = []

    for sentence in sentences:
        sentiment = TextBlob(sentence).sentiment.polarity
        if sentiment < 0:
            negative_words.extend(clean_words(sentence))
    return negative_words


def analyze_regrets(df, text_column='post', parent_column='parent', top_n=30):
    mother_posts = df[df[parent_column].str.contains('Likely mother', na=False)][text_column]
    father_posts = df[df[parent_column].str.contains('Likely father', na=False)][text_column]

    mother_words = []
    mother_negative_words = []
    for post in mother_posts:
        mother_words.extend(clean_words(post))
        mother_negative_words.extend(negative_words(post))

    father_words = []
    father_negative_words =[]
    for post in father_posts:
        father_words.extend(clean_words(post))
        father_negative_words.extend(negative_words(post))

    mother_freq = Counter(mother_negative_words)
    father_freq = Counter(father_negative_words)

    top_mother_words = mother_freq.most_common(top_n)
    top_father_words = father_freq.most_common(top_n)

    return top_mother_words, top_father_words

df = pd.read_csv('regretful_parents_posts_gendered_expanded.csv')
mother_results, father_results = analyze_regrets(df, 'post', 'parent')

print(mother_results)
print(father_results)


