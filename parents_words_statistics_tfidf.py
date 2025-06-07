import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = ['hate', 'cannot', 'thing', 'feel', 'way', 'something', 'anything', 'everything',\
               'hour', 'year', 'day', 'month', 'week', 'mom', 'dad', 'people', 'mother', 'parent', 'kid', 'child', 'baby']

def clean_words(post):
    tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
    lemmatizer = WordNetLemmatizer()
    text = contractions.fix(post.lower()) #разворачиваю слова по типу won't, can't, etc.
    tokens = tokenizer.tokenize(text)
    pos_tags = pos_tag(tokens)
    nouns = []
    
    #учитываю только существительные, так как они должны показать то, о чем именно жалеют родители
    for word, pos in pos_tags:
        if pos.startswith('NN') and word not in stop_words:
            lemmatized_words = lemmatizer.lemmatize(word, pos='n')
            nouns.append(lemmatized_words)
    return nouns

#собираем мешок слов и применяем TF-IDF, чтобы вычленить только важные слова
def apply_tfidf_filtering(all_posts, top_n_words=100):
    texts = []
    for post in all_posts:
        words = clean_words(post)
        if words:
            texts.append(' '.join(words))
    
    vectorizer = TfidfVectorizer(max_features=top_n_words, ngram_range=(1, 2), stop_words=stop_words)
    vectorizer.fit_transform(texts)
    return set(vectorizer.get_feature_names_out())

def analyze_regrets(df, text_column='post', parent_column='parent', top_n=10):
    mother_posts = df[df[parent_column].str.contains('Likely mother', na=False)][text_column].tolist()
    father_posts = df[df[parent_column].str.contains('Likely father', na=False)][text_column].tolist()
    
    all_posts = mother_posts + father_posts
    allowed_words = apply_tfidf_filtering(all_posts, 100)
    
    mother_words = []
    for post in mother_posts:
        words = clean_words(post)
        filtered_words = [word for word in words if word in allowed_words]
        mother_words.extend(filtered_words)
    
    father_words = []
    for post in father_posts:
        words = clean_words(post)
        filtered_words = [word for word in words if word in allowed_words]
        father_words.extend(filtered_words)
    
    mother_freq = Counter(mother_words)
    father_freq = Counter(father_words)
    
    top_mother_words = mother_freq.most_common(top_n)
    top_father_words = father_freq.most_common(top_n)
    
    return top_mother_words, top_father_words

df = pd.read_csv('regretful_parents_posts_gendered_expanded.csv')
mother_results, father_results = analyze_regrets(df, 'post', 'parent')

print("Mother results:")
print(mother_results)
print("\nFather results:")
print(father_results)
