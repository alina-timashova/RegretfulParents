
import contractions
import nltk
import spacy
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

#хотя в BERT есть встроенная токенизация, предобработка текста дает более консистентные результаты
#для этих же целей чистим стоп-слова с помощью двух библиотек: nltk и spacy
def clean_words(post):
    tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
    lemmatizer = WordNetLemmatizer()
    text = contractions.fix(post)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english') and token not in spacy_stopwords]
    return ' '.join(lemmatized_words)

#при написании этой функции обращалась за помощью к ИИ, в первую очередь для правильного вывода результатов через print
def analyze_topics(df, text_column='post', parent_column='parent', nr_topics=7):
    
    #из таблицы берем только те строки, в которых определен один из родителей
    text_column_mother = df[df[parent_column].str.contains('Likely mother', na=False)][text_column]
    text_column_father= df[df[parent_column].str.contains('Likely father', na=False)][text_column]

    #предобработка текста
    preprocessed_texts_mother = [clean_words(post) for post in text_column_mother.tolist()]
    preprocessed_texts_father = [clean_words(post) for post in text_column_father.tolist()]

    #применяю модель BERT для тематического анализа; чтобы был виден прогресс, поставила параметр verbose=True
    representation_model = KeyBERTInspired()
    topic_model_mother = BERTopic(language='english', nr_topics=nr_topics, representation_model=representation_model, verbose=True)
    topic_model_father = BERTopic(language='english', nr_topics=nr_topics, representation_model=representation_model, verbose=True)
    
    topics_mother = topic_model_mother.fit_transform(preprocessed_texts_mother)
    topics_father = topic_model_father.fit_transform(preprocessed_texts_father)

    print('Topics for Mothers:')
    for topic_id in topic_model_mother.get_topics():
        print(f"Topic {topic_id}: {topic_model_mother.get_topic(topic_id)}")

    print('\nTopics for Fathers:')
    for topic_id in topic_model_father.get_topics():
        print(f"Topic {topic_id}: {topic_model_father.get_topic(topic_id)}")

    return {
        'mother_topics': topics_mother,
        'father_topics': topics_father
    }

df = pd.read_csv('regretful_parents_posts_gendered_expanded.csv')
results = analyze_topics(df, text_column='post', parent_column='parent', nr_topics=7)

mother_topics = results['mother_topics']
father_topics = results['father_topics']
