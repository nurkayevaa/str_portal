import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import string
import nltk
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
from bokeh.plotting import figure
import re
from nltk import word_tokenize
from sklearn import cluster
from sklearn import metrics
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache
def read_data():
    return pd.read_csv('ReviewDetails.csv')

def clean_text(text, tokenizer, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.

    Returns:
        Tokenized text.
    """
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features
    



def get_beer_reviews_vectorized(top_n = -1, ngram_range = (1,1), max_features = 1000):
    df = read_data()['review_text']
    df = df.dropna()   # drop any rows with empty reviews
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=max_features,
                                 min_df=2, stop_words=my_stop_words,
                                 ngram_range = ngram_range,
                                 use_idf=True)
    if (top_n >= 0):
        review_instances = df.values[0:top_n]
    else:
        review_instances = df.values
    
    X = vectorizer.fit_transform(review_instances) 
    
    return (X, vectorizer, review_instances)

def print_cluster_features(vectorizer, centroids, n_clusters, top_n_features):
    terms = vectorizer.get_feature_names()
    for i in range(n_clusters):
        print("Cluster %d:" % i, end='')
        for ind in centroids[i, :top_n_features]:
            print(' [%s]' % terms[ind], end='')
        print()




add_selectbox = st.sidebar.selectbox(
    "Page",
    ("Data",  "Positive reviews - Exporation", "Positive reviews - Examples", "Negative reviews - Exporation", "Negative reviews - Examples", "Conclusion")
)
if add_selectbox =="Data":
    st.title('What Amazon customers are saying about portal')
    df = read_data()
    st.text('\n\
        I would like to see what is the reason for good and bad reviews.\n\
    A supervised learning algorithm would be a good choice if we knew what are \n\
    the topics beforehands, for a mission of discovering reasons for reviews\n\
    unsupervised algorithm would be perfect. Facebook has 2 product lines Portal\n\
    and Oculous, these visualisation focuses on Portal, attached note book\n\
    and streamlit could be adopted to Oculus or any product line for thsat matter.\n\
    To see  the text of the review hover over the text\n\
    To simplify the grouping of classes I divided table into 2 parts negative and positives')
    st.dataframe(df[['review_rating','review_text']], width=1700)
# elif add_selectbox == "Preprocessing":
#     st.text('Context matters so teh text had to be converted to tokens\n\
#         It waslemmatise and stemming and converte dto vectors, WWord2vec\n\
#          was used, that methos worked better tahn tfidf tokenizer')
elif add_selectbox == "Positive reviews - Exporation":
    st.text('\n\ I am using an elbow method to find a good way to group classes, on the next slide it is possible to see example of the reviews , in the range 0 to 10')

    df = read_data()
    
    dfp = df[['review_rating','review_text']][df['review_rating']>3]

    nltk_tokens = nltk.word_tokenize(' '.join(list(df['review_text'])))


    my_stop_words = text.ENGLISH_STOP_WORDS.union(["portal", "product", "amazon", "like", "great", "good", "love", "awesome","family", "mother","grandmother", "parent", "child", "grandchildren"])

    dfp = dfp.dropna()   # drop any rows with empty reviews
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000,
                                 min_df=2, stop_words= my_stop_words,
                                 ngram_range = 3,
                                 use_idf=True)


    (X, vectorizer, review_instances) = get_beer_reviews_vectorized(5000, (1,2))
    


    df = read_data()

    dfp = df[['review_rating','review_text']][df['review_rating']>3]

    nltk_tokens = nltk.word_tokenize(' '.join(list(df['review_text'])))


    my_stop_words = text.ENGLISH_STOP_WORDS.union(["portal", "product", "amazon", "like", "great", "good", "love", "awesome","family", "mother","grandmother", "parent", "child", "grandchildren"])

    dfp = dfp.dropna()

    dfp["tokens"] = dfp["review_text"].map(lambda x: clean_text(x, word_tokenize, my_stop_words))

    model = Word2Vec(sentences=list(dfp['tokens']), vector_size=100, workers=1, seed=42)
    tokenized_docs = list(dfp['tokens'])

    vectorized_docs = vectorize(tokenized_docs, model=model)
    len(vectorized_docs), len(vectorized_docs[0])


    X = vectorized_docs
    X = np.array(X)



    inertias = []
    mapping1 = {}
    sqd = []
    K = range(1, 20)

    for k in K:
        #Building and fitting the model
        kmeanModel = KMeans(n_clusters=k, random_state =13).fit(X)
        kmeanModel.fit(X)
        sqd.append( kmeanModel.inertia_)
    #plt.plot(K, sqd)


    p = figure( title='Elbow method', x_axis_label='K', y_axis_label='sqd')
    p.line(K, sqd, legend_label='Trend', line_width=2)
    st.bokeh_chart(p)

 

elif add_selectbox == "Positive reviews - Examples":
    st.text(' Following are groups of reviews that I put as a topic myself, it is almost\n\
        look like a list of resons to buy portal :) \n\
        0 - general Positive Short \n\
        1 - General Positive Communication Family\n\
        2 - Gift \n\
        3 - Communication \n\
        4 - Audio Video\n\
        5 - Short incoherent   \n\
        6 - Communication for family members far apart \n\
        7  -Easy to use\n\
        8 - Picture Qualitys\n\
        9  - Alexa +Sound\n\
        10 - Incoherent Mix \n')
    df = read_data()

    dfp = df[['review_rating','review_text']][df['review_rating']>3]

    nltk_tokens = nltk.word_tokenize(' '.join(list(df['review_text'])))


    my_stop_words = text.ENGLISH_STOP_WORDS.union(["portal", "product", "amazon", "like", "great", "good", "love", "awesome","family", "mother","grandmother", "parent", "child", "grandchildren"])

    dfp = dfp.dropna()


    x = st.slider('Select the group number',  0, 10, 1)  # 👈 thi3 is a widget
 
    dfp["tokens"] = dfp["review_text"].map(lambda x: clean_text(x, word_tokenize, my_stop_words))

    model = Word2Vec(sentences=list(dfp['tokens']), vector_size=100, workers=1, seed=42)
    tokenized_docs = list(dfp['tokens'])

    vectorized_docs = vectorize(tokenized_docs, model=model)
    len(vectorized_docs), len(vectorized_docs[0])


    X = vectorized_docs
    X = np.array(X)



    kmeanModel = KMeans(n_clusters=11, random_state = 42).fit(X)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = list(dfp['review_text'])
    cluster_map['cluster'] = kmeanModel.labels_
    dff = cluster_map[cluster_map['cluster']==x]
    st.dataframe(dff, width=1700)
elif add_selectbox == "Negative reviews - Exporation":
    df = read_data()

    dfn = df[['review_rating','review_text']][df['review_rating']<3]

    nltk_tokens = nltk.word_tokenize(' '.join(list(df['review_text'])))


    my_stop_words = text.ENGLISH_STOP_WORDS.union(["portal", "product", "amazon", "like", "great", "good", "love", "awesome","family", "mother","grandmother", "parent", "child", "grandchildren"])

    dfn = dfn.dropna()   # drop any rows with empty reviews
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000,
                                 min_df=2, stop_words= my_stop_words,
                                 ngram_range = 3,
                                 use_idf=True)


    (X, vectorizer, review_instances) = get_beer_reviews_vectorized(5000, (1,2))
    


    df = read_data()

    dfn = df[['review_rating','review_text']][df['review_rating']<3]

    nltk_tokens = nltk.word_tokenize(' '.join(list(df['review_text'])))


    my_stop_words = text.ENGLISH_STOP_WORDS.union(["portal", "product",'amazon','parents','facebook','would','grandparents', 'even','family'])


    dfn = dfn.dropna()

    dfn["tokens"] = dfn["review_text"].map(lambda x: clean_text(x, word_tokenize, my_stop_words))

    model = Word2Vec(sentences=list(dfn['tokens']), vector_size=100, workers=1, seed=42)
    tokenized_docs = list(dfn['tokens'])

    vectorized_docs = vectorize(tokenized_docs, model=model)
    len(vectorized_docs), len(vectorized_docs[0])


    X = vectorized_docs
    X = np.array(X)


    inertias = []
    mapping1 = {}
    sqd = []
    K = range(1, 20)

    for k in K:
        #Building and fitting the model
        kmeanModel = KMeans(n_clusters=k, random_state =13).fit(X)
        kmeanModel.fit(X)
        sqd.append( kmeanModel.inertia_)
    #plt.plot(K, sqd)


    p = figure( title='Elbow method', x_axis_label='K', y_axis_label='sqd')
    p.line(K, sqd, legend_label='Trend', line_width=2)
    st.bokeh_chart(p)

 

elif add_selectbox == "Negative reviews - Examples":
    df = read_data()
    st.text('\n\
        0 - Extremly Dissatisfied incoherent\n\
        1 - connection \n\
        2 - highly  \n\
        3 - problems setting up \n\
        4 - problems with video \n\
        5 - mix privacy, AI camera \n\
        6 - Does not work at all \n\
        7 - Utility \n\
        8 - have to have fb account\n\
        9 - children - incoherent\n\
        10  -  1 review screen went out \n\
        11 - 1 review long various problems\n\
        12 - 1 review negative incoherent \n\
        13 - Quality - service, mix  \n')

    dfn = df[['review_rating','review_text']][df['review_rating']<3]

    nltk_tokens = nltk.word_tokenize(' '.join(list(df['review_text'])))


    my_stop_words = text.ENGLISH_STOP_WORDS.union(["portal", "product", "amazon", "like", "great", "good", "love", "awesome","family", "mother","grandmother", "parent", "child", "grandchildren"])

    dfn = dfn.dropna()


    x = st.slider('Select the group number',  0, 13, 1)  # 👈 thi3 is a widget
 
    dfn["tokens"] = dfn["review_text"].map(lambda x: clean_text(x, word_tokenize, my_stop_words))

    model = Word2Vec(sentences=list(dfn['tokens']), vector_size=100, workers=1, seed=42)
    tokenized_docs = list(dfn['tokens'])

    vectorized_docs = vectorize(tokenized_docs, model=model)
    len(vectorized_docs), len(vectorized_docs[0])


    X = vectorized_docs
    X = np.array(X)



    kmeanModel = KMeans(n_clusters=19, random_state= 42).fit(X)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = list(dfn['review_text'])
    cluster_map['cluster'] = kmeanModel.labels_
    dff = cluster_map[cluster_map['cluster']==x]
    st.dataframe(dff, width=1700)
elif add_selectbox =="Conclusion":
    st.text('\n\
     While it is very hard to find groups of reviews that are coherent\n\
     to a human undrstanding it is possible.\n\
     There are ways to improve the process by considering each review sentence\n\
     to be a review document.\n\
     While LDA , using clsutering after calculating Calinski Harabasz\n\
     and davies Bouldin scores , did not give coherent results.\n\
     Elbow methos showen in these slides seems to be the best\n\
     method to get a good clustering.')