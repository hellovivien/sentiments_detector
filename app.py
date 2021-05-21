import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, _analyze
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import texthero as hero
from texthero import stopwords
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from page import Page
import requests
import seaborn as sns
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score
import nltk
from sklearn.linear_model import LogisticRegression
# nltk.download('words')
import gensim
import neattext.functions as nfx
from textblob import TextBlob, Word
import dexplot as dxp
from collections import Counter
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
st.set_option('deprecation.showPyplotGlobalUse', False)
import torch
torch.set_default_tensor_type('torch.FloatTensor')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pathlib


LOAD_MODEL = True

# @st.cache(allow_output_mutation=True)
def load_data():
    kaggle = pd.read_csv('Emotion_final.csv', nrows = None)
    kaggle.rename(columns={"Emotion": "emotion", "Text": "content"}, inplace=True)
    kaggle.emotion = kaggle.emotion.apply(lambda x: "happiness" if x == "happy" else x)
    dataworld = pd.read_csv('text_emotion.csv', nrows = None)
    dataworld.drop(columns=["tweet_id", "author"], inplace=True)
    dataworld.rename(columns={"sentiment": "emotion"}, inplace=True)
    kaggle['sentiment'] = kaggle.content.apply(get_sentiment)
    dataworld['sentiment'] = dataworld.content.apply(get_sentiment)
    kaggle['clean_content'] = clean(kaggle.content)
    kaggle['clean_content'] = remove_stopwords(kaggle.clean_content)
    dataworld['clean_content'] = clean(dataworld.content)
    dataworld['clean_content'] = remove_stopwords(dataworld.clean_content)
    return kaggle, dataworld


def wordcloud_generator(data, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data.values))
    st.write(title)
    st.image(wordcloud.to_image())


def extract_keywords(text,num=50):
    tokens = [token for token in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return pd.DataFrame(most_common_tokens, columns=('token','count'))

def plot_keywords(df,title='Title'):
    fig = plt.figure(figsize=(40,20))
    plt.title(title)
    sns.set(font_scale=1.1)
    sns.barplot(y='token',x='count',data=df,orient="h")
#     plt.xticks(rotation=45)
    st.pyplot()
    
    
def remove_stopwords(content):
    custom_stopwords = ("feeling","feel","becaus","want","time","realli","im","think","thing","ive","still","littl","one","life","peopl","need","bit","even","much","dont","look","way","love","start","s","m","quot","work",
    "get","http","go","day", "com","got","see" "4pm","<BIAS>","veri","know","t","like","someth")
    return hero.remove_stopwords(content, spacy_stop_words.union(custom_stopwords))
    
def clean(content):
    import neattext.functions as nfx
    cleaning_steps = ('clean_text','remove_stopwords','remove_userhandles','remove_punctuations')
    for step in cleaning_steps:
        content = content.apply(getattr(nfx, step))
    content = hero.remove_diacritics(content)
    content = hero.remove_urls(content)
    content = hero.preprocessing.remove_digits(content)
    # content = hero.remove_punctuation(content)
    # content = hero.remove_whitespace(content)
    # content = hero.preprocessing.stem(content)
    return content

def predict_emotion(X,model):
    prediction = model.predict(X)
    return dict(zip(model.classes_, model.predict_proba(X)[0]))

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'


def page_data():
    st.markdown("### Analyse du Dataset {}".format(current_dataset_name))
    st.markdown("#### Distribution")
    emotion_names = current_dataset['emotion'].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.countplot(x='emotion',data=current_dataset,order=current_dataset.emotion.value_counts().index)
    st.pyplot(fig)
    pd.DataFrame(current_dataset.emotion.value_counts()).T
    st.write("Certaines émotions ne disposent pas d'assez de données et risquent de diminuer la pertinence de notre modèle.")

    fig, ax = plt.subplots(figsize=(12, 12))
    # dxp.count('emotion', data=current_dataset, split='sentiment', normalize='emotion')
    sns.catplot(data=current_dataset,x='emotion',hue='sentiment',kind='count',size=7,aspect=1.5)
    st.pyplot()
    st.write("Les sentiments detectés ne correspondent pas toujours aux émotions associés. Notemment les émotions négatives comme la colère, la tristesse ou l'inquiètude sont perçus aussi bien de manière positive, négative ou neutre.")


    NUM_TOP_WORDS = 20
    top_20_before = hero.visualization.top_words(current_dataset['content']).head(NUM_TOP_WORDS)
    top_20_after = hero.visualization.top_words(current_dataset['clean_content']).head(NUM_TOP_WORDS)

    fig, ax = plt.subplots(figsize=(12, 12))
    top_20_before.plot.bar(rot=90)
    ax.set_title('Top 20 words before cleaning')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    top_20_after.plot.bar(rot=90)
    ax.set_title('Top 20 words after cleaning')
    st.pyplot(fig)


    
    st.markdown("#### Nuage de mots")

    for emotion in emotion_names:
        wordcloud_generator(current_dataset.query("emotion == '{}'".format(emotion)).clean_content, title=emotion)
        corpus = current_dataset.query("emotion == '{}'".format(emotion)).clean_content.tolist()
        corpus = ' '.join(corpus)
        keywords = extract_keywords(corpus)
        plot_keywords(keywords)

def delete_pseudo(txt):
    '''
    delete the pseudo starting with @ in content
    :param txt: content(string)
    :return: the "clean_content" without pseudo (string)
    '''
    return ' '.join(word for word in txt.split(' ') if not word.startswith('@'))


def lemmatize_text(text):
    '''
    lemmatization of text
    :param text: string
    :return: lemmatize and tokenize text (list)
    '''
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]



def cleaning_text(df_name):
    '''
    All the steps of preprocessing
    :param df_name: name of the df on wich the content column must be preprocessed
    :return: a "clea_content" column
    '''
    # delete pseudo strating with @
    df_name['clean_content'] = df_name['content'].apply(delete_pseudo)
    # method clean from texthero
    df_name['clean_content'] = hero.clean(df_name['clean_content'])
    # delete stopwords with texthero
    default_stopwords = stopwords.DEFAULT
    custom_stopwords = default_stopwords.union(
        set(["feel", "feeling", "im", "get", "http", "ive", "go", "day", "com", "got", "see" "4pm"]))
    df_name['clean_content'] = hero.remove_stopwords(df_name['clean_content'], custom_stopwords)
    # remove urls
    df_name['clean_content'] = hero.remove_urls(df_name['clean_content'])
    # remove angle brakets
    df_name['clean_content'] = hero.remove_angle_brackets(df_name['clean_content'])
    # remove digits
    df_name['clean_content'] = hero.preprocessing.remove_digits(df_name['clean_content'], only_blocks=False)
    # lemmatisation
    # df_name['clean_content'] = df_name['clean_content'].apply(lemmatize_text)

# @st.cache(allow_output_mutation=True)
def build_tabnet():
        model_file_name = 'tabnet_model_{}'.format(current_dataset_name)
        
        df = current_dataset.copy()
        cleaning_text(df)

        X = df['clean_content']
        y = df['emotion']
        # tokenize la data
        tok = Tokenizer(num_words=1000, oov_token='<UNK>')
        # fit le model avec les données de train
        # tok.fit_on_texts(X)
        # X = tok.texts_to_matrix(X, mode='tfidf')
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=1)
        X_test_save = X_test
        tok.fit_on_texts(X_test)
        X_test = tok.texts_to_matrix(X_test, mode='tfidf')
        tok.fit_on_texts(X_train)
        X_train = tok.texts_to_matrix(X_train, mode='tfidf')
        # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y)
        # build model, fit and predict
        model = TabNetClassifier()
        # if LOAD_MODEL and pathlib.Path('{}.zip'.format(model_file_name)).exists():
        #     model.load_model('{}.zip'.format(model_file_name))
        # else:
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy', 'balanced_accuracy', 'logloss']
        )
        

        preds_mapper = {idx: class_name for idx, class_name in enumerate(model.classes_)}
        preds = model.predict_proba(X_test)
        y_pred_proba = np.vectorize(preds_mapper.get)(np.argmax(preds, axis=1))
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
        # model.save_model(model_file_name)
        return model, y_test, y_pred, test_acc


def make_tabnet():
        model, y_test, y_pred, test_acc = build_tabnet()
        st.text('Model Report:\n ' + classification_report(y_test, y_pred))
        st.write(f"BEST VALID SCORE FOR {current_dataset_name} : {model.best_cost}")
        st.write(f"FINAL TEST SCORE FOR {current_dataset_name} : {test_acc}")
        plt.plot(model.history['train_accuracy'], label="train_accuracy")
        plt.plot(model.history['valid_accuracy'], label="valid_accuracy")
        plt.legend()

        st.pyplot(plt)

        

# @st.cache(allow_output_mutation=True)
def build_model(df, model, model_name):
    ml = df.copy()
    X = ml.clean_content
    y = ml.emotion
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    # cv_results = cross_val_score(model, X, y, cv=skf, scoring='f1_micro')
    if model_name != "tabnet":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english")
        X_train_tf = tfidf.fit_transform(X_train)
        model.fit(X_train_tf, y_train)
        X_test_tf = tfidf.transform(X_test)
        y_pred = model.predict(X_test_tf)
        return model, X_test_tf, y_test, y_pred, tfidf


        
def make_svc():
    ml("Linear Support Vector", model=LinearSVC())
def make_nb():
    ml("Naive Bayes", model=MultinomialNB())
def make_log():
    ml("Logistic Regression", model=LogisticRegression())
def make_rf():
    ml("Random Forest", model=RandomForestClassifier())
def make_knn():
    ml("KNN", model=KNeighborsClassifier())
def make_lgbm():
    ml("Linear SVC", model=LGBMRegressor())

def ml(title, model):
    st.markdown("### Machine Learning : {}".format(title))
    model, X_test_tf, y_test, y_pred, _ = build_model(current_dataset, model, title.lower())
    st.write('Accuracy Score - {}'.format(accuracy_score(y_test, y_pred)))
    st.write('Recall Score (macro) - {}'.format(recall_score(y_test, y_pred,average='macro')))
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_confusion_matrix(model,X_test_tf,
                        y_test,
                        normalize='true',
                        cmap=plt.cm.Greens,
                        ax=ax)
    st.pyplot(plt)
    st.text('Model Report:\n ' + classification_report(y_test, y_pred))
    if current_dataset_name == "Kaggle":
        st.markdown("""
        La metric la plus important est le **recall** puisque on cherche à minimiser l'érreur sur l'ensemble de notre jeu de données et pas juste sur une partir de nos prédictions,
        autrement dit bien identifier *happy* a autant d'importance que bien identifier *love* ou n'importe quel autre émotion.
        On observe que notre modèle a d'assez bons résultats pour détecter les émotions *happy* et *sadness* mais que l'émotion *love* est souvent identifier comme *happy* et *surprise* comme *fear* ou *happy*. 
        Cela vient du fait que d'une part nous avons plus de données pour les émotions happy et sadness et que d'autres part 
        il est difficile plus difficile de différencier des émotions qui peuvent être plus ambigue comme la surprise qui peut aussi évoqué la joie et la peur.
        """)
    else:
        st.markdown("""
        Le jeu de données doit être nettoyé davantage
        """)


def page_search():
    search_input = st.text_input('Tell me something...', '')
    model, X_test_tf, y_test, y_pred, tfidf = build_model(kaggle, LinearSVC(), "test")
    if len(search_input)>0:
        search_tf = tfidf.transform([search_input])
        predictions = model.predict(search_tf)
        emotion = predictions[0]
        st.write(emotion)
        response = requests.get("https://api.giphy.com/v1/gifs/random?api_key=u5zI8PiTKx0y7b6Csh5GmUdhgD0hZ315&tag={}&rating=g".format(emotion))
        image_url = response.json()["data"]["image_original_url"]
        st.image(image_url)



###
# MAIN
###
st.sidebar.markdown("### 🤖 Emotions Detector")
start_time = time.time()
kaggle, dataworld = load_data()
app = Page()
app.add_page("Detector", page_search)
app.add_page("TabNet", make_tabnet)
app.add_page("Data Analyse", page_data)
app.add_page("Linear Support Vector", make_svc)
app.add_page("Naive Bayes", make_nb)
app.add_page("Logistic Regression", make_log)
app.add_page("KNN", make_knn)
app.add_page("Random Forest", make_rf)

current_dataset_name = st.sidebar.radio('Data',("Dataworld","Kaggle"))
if current_dataset_name == "Kaggle":
    current_dataset = kaggle
else:
    current_dataset = dataworld
app.run()



















# #PRINT EXEC TIME
# def write_exec_time(message=None):
#     exec_time = time.time()-start_time
#     if message is not None:
#         st.write(message)
#     st.write("Temps d'execution {}".format(exec_time))

# #FUNC TO PRINT EACH FOOD PRODUCT IN STREAMLIT
# def show_results(row):
#     details = {
#         # "product_name":"nom",
#         "score":"score",
#         "brands_tags":"marque",
#         "stores":"point de vente",
#         "pnns_groups_2":"categorie",
#         # "nutriscore_grade":"nutriscore",
#         "ecoscore_grade_fr":"ecoscore",
#     }
#     cols = st.beta_columns(2)
#     if row.image_small_url:
#         cols[0].image(row.image_small_url)
#     cols[1].markdown("**[{}]({})**".format(row.product_name, row.url))
#     cols[1].image("images/nutri{}.png".format(row.nutriscore_grade.upper()), width=100)
#     for field, label in details.items():
#         if field in row:
#             cols[1].markdown("**{}**: {}".format(label, row[field]))
#     st.markdown("------------------------")

# @st.cache
# def load_clean():
#     return pd.read_csv('clean.csv', nrows = None)

# def create_soup(df, search_cols):
#     items = []
#     for col in search_cols:
#         items.append(df[col])
#     return  ps.stem(' '.join(items))

# def search(input, df, model, soup_matrix):
#     result_num = max_results
#     res_elt = st.empty()
#     time_elt = st.empty()
#     with st.spinner('Chargement...'):
#         if model_name == "bert":
#             input_list = model.encode(input, convert_to_tensor=True)
#             cosine_sim = util.pytorch_cos_sim(input_list, soup_matrix)[0]
#             sim_scores = torch.topk(cosine_sim, k=result_num)
#             food_indices = sim_scores[1].tolist()
#             score_values = sim_scores[0].tolist()
#         else:
#             input_list = model.transform([input])
#             cosine_sim = cosine_similarity(soup_matrix, input_list)
#             if(np.count_nonzero(cosine_sim)==0):
#                 return None
#             sim_scores = list(enumerate(cosine_sim))
#             # Sort the foods based on the cosine similarity scores
#             sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#             # Get the scores of the 10 most similar foods. Ignore the first food.
#             # sim_scores = sim_scores[0:result_num]
#             food_indices = [item[0] for item in sim_scores]
#             score_values = [item[1][0] for item in sim_scores]
#         res = df.iloc[food_indices]
#         res["score"] = [math.ceil(score*100) for score in score_values]
#         res = res.query("score >= @min_score")
#         return filter(res)

# def filter(res):
#     if nutriscore.lower() in ["a","b","c","d","e"]:
#         res = res.query("nutriscore_grade == @nutriscore.lower()")
#     if res.shape[0] > max_results:
#         res = res.iloc[0:max_results]
#     return res



# @st.cache(allow_output_mutation=True)
# def build_model(use_count=False):
#     url_cols = ['url','image_small_url','image_url','image_ingredients_small_url','image_ingredients_url','image_nutrition_url','image_nutrition_small_url']
#     search_cols = ["product_name", "pnns_groups_2", "brands_tags","stores"]
#     res_cols = search_cols + ["nutriscore_grade", "ecoscore_grade_fr", "image_small_url", "image_nutrition_small_url", "url"]
#     df = load_clean()
#     df = df.reset_index(drop=True)
#     df.fillna('',inplace=True)
#     soup = df.apply(lambda x: create_soup(x, search_cols), axis=1)
#     soup = soup.to_list()
#     if model_name == "bert":
#         model = SentenceTransformer('paraphrase-distilroberta-base-v1')
#         soup_matrix = model.encode(soup, convert_to_tensor=True)
#         return df, model, soup_matrix
#     elif model_name == "Count":
#         model = CountVectorizer(stop_words=STOPWORDS)
#     else:
#         model = TfidfVectorizer(stop_words=STOPWORDS)
#     model = model.fit(soup)
#     soup_matrix = model.transform(soup)
#     return df, model, soup_matrix




# input_wrap = st.sidebar.empty()
# search_input = input_wrap.text_input('Search', '')
# # search_input = text_input('Search', 'kiwiz')
# model_names = ['Count', 'Tfidf']
# if CUDA:
#     model_names.append('BERT')
# model_name = st.sidebar.selectbox('Model name',model_names)
# max_results = st.sidebar.slider('Max results', 1, 100, 10)
# min_score = st.sidebar.slider('Min score', 0, 100, 1)
# nutriscore = st.sidebar.selectbox('Nutriscore',('Tous', 'A', 'B', 'C', 'D', 'E'))

# df, model, soup_matrix = build_model(model_name)
# DF_SIZE = df.shape[0]
# keywords = []


# #PAGE
# st.markdown("### 🥝 Foodflix")
# st.markdown("""
#     Cette application permet d'obtenir des recommandations alimentaires parmis **{}**
#     aliments grâce aux données d'[Open Food Facts](https://fr.openfoodfacts.org/data).
# """.format(DF_SIZE))
# if model_name != "BERT":
#     keywords = model.get_feature_names()
#     st.markdown("Notre modèle contient **{}** mots clés.".format(len(keywords)))
# res = None
# if len(search_input) > 0:
#     res = search(search_input, df, model, soup_matrix)
#     if res is None:
#         if len(keywords) > 0:
#             keywords_match = [word for word in keywords if (search_input[0] == word[0] and len(word)>2)]
#             alternatives = process.extract(search_input, keywords_match, limit=10)
#             alternatives = [alt[0] for alt in alternatives if alt[1]>=75]
#             if len(alternatives) > 0:
#                 st.markdown("Vous avez recherché **{}** mais nous n'avons rien trouvé 😥".format(search_input))
#                 alt = st.selectbox("Je pense que vous vouliez dire:", alternatives)
#                 res = search(alt, df, model, soup_matrix)
#                 res.apply(lambda x: show_results(x), axis=1)
#             else:
#                 st.write("Aucun résultat votre recherche est vraiment étrange...")
#     else:
#         res.apply(lambda x: show_results(x), axis=1)
#     write_exec_time()
# else:
#     filter(df).apply(lambda x: show_results(x), axis=1)
