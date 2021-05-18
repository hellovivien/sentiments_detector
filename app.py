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
import texthero as hero
from texthero import stopwords
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from page import Page
import requests
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score

@st.cache
def load_data():
    kaggle = pd.read_csv('Emotion_final.csv', nrows = None)
    kaggle.rename(columns={"Emotion": "sentiment", "Text": "content"}, inplace=True)
    kaggle.sentiment = kaggle.sentiment.apply(lambda x: "happiness" if x == "happy" else x)
    dataworld = pd.read_csv('text_emotion.csv', nrows = None)
    dataworld.drop(columns=["tweet_id", "author"], inplace=True)
    return kaggle, dataworld


def wordcloud_generator(data, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data.values))
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    st.pyplot(plt)
    
    
def remove_stopwords(content):
    custom_stopwords = ("feel","becaus","want","time","realli","im","think","thing","ive","still","littl","one","life","peopl","need","bit","even","much","dont","look","way","love","start","s","m","quot","work",
    "get","http","go","day", "com","got","see" "4pm","<BIAS>","veri","know","t","like","someth")
    return hero.remove_stopwords(content, spacy_stop_words.union(custom_stopwords))
    
def clean(content):
    content = hero.remove_diacritics(content)
    content = hero.remove_urls(content)
    content = hero.preprocessing.remove_digits(content)
    content = hero.remove_punctuation(content)
    content = hero.remove_whitespace(content)
    content = hero.preprocessing.stem(content)
    return content


def page_data():
    st.markdown("### Analyse du Dataset {}".format(current_dataset_name))
    st.markdown("#### Distribution")
    sentiment_names = current_dataset['sentiment'].value_counts().index.tolist()
    # current_dataset['sentiment'].value_counts().sort_values(ascending=False).plot(kind='bar')
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.countplot(x="sentiment", data=current_dataset, palette="Set3", dodge=False,  order = current_dataset['sentiment'].value_counts().index)
    st.pyplot(fig)
    pd.DataFrame(current_dataset.sentiment.value_counts()).T
    NUM_TOP_WORDS = 20
    cloud = current_dataset.copy()
    cloud.content = clean(cloud.content)
    cloud.content = remove_stopwords(cloud.content)
    top_20_before = hero.visualization.top_words(current_dataset['content']).head(NUM_TOP_WORDS)
    top_20_after = hero.visualization.top_words(cloud['content']).head(NUM_TOP_WORDS)



    fig, ax = plt.subplots(figsize=(12, 12))
    top_20_before.plot.bar(rot=90)
    ax.set_title('Top 20 words before cleaning')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    top_20_after.plot.bar(rot=90)
    ax.set_title('Top 20 words after cleaning')
    st.pyplot(fig)


    
    st.markdown("#### Nuage de mots")

    for sentiment in sentiment_names:
        wordcloud_generator(cloud.query("sentiment == '{}'".format(sentiment)).content, title=sentiment)

@st.cache
def build_model_svc(df):
    ml = df.copy()
    ml.content = clean(ml.content)
    ml.content = remove_stopwords(ml.content)
    X = ml.content
    y = ml.sentiment
    model = LinearSVC()
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    # cv_results = cross_val_score(model, X, y, cv=skf, scoring='f1_micro')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english")
    X_train_tf = tfidf.fit_transform(X_train)
    model.fit(X_train_tf, y_train)
    X_test_tf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tf)
    return model, X_test_tf, y_test, y_pred, tfidf


def page_model_svc():
    st.markdown("### Modèle 1: TfidfVectorizer + LinearSVC")
    model, X_test_tf, y_test, y_pred, _ = build_model_svc(current_dataset)
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
        autrement dit bien identifier *happy* a autant d'importance que bien identifier *love* ou n'importe quel autre sentiment.
        On observe que notre modèle a d'assez bons résultats pour détecter les sentiments *happy* et *sadness* mais que le sentiment *love* est souvent identifier comme *happy* et *surprise* comme *fear* ou *happy*. 
        Cela vient du fait que d'une part nous avons plus de données pour les sentiments happy et sadness et que d'autres part 
        il est difficile plus difficile de différencier des sentiments qui peuvent être plus ambigue comme la surprise qui peut aussi évoqué la joie et la peur.
        """)
    else:
        st.markdown("""
        Le jeu de données doit être nettoyé davantage
        """)


def page_search():
    search_input = st.text_input('Tell me something...', '')
    model, X_test_tf, y_test, y_pred, tfidf = build_model_svc(kaggle)
    if len(search_input)>0:
        search_tf = tfidf.transform([search_input])
        predictions = model.predict(search_tf)
        sentiment = predictions[0]
        st.write(sentiment)
        response = requests.get("https://api.giphy.com/v1/gifs/random?api_key=u5zI8PiTKx0y7b6Csh5GmUdhgD0hZ315&tag={}&rating=g".format(sentiment))
        image_url = response.json()["data"]["image_original_url"]
        st.image(image_url)



###
# MAIN
###
st.sidebar.markdown("### 🤖 Sentiments Detector")
start_time = time.time()
kaggle, dataworld = load_data()
app = Page()
app.add_page("Detector", page_search)
app.add_page("Data Analyse", page_data)
app.add_page("Modèle 1 SVC", page_model_svc)
current_dataset_name = st.sidebar.radio('Data',("Kaggle","Dataworld"))
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
