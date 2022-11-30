import streamlit as st
import tweepy #twitter api
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from textblob import Word
from textblob import TextBlob
#from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import matplotlib.pyplot as plt


st.set_page_config(page_title = "Twitter Sentiment Analyzer", page_icon="🐤", layout = "wide")





#setting up twitter
consumerKey = 'S8iEJ1hMMBit7radc5FaPZkTQ'
consumerSecret = 'MvmrjGKV6TTbbAWpzqJtc6RuxyZEg9uwdYhdvWc5NEiucn2Gh6'
accessToken = '1575957976090820619-vhfKRHgBKBPVS0Y7KbSmC1LqZzUNJk'
accessTokenSecret = 'I6tU33RIVclvJxZG5H4nhDWCDzZgSrq3Dpl88b2r5mBtO'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    #Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
twitter_api = tweepy.API(auth)






#fetch tweet
def get_tweets():
    #user_name = 'tony'
    #posts = twitter_api.user_timeline(screen_name=user_name, count = 5, lang ="en", tweet_mode="extended")

    #df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['tweets'])
    #st.success(f'Getting the latest 5 form {user_name} ')
    #changing to lowercase
    #df["tweets"] = df["tweets"].str.lower()
    #st.write(df)

    # serching for trending and hashtags on twitter
    all_t = tweepy.Cursor(
                # TODO: Set up Premium search?
                twitter_api.search_tweets,
                q=user_name,
                lang="en",
                count=search_limt,
                include_entities=False,
            ).items(search_limt)
    df2 = pd.DataFrame([tweet.text for tweet in all_t], columns=['tweets'])
    df2["tweets"] = df2["tweets"].str.lower()
    # Remove user names from tweets
    text = []
    for words in df2.tweets:
        text.append(words)
    texts = []
    for x in range(len(text)):
        users_removed = " ".join(word for word in text[x].split() if not word.startswith(("@",".","http:","{link}",'rt', 'http')))
        texts.append(users_removed)
    clean_data = pd.DataFrame(texts,columns=["tweet"])
    # Remove  punctuation
    clean_data['tweet'] = clean_data.tweet.str.replace('[^\w\s]','')
    #Remove Special characters
    clean_data['tweet'] = clean_data.tweet.str.replace('#','')
    clean_data['tweet'] = clean_data.tweet.str.replace('@','')

   
    #st.write(tokenize(clean_data))
    lemma = WordNetLemmatizer()
    clean_data['lemmatization'] = clean_data.tweet.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
    pol = []
    for text in clean_data["lemmatization"]:
        pol.append(np.sum(TextBlob(text).polarity))
    clean_data['polarity'] = pol

    clean_data.loc[(clean_data["polarity"]>=0.4),"sentiment"] = "Positive emotion"
    clean_data.loc[(clean_data["polarity"]<0),"sentiment"] = "Negative emotion"
    clean_data.loc[(clean_data["polarity"].between(0,0.4,inclusive="left")),"sentiment"] = "Neutral emotion"
    clean_data.sentiment.value_counts()

    #tfidf = TfidfVectorizer()

    # Fit and transform vectorizer
    #data_to_model = tfidf.fit_transform(clean_data["lemmatization"])
    #filename = 'finalized_model.sav'
    #loaded_model = joblib.load(filename)
    #result = loaded_model.predict(data_to_model)
    

    return clean_data
    #st.write(result)

    

#tokenize
def tokenize(data):
    word_tokens = []

    for words in data["tweet"]:
        word_tokens.append(word_tokenize(words))
    return word_tokens


with st.container():
    #title page setup

    a, b = st.columns([1, 10])
    with a:
        st.text("")
        st.image("logoOfficial.png", width=50)
    with b:
        st.title("Twitter Sentiment Analyzer")
    url = "https://beast001-twitter-sentiment-analyzer-streamlit-app-pubhvw.streamlit.app/"
    st.write("Interested in analysing more? [link](%s)" % url)

    st.write("Enter any topic or trending tags to analys the sentiment of the users")

    st.write("")

    with st.container():
        st.write("---")
        left_column, right_column =  st.columns(2)
        with left_column:
            st.write("##")

            #start of form

            with st.form(key="my_form"):
                search_params = {}
                #a, b = st.columns([1, 1])              
                user_name = st.text_input("*Enter topic you want to analysie (with or without the #)*")
                search_limt = st.slider("Tweet limit", 1, 1000, 100)
                submit_button = st.form_submit_button(label="Submit")

            #end of form 
            if user_name != '':
                clean_df = get_tweets()
                st.write(clean_df['tweet'])
            else:
                st.warning('Get some tweets')
                clean_df = 'No tweets to analyze yet'
        with right_column:
            #right column page
            if isinstance(clean_df, pd.DataFrame):

                # Change values of I can't tell to positive, negative or no emotion 
                clean_df.loc[(clean_df["polarity"]>0),"sentiment"] = "Positive emotion"
                clean_df.loc[(clean_df["polarity"]==0),"sentiment"] = "Negative emotion"
                clean_df.loc[(clean_df["polarity"]<0),"sentiment"] = "Neutral emotion"

                # Visualize the target variable
                fig, ax = plt.subplots()
                labela = 'Neutral','Positive','Negative'
                explode = (0, 0.1,0)
                ax.pie(clean_df["sentiment"].value_counts(),explode=explode, labels=labela, autopct='%1.1f%%', shadow=True, startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
            else:
                st.write('Waiting for model to run!')
                
           

