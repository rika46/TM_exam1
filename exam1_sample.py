#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                             #--------------#Packages#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#import packages
import requests  
import re  
import pandas as pd  

import numpy as np   
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#Seaborn is widely used for visualition
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
#install and import tweepy package, An easy-to-use Python library for accessing the Twitter API.
import tweepy as tw
#Authentication is handled by the tweepy.AuthHandler class
from tweepy import OAuthHandler

from nltk.corpus import stopwords
from nltk import trigrams
import string
import nltk
from nltk.stem.porter import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#PCA in sklearn package allows dimensionality reduction functionality
from sklearn.decomposition import PCA


from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                             #--------------#News API#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#decide on the topics and list it
topics=['coffee', 'dosa', 'pasta']

#creating a .csv file to store the articles 
filename="/Users/rika/Documents/TM/exam1/exam1.csv"
MyFILE2=open(filename,"w")
column_names="LABEL,Headline\n"
MyFILE2.write(column_names)
MyFILE2.close()
#itertate through for each topic
#endpoint of the API
endpoint="https://newsapi.org/v2/everything"
for topic in topics:
    #give in the api key and needed parameters
    URLPost = {'apiKey':'6209a3fea75e4b818c79ff9501155c00',
               'q':topic, 'searchln': "description"
    }
    #request for json
    response=requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    MyFILE2=open(filename, "a")
    LABEL=topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")
        Headline=items["description"]
        Headline=str(Headline)
        #remove words with numbers
        Headline=re.sub(r"[A-Za-z]+\d+|\d+[A-Za-z]+",'',Headline).strip()
        #remove numbers
        Headline=re.sub(r'\d', '', Headline)
        #replace commas and dots with empty string
        Headline=Headline.replace(',', '')
        Headline=Headline.replace('.', '')
        Headline=' '.join(Headline.split())
        #remove words with less than 4 letters and greater than 10 letters
        #Headline = ' '.join([wd for wd in Headline.split() if len(wd)>=4 and len(wd)<=10])
        print(Headline)
        #write in the column names for teh csv file
        WriteThis=str(LABEL)+"," + str(Headline) + "\n"
        print(WriteThis)
    
        MyFILE2.write(WriteThis)
    
## CLOSE THE FILE
MyFILE2.close()
news_df = pd.read_csv("/Users/rika/Documents/TM/exam1/exam1.csv")
tem_df = news_df.copy()
#news_df = news_df.loc[news_df['LABEL'].isin(["Bitcoin","Tether","Dogecoin", "polkadot"])]
#print(news_df.isnull().sum())
news_df['LABEL'] = news_df['LABEL'].apply(lambda x: x.lower())
print(news_df)
#news_df = news_df.drop("Unnamed: 0",axis=1)
news_df['Headline'].replace('', np.nan, inplace=True)
#!#
#############if there are nan in Headline##############
news_df = news_df.dropna()
#############if there are nan in Headline##############
#!#
print(news_df.info())

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                        #--------------#Corpus#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#path to the corpus folder
newpath = r'/Users/rika/Documents/TM/exam1/corpus'
if not os.path.exists(newpath):
    os.makedirs(newpath)
#endpoint of the API
endpoint="https://newsapi.org/v2/everything"
#itertate through for each topic
for topic in topics:
    i = 1
    #give in the api key and needed parameters
    URLPost = {'apiKey':'6209a3fea75e4b818c79ff9501155c00',
               'q':topic, 'pageSize':10, 'page':1
    }
    #request for json
    response=requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    LABEL=topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")
        Headline=items["description"]
        Headline=str(Headline)
        Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
        ## Be sure there are no commas in the headlines or it will
        ## write poorly to a csv file....
        Headline=Headline.replace(',', '')
        Headline=re.sub("\n|\r", "", Headline)
        Headline=Headline.replace('.', '')
        #remove words with less than 4 letters and greater than 10 letters
        Headline = ' '.join([wd for wd in Headline.split() if len(wd)>=4 and len(wd)<=10])
        Headline=' '.join(Headline.split())
        #create .txt file for each article
        MyFILE1 = open(f"{newpath}/{LABEL}{i}.txt","w")
        #increment i
        i += 1
        WriteThis=str(Headline) + "\n"
        print(WriteThis)
        MyFILE1.write(WriteThis)
        MyFILE1.close()
        
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                        #--------------#Cleaning df#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#removing patterns

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, ' ', input_txt)
    return input_txt   

# removing twitter handles (@user)
news_df['Headline'] = np.vectorize(remove_pattern)(news_df['Headline'], "@[\w]*#")
#removing urls    
news_df['Headline'] = np.vectorize(remove_pattern)(news_df['Headline'], "https?://[A-Za-z./]'*")

#removing special characters, numbers, punctuations
news_df['Headline'] = news_df['Headline'].str.replace("[^a-zA-Z]", " ")

news_df['Headline'] = news_df['Headline'].str.replace("https", "")
    #converting all the words to lowercase 
news_df['Headline'] = news_df['Headline'].str.lower()


#finally removing stopper words from the text column
stop = stopwords.words('english')+['the', 'go', 'get', 
                                   'see', 'take', 'thing', 'like', 'one', 'say','via','san']
news_df['Headline'] = news_df['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]) )
#removing words of length lesser than 3 and greater  than 10 
news_df['Headline'] = news_df['Headline'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3 and len(w)<15])) 


news_df["Headline"] = news_df["Headline"].astype(str).apply(lambda x: ''.join(x))

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                       #-------------#wordcloud#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

text = " ".join(i for i in news_df.Headline)
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure( figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Wordcloud before cleaning')
plt.show()

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #------------------#Stemmer#---------------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


Stem_df = news_df.copy(deep=True)
#removing all rows in the Stem_df DataFrame where the value in the Headline column is an empty string.
Stem_df = Stem_df[Stem_df.Headline!= '']
#creating a PorterStemmer object, which is a tool for stemming words.
A_STEMMER=PorterStemmer()
#performing stemming on each word in the Headline column of the Stem_df DataFrame.
Stem_df['Headline'] = Stem_df['Headline'].apply(lambda x: ' '.join([A_STEMMER.stem(word) for word in x.split()]))

print(Stem_df.head(10))
news_df = news_df[news_df.Headline!= '']


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #------------------#Lemmatization#---------------------#
#--------------#############--------------#--------------############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


lem_df = news_df.copy(deep=True)
lemmatizer = WordNetLemmatizer()    # Instantiate our Stemmer object
lem_df['Headline'] = lem_df['Headline'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
lem_df = lem_df[lem_df.Headline!= '']
print(lem_df.head(10))
news_df["Headline"].values.tolist()
news_df.columns.values.tolist()
print(news_df)

lem_df.to_csv(r'/Users/rika/Documents/TM/exam1/after_lem_exam1.csv',index=False)

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #-------------#countvectorizer#--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#countvectorize to create words frequency array
mycv1 = CountVectorizer(input = 'content', stop_words = 'english', max_features=100)
mymat = mycv1.fit_transform(lem_df['Headline'])
mycols = mycv1.get_feature_names()
mymat = mymat.toarray()
mydf=pd.DataFrame(mymat, columns = mycols)
print(mydf)
mydf.columns.values.tolist() 

mydf.to_csv(r'/Users/rika/Documents/TM/exam1/exam1_cv.csv',index=False)


#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                          #-------------#TFIDF#--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#tfidf to create words frequency array
vect = TfidfVectorizer(stop_words='english', max_features=3000)

X = vect.fit_transform(lem_df.pop('Headline')).toarray()

mydf1 = pd.DataFrame(X, columns=vect.get_feature_names())
mydf1.columns.values.tolist()

mydf1.to_csv(r'/Users/rika/Documents/TM/exam1/exam1_tfidf.csv',index=False)



#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#Transaction data#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


df_exam_final = pd.read_csv("/Users/rika/Documents/TM/exam1/after_lem_exam1.csv")

df_exam_final['transaction'] = df_exam_final['Headline'].str.strip('()').str.split(' ')
#exploding the dataframe to create transaction based data

label = df_exam_final['LABEL']


df_transaction= pd.DataFrame(df_exam_final['transaction'].tolist())

#df_transaction.insert (0, label)
#some values had just '#' so removing it and replacing with 'None'
df_transaction.replace({'#': None},inplace =True, regex= True)
#removing index and column names from the data and storing it in a csv file
df_transaction.to_csv(r'/Users/rika/Documents/TM/exam1/exam1_transaction.csv',
header=None,index=False)
df_transaction.to_csv(r'/Users/rika/Documents/TM/exam1/exam1_transaction_nolabel.csv',
header=None,index=False)

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#Kmeans#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


exam1_clust_cv = pd.read_csv("/Users/rika/Documents/TM/exam1/exam1_tfidf.csv", index_col=0) 
df_exam1_clust = pd.read_csv("/Users/rika/Documents/TM/exam1/after_lem_exam1.csv", index_col=0) 

#plotting silhouette score to determine the best number of clusters
silhouette_coefficients = []
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(exam1_clust_cv)
    score = silhouette_score(exam1_clust_cv, kmeans.labels_)
    silhouette_coefficients.append(score)
    
    
#plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.title("Silhouette Interpretation")
plt.show()

#running kmeans model on the tfidf dataframe
kmeans_cv = KMeans(n_clusters=6)
kmeans_cv.fit(exam1_clust_cv)
labels_cv = kmeans_cv.predict(exam1_clust_cv)
print(labels_cv)


#tfidf to create words frequency array
vect = TfidfVectorizer(stop_words='english', max_features=5000)

X = vect.fit_transform(df_exam1_clust['Headline'].values.astype('U'))

#X = vectorizer.fit_transform(text_clust['Clean_Text'].values.astype('U'))

# initialize PCA with 3 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]
df_exam1_clust['x0'] = x0
df_exam1_clust['x1'] = x1



# initialize kmeans with 3 centroids
#iterator=1
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    # fit the model
    kmeans.fit(X)
    col_name = 'cluster' + str(i)
    # store cluster labels in a variable
    clusters = kmeans.labels_
    df_exam1_clust.loc[:, col_name] = clusters 
    #iterator += 1 
    def get_top_keywords(n_terms):
        """This function returns the keywords for each centroid of the KMeans"""
        print("\n")
        print("When K = ",i)
        print("\n")
        print('CENTROIDS ARE', ['bold'])
      
        df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
        terms = vect.get_feature_names_out() # access tf-idf terms
        for k,r in df.iterrows():
            print('Cluster {}'.format(k))
            # for each row of the dataframe, find the n terms that have the highest tf idf score
            print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) 
            
    get_top_keywords(20)

#Now we have three K means results.
# Lets map clusters to appropriate labels 
cluster_map = {0: "Cluster1", 1: "Cluster2", 2: "Cluster3", 3: "Cluster4", 4: "Cluster5", 5: "Cluster6", 6: "Cluster7", 7: "Cluster8", }
# apply mapping to cluster3 column
df_exam1_clust['cluster6_label'] = df_exam1_clust['cluster6'].map(cluster_map)

plt.title("Count of each label - No of Clusters = 6", fontdict={"fontsize": 18})
df_exam1_clust.groupby("cluster6_label")["cluster6"].count().plot.pie(figsize=(5,5),autopct='%1.1f%%',label='cluster3_label')
plt.legend()

clust_6 = df_exam1_clust[["Headline", "cluster6"]]


plt.figure(figsize=(10, 10))
sns.lmplot(x='x0', y='x1', data=df_exam1_clust, hue='cluster6', fit_reg=False).set(title='K-Means Clustering K=6 ')



#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#Hclsut - R#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#

#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#ARM - R#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#



#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
                                         #--------------#LDA#---------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#
#--------------#############--------------#--------------#############--------------#--------------#############--------------#


exam1_LDA_cv = pd.read_csv("/Users/rika/Documents/TM/exam1/exam1_cv.csv", index_col=0) 

df_exam1_LDA = pd.read_csv("/Users/rika/Documents/TM/exam1/after_lem_exam1.csv", index_col=0) 

mycv1 = CountVectorizer(input = 'content', stop_words = 'english', max_features=3000)
mymat = mycv1.fit_transform(df_exam1_LDA['Headline'])
mycols = mycv1.get_feature_names()
mymat1 = mymat.toarray()

################ LDA ################


# We have 5 different cuisines in our dataset and therefore we will expect LDA to produce 5 topics
num_topics = 3

# Input data frame for LDA
lda_input_df = exam1_LDA_cv
#lda_input_df = lda_input_df.drop(columns=['Label'])

# Instantiate the LDA model with 100 iterations and 5 topics
lda_model_DH = LatentDirichletAllocation(n_components=num_topics,
                                         max_iter=100, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(lda_input_df)
#print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

lda_model_DH.fit(mymat)

# Get the matrix of values which can then be used to obtain top 15 words for each topic
word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()
num_top_words = 15
vocab = mycv1.get_feature_names_out()
vocab_array = np.asarray(vocab)

# Plot the top 15 words under each topic using matplotlib
fontsize_base = 15
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 2)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)
plt.tight_layout()
plt.show()









