import numpy as np
import pandas as pd
import nltk
from flask import Flask,request,render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('DatafinitiElectronicsProductsPricingData.csv')
# select only the desired columns
df = df[['id', 'name','brand', 'categories','sourceURLs']]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df["name"] = df["name"].str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)

def tokenization(txt):
    tokens = nltk.word_tokenize(txt.lower())
    stemming = [stmer.stem(w) for w in tokens if not w in stopwords.words('english') and w.isalnum()]
    return " ".join(stemming)

df['name'] = df['name'].apply(lambda x:tokenization(x))
df['categories'] = df['categories'].apply(lambda x:tokenization(x))
df['name_cate'] = df['name'] + " " + df['categories']


def cosine_sim(txt1,txt2):
    obj_tfidf = TfidfVectorizer(tokenizer=tokenization)
    tfidfmatrix = obj_tfidf.fit_transform([txt1,txt2])
    similarity = cosine_similarity(tfidfmatrix)[0][1]
    return similarity

def recommation(query):
    tokenized_query = tokenization(query)
    df['similarity'] = df['name_cate'].apply(lambda x: cosine_sim(tokenized_query,x))
    final_df = df.sort_values(by=['similarity'],ascending=False).head(20)[['name','brand','categories','sourceURLs']]
    return final_df
#     idx = df[df['name'] == query].index[0]

#creating app
app = Flask(__name__)

#paths
@app.route('/')
def index():
    names = df['name'].values
    return render_template("index.html", name= names)

@app.route('/predict',methods=['POST'])
def predict():
    query = request.form['name']
    final_df = recommation(query)
    print(final_df)

    return render_template('index.html',df = final_df)



# python main
if __name__ == "__main__":
    app.run(debug=True)