import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')
sns.set_theme(context="notebook")


#---------DEVELOPMENT AND EVALUATION READING-------#
df = pd.read_csv("development.csv")
df.drop(columns=["Id"], inplace=True)
evaluation = pd.read_csv("evaluation.csv")
evaluation.drop(columns=["Id"], inplace=True)


#---------NULL ARTICLE, \\N ARTICLE AND ADV TITLE PREPROCCESSING--------------#

df = df[~df['title'].isna()]
df = df[~df["article"].isna()]
#\\n fixing
df["is_null"] = (df["article"] == "\\N").astype(int)
df["article"] = df["article"].replace("\\N", "")
evaluation["is_null"] = ((evaluation["article"] == "\\N") | evaluation["article"].isna()).astype(int)
evaluation["article"] = evaluation["article"].replace("\\N", "")
evaluation["article"] = evaluation["article"].fillna("")

#\\adv
df["is_adv"] = df["title"].str.startswith("ADV").astype(int)
df["title"] = df["title"].str.replace(r'^ADV.*', 'ADV', regex=True)
df.loc[df["title"].str.startswith("ADV"), "article"] = ""
evaluation["is_adv"] = evaluation["title"].str.startswith("ADV").astype(int)
evaluation["title"] = evaluation["title"].str.replace(r'^ADV.*', 'ADV', regex=True)
evaluation.loc[evaluation["title"].str.startswith("ADV"), "article"] = ""

#-----------TEXT PREPROCESSING--------------#

#patterns regex
regex_url = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
regex_query    = re.compile(r"\?[^ \t\r\n\"'>]+")
regex_photo   = re.compile(r"(?i)\bhttps?://|www\.")
regex_separator     = re.compile(r"[\/\.\-_=%:&]+")
rexeg_tags      = re.compile(r"<[^>]+>")
regex_escape      = re.compile(r"\\[ntr]")
regex_notalnum   = re.compile(r"[^a-zA-Z0-9\s]")
regex_blanks       = re.compile(r"\s+")
regex_digitstoken = re.compile(r"\b\w*\d+\w*\b")

def clean_text(text):
    if not isinstance(text, str):
        return ""

    
    urls = regex_url.findall(text)
    if urls:
        for u in urls:
            u = regex_query.sub("", u)     # drop query
            u = regex_photo.sub("", u)     # drop http/www
            u = regex_separator.sub(" ", u)      
            text += " " + u
    #clean the links without eliminating the information inside such us rss/entertainment, rss/sport, rss/world etc

    #removing html tags
    text = rexeg_tags.sub(" ", text)

    #removing escape
    text = regex_escape.sub(" ", text)

    #only numbers and letters, so i remove . , # ?
    text = regex_notalnum.sub(" ", text)

    #lowering text
    text = text.lower()
    text = regex_blanks.sub(" ", text).strip()
    
    #words with number inside that can be rumor
    text = regex_digitstoken.sub(" ", text)

    #file extension (images link etc)
    text = re.sub(r"\b(jpg|jpeg|png|gif|webp|svg)\b", " ", text)

    #recurring technical tokens surviving 
    text = re.sub(
    r"\b(com|net|org|www|img|images|photo|default|capt|sge|yimg|rd|us)\b", " ", text)

    #letters and too short code
    text = re.sub(r"\b[a-z]{1,2}\b", " ", text)

    #last blanks normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text

df['title_clean'] = df['title'].apply(clean_text)
df['article_clean'] = df['article'].apply(clean_text)
evaluation["article_clean"] = evaluation["article"].apply(clean_text)
evaluation['title_clean'] = evaluation['title'].apply(clean_text)

#----------TIMESTAMP PREPROCESSING-------------#

df['ts_is_missing'] = (df['timestamp'] == '0000-00-00 00:00:00').astype(int)
dt_series = pd.to_datetime(df['timestamp'], errors='coerce')
df['yr'] = dt_series.dt.year.fillna('MISSING').astype(str)
df['mo'] = dt_series.dt.month.fillna('MISSING').astype(str)
df.drop(columns=["timestamp"], inplace=True)

evaluation['ts_is_missing'] = (evaluation['timestamp'] == '0000-00-00 00:00:00').astype(int)
dt_series = pd.to_datetime(evaluation['timestamp'], errors='coerce')
evaluation['yr'] = dt_series.dt.year.fillna('MISSING').astype(str)
evaluation['mo'] = dt_series.dt.month.fillna('MISSING').astype(str)
evaluation.drop(columns=["timestamp"], inplace=True)

#-------------SOURCE PREPROCESSING---------------#

source_mapping = {
    
    #Infotainment    
    'RedNova': 'Infotainment', 'CNET': 'Infotainment', 'CNET\\': 'Infotainment',
    'Register': 'Infotainment', 'ZDNet.com': 'Infotainment', 'Wired': 'Infotainment',
    'Computerworld': 'Infotainment', 'ComputerWorld': 'Infotainment', 'InfoWorld': 'Infotainment',
    'PCWorld': 'Infotainment', 'TechNewsWorld': 'Infotainment', 'eWeek': 'Infotainment',
    'Apple': 'Infotainment', 'Search': 'Infotainment', 'SecurityFocus': 'Infotainment',

    
    #Business&Finance    
    'Forbes': 'Business&Finance', 'Bloomberg': 'Business&Finance', 'Motley': 'Business&Finance',
    'TheStreet.com': 'Business&Finance', 'Financial': 'Business&Finance', 'Ananova': 'Business&Finance',


    #Sport_Newspaper
    'ESPN': 'Sports_Newspaper', 'Sports': 'Sports_Newspaper', 'MLB.com': 'Sports_Newspaper','TSN.ca': 'Sports_Newspaper',
    'sportinglife.com': 'Sports_Newspaper', 'CNN/SI': 'Sports_Newspaper',


    # Global Wire produces raw news,
    # Global Broadcaster interprets and narrates news (editorial)
    # Digital Portal redistributes news (aggregator)    

    'Reuters': 'Global_Wire', 'Xinhua': 'Global_Wire', 'IPS': 'Global_Wire',
    'BBC': 'Global_Broadcaster', 'CNN': 'Global_Broadcaster', 'ABC': 'Global_Broadcaster',
    'MSNBC': 'Global_Broadcaster', 'CBS': 'Global_Broadcaster', 'Al-Jazeera': 'Global_Broadcaster',     
    'Yahoo': 'Digital_Portal', 'Topix.Net': 'Digital_Portal', 'Topix': 'Digital_Portal',
    'Rediff': 'Digital_Portal',

    # Prestigious press
    'Guardian': 'Prestige_Press', 'Washington': 'Prestige_Press', 'Boston': 'Prestige_Press',
    'Times': 'Prestige_Press', 'Independent': 'Prestige_Press', 'Scotsman': 'Prestige_Press',

    # Health and Lifestyle
    
    'Medical': 'Health_Magazine', 'WebMD': 'Health_Magazine',
    'Health': 'Health_Magazine', 'HealthCentral.com': 'Health_Magazine',
    'drkoop.com': 'Health_Magazine',

    'Time': 'General_Magazine',
    'Newsweek': 'General_Magazine'
}

#All the others source (almost irrelevant and unknown will go under Other_Unknown category)

df['source_clean'] = df['source'].str.strip()
df['source_category'] = df['source_clean'].map(source_mapping).fillna('Other_Unknown')
evaluation["source_clean"] = evaluation["source"].str.strip()
evaluation["source_category"] = evaluation["source_clean"].map(source_mapping).fillna("Other_Unknown")


#--------PREPARING TO TFIDF AND COLUMN TRANSFORMER------------#

df_clean = df.drop(columns=["title", "article", "source", "page_rank", "source_clean"])
eval_clean = evaluation.drop(columns=["title", "article", "source", "page_rank", "source_clean"])


df_clean['full_content'] = df_clean['title_clean'].astype(str) + " " + df_clean['article_clean'].astype(str)
df_clean.drop(columns=["title_clean", "article_clean"], inplace=True)
eval_clean['full_content'] = eval_clean['title_clean'].astype(str) + " " + eval_clean['article_clean'].astype(str)
eval_clean.drop(columns=["title_clean", "article_clean"], inplace=True)

#-------------COLUMNTRANSFORMER FOR CATEGORICAL + TFIDF FOR FULL CONTENT (TITLE + ARTICLE)---------------------#

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

X_develop = df_clean.drop(columns=["label"])
y_develop = df_clean['label']
X_eval = eval_clean

def to_text(series):
    return series.squeeze().astype(str)

preprocess = ColumnTransformer(
    transformers=[
        (
            "tfidf",
            Pipeline(steps=[
                ("selector", FunctionTransformer(to_text, validate=False)),
                ("tfidf", TfidfVectorizer(                    
                    ngram_range=(1,2),
                    min_df = 5,
                    max_df = 0.85,                      
                    stop_words="english"                    
                ))
            ]),
            ["full_content"]
        ),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['yr', 'mo', 'source_category'])
             
    ], remainder="passthrough"
    
)


#----------------------MODEL FITTING AND PREDICTING---------------------------#

X_develop_tfidf = preprocess.fit_transform(X_develop)
X_eval_tfidf = preprocess.transform(X_eval)
model_def = LinearSVC(C=0.1, loss="squared_hinge", random_state=42, max_iter=2000, class_weight="balanced")
model_def.fit(X_develop_tfidf, y_develop)
y_eval_pred = model_def.predict(X_eval_tfidf)

#---------------------SUMBISSION-----------------#

submission = pd.DataFrame({
    "Id": evaluation.index, 
    "Predicted": y_eval_pred
})

submission.to_csv("submission.csv", index=False)
submission = pd.DataFrame({
    "Id": evaluation.index, 
    "Predicted": y_eval_pred
})
