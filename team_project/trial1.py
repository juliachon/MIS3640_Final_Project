import pandas as pd
import re
import nltk 
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# #%matplotlib inline
# import seaborn as sns
# from bokeh.models import *
# from bokeh.plotting import *
# from bokeh.io import *
# from bokeh.tile_providers import *
# from bokeh.palettes import *
# from bokeh.transform import *
# from bokeh.layouts import *
# from bokeh.plotting import ColumnDataSource, figure, output_file, show
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import r2_score
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

#reading data
data = pd.read_csv(r"team_project/listings.csv") #for when deedee is coding
#data = pd.read_csv(r"listings.csv") #for when julia is coding

#created dataframe of relevant variables
df_name = pd.DataFrame(data, columns = ['name','neighbourhood_group','neighbourhood'])
# print(df_name)

#cleaning/checking data

#checking datatype
#print(df.dtypes)
#checking for null values
#print(df.isnull().sum())
#renaming columns to make more sense
df_new_name = df_name.rename(columns={'neighbourhood_group':'Borough','neighbourhood':'Neighborhood','name':'Name'})
# print(df_new)

#remove all the non-letters and change the names to lowercase, for ease of analysis
def clean_text(string_in):
    string_in = re.sub("[^a-zA-Z]", " ", str(string_in))  # Replace all non-letters with spaces
    string_in = string_in.lower()                         # Tranform to lower case    
    
    return string_in.strip()

df_new_name["cleaned_name"] = df_new_name.Name.apply(clean_text)
df_new_name = df_new_name.drop(columns=['Name'])
df_new_name = df_new_name.rename(columns={'cleaned_name':'Name'})
df_new_name = df_new_name.sort_values(by=['Neighborhood'])

name_list = df_new_name["Name"].tolist()
# name_list = word_tokenize(name_list)
name_list = " ".join(name_list)
name_str = str(name_list)
wordList = re.sub("[^\w]", " ",  name_str).split()
# print(wordList)

def word_polarity(wordList):
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]

    for word in wordList:               
        testimonial = TextBlob(word)
        if testimonial.sentiment.polarity >= 0.7:
            pos_word_list.append(word)
        elif testimonial.sentiment.polarity <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    

    # print('Positive :',pos_word_list) 
    # print('Neutral :',neu_word_list)    
    return('Negative :',neg_word_list)



print(word_polarity(wordList))
# print(type(wordList)
