import pandas as pd
import numpy as np	
data =pd.read_csv(r"C:\Users\user\Downloads\spam.csv", encoding='latin-1')
data.head()
data.isnull().sum()
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1, inplace= True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
data['v2_lower']= data['v2'].apply(lambda x: x.lower())
data.head()
import nltk
nltk.download('stopwords')
st_words = stopwords.words('english')
data['count_before_stop'] = data['v2_lower'].apply(lambda row:len(row.split(" ")))
nltk.download('wordnet')
lemmatizerword= WordNetLemmatizer()
data['v2_lower']= data['v2_lower'].apply(lambda x: " ".join([lemmatizerword.lemmatize(word) for word in x.split(" ")]))
data['v2_lower'] = data['v2_lower'].apply(lambda x: " ".join([word for word in x.split(" ") if word not in st_words]))
import re
data['v2_lower_new']= data['v2_lower'].apply(lambda x: re.sub('[^a-zA-Z]+',' ', x))
from sklearn.feature_extraction.text import TfidfVectorizer
TD = TfidfVectorizer()
TD.fit(data['v2_lower_new'])
X = TD.transform(data['v2_lower_new'])
transformedX = pd.DataFrame(X.toarray(), columns=TD.get_feature_names())
transformedX.shape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
LE.fit(data['v1'])
data['v1'] = LE.transform(data['v1'])
train_x, test_x, train_y, test_y = train_test_split(transformedX, data['v1'], test_size=0.2, random_state=100)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(train_x,train_y)
pred_y= model.predict(test_x)
from sklearn.metrics import classification_report
print(classification_report(pred_y,test_y))
