import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import numpy as np
df1=pd.read_csv('training_text',sep='\|\|')
print(df1.head(2))
df2=pd.read_csv('training_variants',sep=',')
print(df2.head(2))
df3=pd.concat([df2.set_index('ID'),df1.set_index('ID')],axis=1,join='inner').reset_index()
print(df3.head(2))


df3.drop("Variation",axis=1,inplace=True)
#converting gene to integer

text_digit_val={}
def convert_to_int(val):
    return text_digit_val[val]

column_contents=df3['Gene'].values.tolist()
unique_elements=set(column_contents)
x=0
for ele in unique_elements:
    if ele not in text_digit_val:
        text_digit_val[ele]=x
        x=x+1

df3['Gene']=list(map(convert_to_int,df3['Gene']))
target=df3['Gene']
df3.drop("ID",axis=1,inplace=True)
df3.drop("Gene",axis=1,inplace=True)
train=df3['Text']

cv=TfidfVectorizer(min_df=1,stop_words='english')
X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2,random_state=4)
x_traincv=cv.fit_transform(X_train)
y_train=y_train.astype('int')
x_testcv=cv.transform(X_test)
clf=MultinomialNB()
clf.fit(x_traincv,y_train)

pred=clf.predict(x_testcv)
count=0
actual=np.array(y_test)
for i in range(len(pred)):
    if(pred[i]==actual[i]):
    
        count=count+1


print('accuracy',float(count)/len(pred))

