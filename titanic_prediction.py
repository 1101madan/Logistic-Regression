import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title('Titanic Survival Prediction')
st.sidebar.header('User Input Features')
def user_input_fetures():
  st_pclass = st.sidebar.selectbox('Passenger Class',('1','2','3'))
  st_age = st.sidebar.number_input('Age')
  st_sibsp = st.sidebar.number_input('Number of siblings/spouses')
  st_parch = st.sidebar.number_input('Number of parents/children')
  st_fare = st.sidebar.number_input('Fare')
  st_sex_male = st.sidebar.number_input("Male(enter '1' if yes or '0')")
  st_sex_female = st.sidebar.number_input("Female(enter '1' if yes or '0')")
  st_embarked_c = st.sidebar.number_input("Embarked at C(enter '1' if yes or '0')")
  st_embarked_q = st.sidebar.number_input("Embarked at Q(enter '1' if yes or '0')")
  st_embarked_s = st.sidebar.number_input("Embarked at S(enter '1' if yes or '0')")
  data = {
      'Pclass':st_pclass,
      'Age':st_age,
      'SibSp':st_sibsp,
      'Parch':st_parch,
      'Fare':st_fare,
      'Sex_female':st_sex_female,
      'Sex_male':st_sex_male,
      'Embarked_C':st_embarked_c,
      'Embarked_Q':st_embarked_q,
      'Embarked_S':st_embarked_s
  }
  features = pd.DataFrame(data,index=[0])
  return features

df = user_input_fetures()
st.subheader('User Input features')
st.write(df)

titanic_train_streamlit = pd.read_csv('https://raw.githubusercontent.com/1101madan/Logistic-Regression/main/Titanic_train.csv')
titanic_train_streamlit.drop(['PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)
titanic_train_streamlit['Age']= titanic_train_streamlit['Age'].fillna(titanic_train_streamlit['Age'].mean())
titanic_train_streamlit = pd.get_dummies(titanic_train_streamlit,columns=['Sex','Embarked'],dtype=int)

X = titanic_train_streamlit.iloc[:,1:]
Y = titanic_train_streamlit.iloc[:,0]

model = LogisticRegression()
model.fit(X,Y)
prediction = model.predict(df)
prediction_probability = model.predict_proba(df)

st.subheader('The Probability of the Passenger')
st.write('The Passenger Survives' if prediction_probability[0][1]>0.5 else 'The Passenger Does not Survives')

st.subheader('Survival Probability:')
st.write(prediction_probability)

