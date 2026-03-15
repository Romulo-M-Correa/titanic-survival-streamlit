import streamlit as st
import pandas as pd
import joblib

# carregar modelo
modelo = joblib.load("modelo_titanic_tunado.pkl")

st.title("🚢 Titanic Survival Predictor")

st.write("Preencha os dados do passageiro")

pclass = st.selectbox("Classe", [1,2,3])
sex = st.selectbox("Sexo", ["male","female"])
age = st.slider("Idade",0,80,30)
sibsp = st.number_input("SibSp",0)
parch = st.number_input("Parch",0)
fare = st.number_input("Fare",0.0)
embarked = st.selectbox("Embarked",["C","Q","S"])

family_size = sibsp + parch + 1
is_child = 1 if age < 12 else 0

dados = pd.DataFrame([{
    "PassengerId":999,
    "Pclass":pclass,
    "Sex":sex,
    "Age":age,
    "SibSp":sibsp,
    "Parch":parch,
    "Fare":fare,
    "Embarked":embarked,
    "FamilySize":family_size,
    "IsChild":is_child
}])

if st.button("Prever"):
    previsao = modelo.predict(dados)[0]
    prob = modelo.predict_proba(dados)[0][1]

    if previsao == 1:
        st.success("Provavelmente sobreviveria")
    else:
        st.error("Provavelmente não sobreviveria")

    st.write("Probabilidade:", round(prob*100,2), "%")
