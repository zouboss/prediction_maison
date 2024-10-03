import joblib
import numpy as np
import streamlit as st


# Chargement du modèle
model = joblib.load("model1.joblib")

# Définition d'une fonction d'inférence
def inference(bedroomabvgr, lotarea, overallqual, yearbuilt, yearremodadd, totalbsmtsf, grlivarea, fullbath,
                         garagecars):
    new_data = np.array([bedroomabvgr, lotarea, overallqual, yearbuilt, yearremodadd, totalbsmtsf, grlivarea, fullbath,
                         garagecars])
    pred = model.predict(new_data.reshape(1, -1))
    return pred

# Création de l'interface utilisateur
st.title("              MA MAISON SN")
st.subheader("Site developpé par  Pape Alassane Coly")
st.markdown("Grâce à Cette application vous pouvez prédire le Prix d'une maison en indiquant les caracteristiques essentielles ")


bedroomabvgr = st.slider('Nombre de chambres', 0, 10, 1)
lotarea = st.slider('Surface du terrain (en pieds carrés)', 1300, 215245, 7000)
overallqual = st.slider('Qualité globale de la maison (sur une échelle de 1 à 10)', 1, 10, 5)
yearbuilt = st.slider('Année de construction', 1800, 2022, 2000)
yearremodadd = st.slider('Année de la dernière rénovation', 1800, 2022, 2010)
totalbsmtsf = st.slider('Surface du sous-sol (en mètres carrés)', 0, 568, 93, step=1)
grlivarea = st.slider('Surface habitable (en mètres carrés)', 31, 524, 139, step=1)
fullbath = st.slider('Nombre de salles de bain complètes', 0, 5, 2)
garagecars = st.slider('Nombre de voitures pouvant être garées dans le garage', 0, 4, 2)


# Création du bouton "Predict" qui retourne la prédiction du modèle
if st.button("Prédire"):
    prediction = inference(bedroomabvgr, lotarea, overallqual, yearbuilt, yearremodadd, totalbsmtsf, grlivarea,
                           fullbath, garagecars)
    #resultat = "Le prix (en francs CFQ) de cette maison est égal à : XOF" + str(prediction[0])
    resultat = "Le prix (en francs CFA) de cette maison est égal à : " + str(int(prediction[0] * 550)) + " FCFA"
    st.success(resultat)