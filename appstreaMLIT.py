import streamlit as st
import pickle
import numpy as np

# Charger le modèle et le scaler
model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Titre de l'application
st.title("🔍 Prédiction d'annulation de réservation d'hôtel")

st.write("Veuillez saisir les informations suivantes :")

# Formulaire de saisie
lead_time = st.number_input("Lead Time", min_value=0)
stays_in_weekend_nights = st.number_input("Nuits en week-end", min_value=0)
stays_in_week_nights = st.number_input("Nuits en semaine", min_value=0)
adults = st.number_input("Nombre d'adultes", min_value=0)
children = st.number_input("Nombre d'enfants", min_value=0)
babies = st.number_input("Nombre de bébés", min_value=0)
adr = st.number_input("ADR (tarif moyen journalier)", min_value=0.0)
total_of_special_requests = st.number_input("Total des demandes spéciales", min_value=0)

# Bouton pour faire une prédiction
if st.button("Prédire"):
    features = [
        lead_time,
        stays_in_weekend_nights,
        stays_in_week_nights,
        adults,
        children,
        babies,
        adr,
        total_of_special_requests
    ]
    
    # Mise à l'échelle
    features_scaled = scaler.transform([features])

    # Prédiction
    prediction = model.predict(features_scaled)[0]
    result = "❌ Annulée" if prediction == 1 else "✅ Confirmée"
    
    st.subheader("Résultat de la prédiction :")
    st.success(result)
