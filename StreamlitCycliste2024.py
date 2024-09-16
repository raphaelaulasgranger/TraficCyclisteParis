#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Formation Data scientest JUIN24 
DATA ANALYST 

"""
import re
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
import plotly.graph_objects as go
from streamlit.components.v1 import html
import plotly.express as px
# from PIL import Image
from streamlit_folium import folium_static
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

# @st.cache(persist=True)
df=pd.read_csv("./data_D_velo_meteo21-24.csv", parse_dates=[0], index_col=0)
compteurs = pd.read_csv('./llistecompteur.csv')


def nettoyer_nom(nom):
    # Dictionnaire de correspondance pour les caractères accentués
    accents = {
        'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ý': 'y', 'ÿ': 'y',
        'ç': 'c',
        'ñ': 'n'
    }
    
    # Convertir en minuscules
    # nom = nom.lower()
    
    # Remplacer les caractères accentués
    for accent, sans_accent in accents.items():
        nom = nom.replace(accent, sans_accent)
    
    # Remplacer les espaces par des underscores et supprimer les caractères non alphanumériques
    nom = re.sub(r'[^A-Za-z0-9_]', '', nom.replace(' ', '_'))
    
    return nom




compteurs.columns = [nettoyer_nom(col) for col in compteurs.columns]


st.sidebar.title("Sommaire")
pages = ["Projet", "Jeux de données", "Data Visualisation",
         "Cartographie", "Etude temporelle", "Analyse de resultats"]


page = st.sidebar.radio("Aller vers", pages)



# Ajouter un encadré en dessous du menu dans la barre latérale
st.sidebar.markdown(
    """
    <div style="border: 2px solid #3bd8c5; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
        <p style="font-size: 16px; color: #333;" 
        container = st.container(border=True) 
        <p style='text-align: center;'>Réalisé par :</p>
        <p style='text-align: center;'><b>Fadela Chana-Itema</b></p>
        <p style='text-align: center;'><b>Raphaël Aulas-Granger</b></p>
        <p style='text-align: center;'><b>Anh Nguyen LE</b></p>
        <p style='text-align: center;'><b>Loan Le Guillou</b></p>
        
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(  """
 <h3 style="text-align: center;">
     Promotion BootCamp Data Analyst - Juin 2024
 </h3>
 """,
 unsafe_allow_html=True
 )

if page == pages[0]:
    url_gif = "https://i.giphy.com/l3PAGCARIwGjxE4gTJ.webp"
    st.markdown(f"![Cycliste à Paris]({url_gif})")
    st.title("Le vélo à Paris")
    st.header("Etude du trafic cycliste de Janvier 2021 à Juillet 2024")
    st.markdown(
    """
    <h3 style="text-align: center;">
        Projet réalisé dans le cadre de la formation Data Analyst de DataScientest.com Promotion Bootcamp Juin 2024
    </h3>
    """,
    unsafe_allow_html=True
)
   
    container = st.container(border=True)
    container.markdown("<p style='text-align: center;'>Réalisé par :</p>", unsafe_allow_html=True)
    container.markdown("<p style='text-align: center;'><b>Fadela Chana-Itema</b></p>", unsafe_allow_html=True)
    container.markdown("<p style='text-align: center;'><b>Raphaël Aulas-Granger</b></p>", unsafe_allow_html=True)
    container.markdown("<p style='text-align: center;'><b>Anh Nguyen LE</b></p>", unsafe_allow_html=True)
    container.markdown("<p style='text-align: center;'><b>Loan Le Guillou</b></p>", unsafe_allow_html=True)

    st.subheader("Contexte :")
    st.markdown(" Depuis le changement de municipalité en 2001, Paris a vu son espace public être radicalement "
                "transformé pour un partage nouveau de la voirie.  \n"
                "Mise en place d’un système de vélo partagé (Vélib), transformation de la voirie par la création "
                "d’un réseau de pistes cyclables destinées tant à protéger le piéton des incivilités des cyclistes "
                "que protéger ces derniers des automobiles, le mode de déplacements des Parisiennes et "
                "Parisiens a été accompagné par une politique audacieuse.  \n")
    st.markdown("Si on comptait 200 km de pistes cyclables en 2001, c’est aujourd’hui près de 2000 km d’infrastructures dont dispose la capitale sur 36 % de ses axes. Paris concentre à elle seule "
                "aujourd’hui les 2/3 des aménagements franciliens pour le vélo.  \n"
                "Paris, avec Venise, compte parmi les villes où les ménages possèdent le moins une automobile.  \n"
                "Les modes de déplacements des Parisiennes et Parisiens se sont considérablement modifiés, "
                "avec une estimation à 930000 trajets quotidiens en Vélo à PARIS aujourd’hui.")
    st.subheader("Objectif :")
    st.markdown( 
                "L’objectif de ce  projet est de qualifier et quantifier ces usages et d’esquisser des approches "
                "opérationnelles pour les services publics parisiens.  \n"
                "De premières données sont disponibles : les compteurs de fréquentations des pistes cyclables "
                "fourniront le cœur des données analysées pour cette étude.")

    st.header("Cette présentation est un résumé du projet pour une exploitation via streamlit et une mise en ligne")




# Nouvelle page Jeux de donnée

if page == pages[1]:
    st.header("Exploration de données")
    st.subheader("Données principales")
    st.markdown("Jeu de données des comptages horaires de vélos par compteur et localisation des sites de "
                "comptage en J-1 sur 13 mois glissants.")
    st.write("Origine : *opendata.paris.fr* [link](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/)")
    st.markdown("Producteur : Direction de la Voirie et des Déplacements - Ville de Paris")
    st.markdown("Territoire : Paris")
    st.markdown("Dernier traitement : 9 juillet 2024 08:24 (métadonnées), 9 juillet 2024 08:24 (données)")
    st.markdown("Créé : 22 avril 2020")
    st.markdown("Publié : 22 avril 2020")
    st.markdown("Fréquence de mise à jour : Quotidienne")
    st.subheader("Aperçu du premier jeu de données :")
    
#Premier DataFrame    
    st.subheader("Nous avons effectué un nettoyage de données afin de les rendre exploitables")
    st.markdown("Les données initiales comportent un total de 16 colonnes pour 1 001 651 lignes, aucune d’entre elles "
                "n’est un doublon. Elles sont issues de l’extraction sur 13 mois glissants précédemment "
                "évoquée. Nous travaillons sur la plage du 1er juin 2023 - 2 juillet 2024.")
    st.subheader("Traitement des données :")
    st.markdown("Les valeurs manquantes : Nous avons traité les valeurs manquantes dans les colonnes que nous avons gardées : "
                "Pour 3 compteurs, à savoir les compteurs "
                "« 90 Rue De Sèvres NE-SO », « 90 Rue De Sèvres SO-NE », « 30 rue Saint Jacques N-S’ "
               "nous avons fait le choix de leur affecter des coordonnées GPS correspondant à l’adresse "
                "postale de ces adresses, et de leur attribuer 2 identifiants aléatoires afin de permettre un "
                "regroupement ultérieur des compteurs sur les sites 90 rue de Sèvres et 30, rue St-Jacques.")


    st.markdown("Les données ont été regroupées par jour."
                "Nous avons colligé les données disponibles sur les archives data.gouv.fr"
                "Ceci constitue ainsi une base de comptage de 2021 à 2024.")
    st.subheader("Ajout de variables :")
    st.markdown("Nous avons crée des variable Jours, Mois, Années, ainsi que la longitude et la latitude à partir des données disponible dans le DataFrame.")
    # st.write ( df.columns)
    st.dataframe(df[['Comptage','annee', 'mois', 'joursem', 'weekend', 'vacances', 'feries']].head(10))
    checkbox1 = st.checkbox("Afficher les NA") 
    if checkbox1: 
        st.dataframe(df[['Comptage','annee', 'mois', 'joursem', 'weekend', 'vacances', 'feries']].isna().sum())

#Deuxieme DataFrame    
    
    # st.subheader("Aperçu du fichier retravaillé")
    # st.dataframe(df.head(10))
    # checkbox2 = st.checkbox("Afficher les NA apres traitement") 
    # if checkbox2:   
    #     st.dataframe(df2.isna().sum())
    # st.markdown("Ce fichier retravaillé nous a permit de faire des visualisations plus detaillés dans le temps que vous pourrez observer dans l'onglet Data Visualisation ainsi qu'une cartographie dans l'onglet suivant.")
    
    st.subheader("L’historique des données de bornages cyclistes")
    st.markdown(
                "Nous constatons l’important effort de la Ville de Paris pour ses équipements à "
                "usage des cyclistes. Ainsi, si on comptait **53** compteurs en 2017, nous en dénombrons ainsi "
                " **204** en 2024. "
                "Ceci pose une problématique quant à la comparaison des mesures du trafic dans le temps. "
                "Pour ne pas gonfler artificiellement la mesure de la pratique cycliste dans Paris par "
                "l’augmentation du nombre de capteurs, nous faisons le choix de ne retenir que des compteurs "
                "existants depuis 2021,ce qui nous fait **73** compteurs, nonobstant les quelques changements de nom des capteurs au fil des "
                "progrès et choix techniques sur les bases de données par le ville de Paris.")
    
    st.markdown("***Nous avons choisi d'ajouter des données Météos afin d'analyser la fréquentation en fonction des "
                "conditions climatiques et d'adapter le DataFrame au machine learning :***")
    
#Troisième DataFrame
    
    st.subheader("Aperçu du fichier avec données météos")
    st.dataframe(df.head(10))
    checkbox3 = st.checkbox("Afficher les NA avec météo") 
    if checkbox3: 
        st.dataframe(df.isna().sum())
    
    st.subheader("Les données météo")
    st.markdown("Les données météo sont disponibles sur le site : *Meteo.data.gouv.fr* [link](https://meteo.data.gouv.fr/)")
    st.markdown("Elles disposent d’une multitude de données, allant du point de rosée jusqu’à la force du vent.  \n"
                "Elles sont d’une fréquence horaire et proviennent de 6 stations météo parisiennes différentes.  \n"
                "Nous gardons la station « Montsouris » qui ne contient que très peu de valeurs manquantes et "
                "en considérant que les données météo sont très similaires dans tout Paris.  \n Parmi les "
                "160 données, nous faisons le choix de ne retenir que les données de vent, de pluie, de neige, et "
                "de température, considérant ces données comme les plus pertinentes pour une analyse sur "
                "4 ans et expliquant les choix des Parisiennes et Parisiens de faire appel au vélo comme moyen de transport.  \n")

    st.markdown("17 valeurs sont manquantes. Il s'agit de 17 jours en 2023 durant lesquels aucune donnée n'est remontée."
                 "Cette situation est signalée par la Mairie de Paris. \n Nous faisons le choix de remplacer ces données"
                 "par la mediane des mesures à J+7 et J-7.")
    
    

############################################# Nouvelle page Data 
############################################# Visualisation

if page == pages[2]:
    st.header("Data Visualisation")
    st.subheader("Tableau des corrélations :")
    # st.pyplot ( df.corr())
    # # st.image("tabcorr.png",  use_column_width=True)
    # st.header("Heatmap de corrélation")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr() , annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("La pratique du vélo au cours de la semaine:")  

##
    # Définir les noms des jours de la semaine
    # Calculer la moyenne quotidienne de Y (tous les jours inclus)
    quot_moy = df.groupby('joursem')['Comptage'].mean()
    
    # Créer l'histogramme
    fig = go.Figure(data=[go.Bar(
        x=quot_moy.index,
        y= quot_moy.values,
        text=quot_moy.values.round(2),
        textposition='outside',
        hovertemplate='Date: %{x}<br>Moyenne: %{y:.2f}<extra></extra>'
    )])
    
    # Personnaliser l'apparence
    fig.update_layout(
        title='Moyenne quotidienne de trafic',
        xaxis_title='Jour de la semaine',
        yaxis_title='Moyenne du comptage',
        xaxis_tickangle=-45,
        bargap=0.1 , # Espace entre les barres

        xaxis=dict(
            titlefont=dict(size=18, color="#7f7f7f"),
            tickfont=dict(size=14, color="#7f7f7f"),
            tickmode = 'array',
            tickvals = [0, 1, 2, 3, 4, 5, 6],
            ticktext = ['Lundi','Mardi','Mercredi', 'Jeudi','Vendredi','Samedi','Dimanche']
        ),

        
    )
    
    # Ajuster la mise en page pour une meilleure lisibilité
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_tickformat='%Y-%m-%d'
    )
    
    # Afficher l'histogramme dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
    st.markdown("À travers ce graphique, nous pouvons observer les tendances d’utilisation des vélos à "
                "Paris en fonction des jours de la semaine. Il montre que l’utilisation des vélos à Paris est "
                "principalement concentrée durant les jours de travail. Cela indique que les vélos sont utilisés "
                "en semaine, probablement pour des trajets domicile-travail, des courses ou d’autres activités "
                "régulières. Nous constatons également une légère diminution le lundi et le vendredi, qui "
                "peuvent être expliqués par le fait que ces deux jours sont des jours de télétravail préférés des "
                "salariés. Benoît Serre, vice-président de l’Association des directeurs des ressources humaines "
                "(ANDRH), le confirme : depuis que le travail hybride s’est mis en place, avec la crise sanitaire, "
                "le travail à distance s’est installé dans les entreprises le vendredi et le lundi, avec une légère "
                "préférence pour le vendredi.  \n"
                "Il y a une baisse notable de l’utilisation des vélos le samedi et le dimanche, avec le dimanche "
                "enregistrant la plus faible moyenne de comptage horaire. La baisse durant le week-end pourrait "
                "être liée à des habitudes de déplacement différentes, comme l’utilisation de la voiture, des "
                "transports en commun ou la réduction des déplacements.")

#AU FIL DES MOIS 
    st.subheader("La pratique du vélo au cours des mois ( prise en compte des jours travaillés seulement)")  

    df['C_filtered'] = df['Comptage'].where(df['weekend'] == 0)
    
    Moy_mensuelle_horsweekend = df['C_filtered'].resample('M').mean()

    # Créer l'histogramme
    fig = go.Figure(data=[go.Bar(
        x=Moy_mensuelle_horsweekend.index.strftime('%Y-%m'),
        y=Moy_mensuelle_horsweekend.values,
        text=Moy_mensuelle_horsweekend.values.round(2),
        textposition='auto',
    )])
    
    # Personnaliser l'apparence
    fig.update_layout(
        title='Moyenne mensuelle de Y',
        xaxis_title='Mois',
        yaxis_title='Trafic mensuel',
        xaxis_tickangle=-45
        )
    # Afficher l'histogramme dans Streamlit
    st.plotly_chart(fig)
    st.markdown("À travers ce graphique, nous pouvons observer les tendances d’utilisation des vélos à "
             "Paris en fonction des mois."
            "On voit une saisonnalité forte, avec une baisse les mois d'août et décembre.")


# ANALYSE METEO

    st.header('La météo au fil des saisons…')
    # Création des cases à cocher
    show_vent = st.checkbox("Afficher les données de vent"  )
    show_pluie = st.checkbox("Afficher les données de pluie", value=True)
    show_temp = st.checkbox("Afficher les données de température")
    show_neige = st.checkbox("Afficher les données de neige")   
    
    # Création du graphique
    fig = go.Figure()

    
    # Palette de couleurs
    colors = ['red', 'blue', 'green', 'orange']
    
    # Dictionnaire des données à afficher
    data_to_show = {
        'vent': show_vent,
        'pluie': show_pluie,
        'temperature': show_temp,
        'neige' : show_neige
    }
    
    # Ajout des traces pour chaque variable sélectionnée
    for i, (col, show) in enumerate(data_to_show.items()):
        if show:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line_color= colors[i])
            ))


    
    # if show_vent:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['Vent'], mode='lines', name='Vent', line_color=colors[1] ))
          
    # if show_pluie:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['Pluie'], mode='lines', name='Pluie' , line_color=colors[3]))
    
    # if show_temp:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['Temperature'], mode='lines', name='Température' , line_color=colors[2]))
    
    # if show_neige:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['Neige'], mode='lines', name='Neige' , line_color=colors[0]))
       
    # st.subheader("La pluie")
    # Mise à jour du layout
    fig.update_layout(
        title="Données météorologiques",
        xaxis_title="Date",
        yaxis_title="Valeur",
        legend_title="Légende"
    )
    
    # Affichage du graphique
    st.plotly_chart(fig)
    
    # Affichage du dataframe
    st.write("Données brutes :")
    st.dataframe(df)

    # # fig = px.line(df, x=df.index, y='Pluie', title='La pluie à PARIS')

    # # Personnalisation du graphique
    # fig.update_xaxes(title='Date')
    # fig.update_yaxes(title='les données météo')
    
    # # Affichage du graphique dans Streamlit
    # st.plotly_chart(fig
    #                )


####
    st.subheader("La température à PARIS")

    fig = px.line(df, x=df.index, y='Temperature', title='température')

    # Personnalisation du graphique
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Température')
    
    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)

    st.subheader("La neige à PARIS")

    fig = px.line(df, x=df.index, y='Neige', title='La neige')

    # Personnalisation du graphique
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Neige')
    
    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)


    
    
    st.subheader("La pratique du vélo au fil des saisons… 2021 à 2024")

    fig = px.line(df, x=df.index, y='Comptage', title='Évolution du comptage des vélos au cours du temps')

    # Personnalisation du graphique
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='traffic cycliste')
    
    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)


############################################################################ CARTO
# Nouvelle page Cartographie   
    
if page == pages[3]:
    st.header("Cartographie")
    st.subheader("Emplacement géographique des compteurs:")
 
    compteurs[['latitude', 'longitude']] = compteurs['Coordonnees_geographiques'].str.split(',', expand=True).astype(float)
# Coordonnées de Paris
   
    center_lat =  compteurs['latitude'].mean()
    center_lon = compteurs['longitude'].mean()
    # paris_coords = [48.8566, 2.3522]
    paris_coords = [center_lat,center_lon] 
# carte Paris
    map_paris = folium.Map(location=paris_coords, zoom_start=12)


    for idx, row in compteurs.iterrows():
        folium.CircleMarker(location=[row['latitude'], row['longitude']],
                            popup=row['Nom_du_compteur'],icon=folium.Icon(icon='info-sign'),
                            color="blue",
                            fill=True,
                            fillColor="blue"
                           ).add_to(map_paris)
                    
    #map_paris.save("map_paris.html")     
    folium_static(map_paris)
    
        
# Lire le contenu du fichier HTML
    # with open('map_paris.html', 'r') as f:
    #    map_html = f.read()

#Afficher le contenu HTML dans Streamlit
    # st.components.v1.html(map_html, height=600, scrolling=True)


    st.markdown("Il y avait au demarrage **53** bornes de comptage en 2021, nous en comptons desormais **204** en 2024."
             "La carte géographique que nous avons pu construire à partir des coordonnées géographiques "
             "montre les emplacements des compteurs de vélos que nous avons retenus." 
             " \n"
             "Les points bleus indiquent l’emplacement des compteurs. Ils semblent être répartis de "
             "manière relativement homogène à travers la ville, couvrant la plupart des arrondissements.  \n"
             
               )


# ############################################################## 
# ############################################################## Nouvelle page Etude temporelle

if page == pages[4]:
    st.header("Machine learning et Etude temporelle")
    st.subheader("Données principales")        
    st.markdown("L’objectif de ce projet est de mettre en œuvre une modélisation du trafic cycliste dans Paris en "
                "vue de prédire son intensité, en fonction des conditions météo dont la prédiction est "
                "relativement fiable, de la période de l’année et des jours de la semaine, ce que nous "
                "appellerons les données civiles.  \n")




    # STANDARDISATION
    scaler = MinMaxScaler()
    var_expl = ["joursem", "mois"]
    var_num = ["Pluie", "Vent", "Temperature", "Neige"]
    
    # on standardise var_num
    for col in var_num:
        df[col + "_s"] = scaler.fit_transform(df[[col]])
    
    # on encode var_expl
    df_encoded = pd.get_dummies(df, columns=["joursem"], prefix="c_joursem", dtype=int)
    df_encoded = pd.get_dummies(df_encoded, columns=["mois"], prefix="c_mois", dtype=int)
    col_utiles = [
            "Comptage",
            "annee",
            "weekend",  
            "vacances",
            "feries",
            "Pluie_s",
            "Vent_s",
            "Temperature_s",
            "Neige_s",
            "c_joursem_0",
            "c_joursem_1",
            "c_joursem_2",
            "c_joursem_3",
            "c_joursem_4",
            "c_joursem_5",
            "c_joursem_6",
            "c_mois_1",
            "c_mois_2",
            "c_mois_3",
            "c_mois_4",
            "c_mois_5",
            "c_mois_6",
            "c_mois_7",
            "c_mois_8",
            "c_mois_9",
            "c_mois_10",
            "c_mois_11",
            "c_mois_12"
        ]
    dfr = df_encoded[col_utiles]

    # valeurs manquantes 
    # Remplacement des outliers par la moyenne de  (j+7 + j-7 )/2 
    def remplace_datas_par_moyenne_locale(df, selection):
        for idx in df[selection].index:
            start_date = idx - pd.Timedelta(days=7)
            end_date = idx + pd.Timedelta(days=7)
            mean_value = (df.loc[start_date, 'Comptage']  + df.loc[end_date, 'Comptage'])  /2 
            df.loc[idx, 'Comptage'] = mean_value 
            # display(  df.loc[idx])
        return df

    ## remplacement des data manquantes par J+7 / J-7 

    df_missing_dates = dfr['Comptage'].isna()
    dfr = remplace_datas_par_moyenne_locale  ( dfr , df_missing_dates)
    # missing_dates = dfr [dfr.isna().any(axis=1)].index
    # print("Dates manquantes :",len ( dfr [dfr.isna().any(axis=1)].index),' / ', dfr.shape[0])
    # il manque 17 mesures
   
    # valeurs extremes
    def remplacer_local_extremes(df, column, window_size='7D', threshold=3):
        """
        Élimine les valeurs extrêmes locales d'une série temporelle.
        
        :param df: DataFrame contenant les données
        :param column: Nom de la colonne contenant les mesures
        :param window_size: Taille de la fenêtre glissante (impair)
        :param threshold: Nombre d'écarts-types au-delà duquel une valeur est considérée comme extrême
        :return: DataFrame avec les valeurs extrêmes remplacées par NaN
        """
       
        # Calcul  la moyenne mobile et l'écart-type mobile
        rolling_mean = df[column].rolling(window=window_size, center=True).mean()
        rolling_std = df[column].rolling(window=window_size, center=True).std()
       
        # Identification des valeurs extrêmes
        lower_bound = rolling_mean - (threshold * rolling_std)
        upper_bound = rolling_mean + (threshold * rolling_std)
        # valeur de remplacement
        val_remp = (df[column].shift(-7) +  df[column].shift(+7)  )/2 
    
        # display ( df [~((df[column] >= lower_bound)  & (df[column] <= upper_bound)) ] )
        return df[column].where( ( (df[column] >= lower_bound)  & (df[column] <= upper_bound) )  , other= val_remp   ) 
    # Utilisez la fonction
    dfr['Comptage'] = remplacer_local_extremes(dfr, 'Comptage', window_size='7D', threshold=2 ) 
    # display ( dfr )


#################### Visu   ACF PACF et  test Augmented Dickey-Fuller
    st.header ( "etude de la serie temporelle")
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # Créer les subplots
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    # Tracer l'ACF
    plot_acf(dfr['Comptage'], lags=30, zero=True, ax=ax_acf)
    ax_acf.set_title('ACF - comptage')
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('Corrélation')
    ax_acf.grid(True)
    # Ajuster les graduations sur l'axe x pour l'ACF
    ax_acf.set_xticks(np.arange(0, 31, 1))
    st.write("Graphique ACF")
    st.pyplot(fig_acf)


    
    # Tracer le PACF
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    plot_pacf(dfr['Comptage'], lags=30, zero=True, ax=ax_pacf)
    ax_pacf.set_title('PACF - comptage')
    ax_pacf.set_xlabel('Lag')
    ax_pacf.set_ylabel('Corrélation partielle')
    ax_pacf.grid(True)
    
    # Ajuster les graduations sur l'axe x pour le PACF
    ax_pacf.set_xticks(np.arange(0, 31, 1))
    st.write("Graphique PACF")
    st.pyplot(fig_pacf)
    

    
    # Analyse de la stationnarité de la variable

    st.header( "test de Dickey-Fuller augmenté")
    # Effectuer le test de Dickey-Fuller augmenté
    result = adfuller(dfr['Comptage'])
    # Formater les résultats dans un tableau
    table = [
        ['Valeur de test', result[0]],
        ['P-valeur', result[1]],
        ['Conclusion', 'La série est stationnaire' if result[1] < 0.05 else 'La série est non stationnaire']
    ]
    # Afficher les résultats sous forme de tableau
    st.write(tabulate(table, headers=['Métrique', 'Valeur'], tablefmt='github'))

    st.subheader("Configuration :")
    st.markdown("Le graphe ACF et le graphe PACF nous donne les valeurs P et Q "
                "Après avoir fait une première différenciation de 7, et de constater" 
                "la saisonnalité de 7 jours qui apparait, nous faisons le choix suivants"
                " (p, d, q) = (1, 0, 1)   pour la partie non saisonnière"
                " (P, D, Q, s) = (0, 1, 1, 7)    pour la fréquence hebdomadaire" 
                "le choix de ces coefficients procède d'une recherche itérative."
                "Nous faisons le choix de ne pas intégrer dans les paramètres"
                "la saisonnalité annuelle, que nous allons injecter via la méthode "
                "décrite dans ce lien "
                "https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a")
    

    # prepare Fourier terms
    
    dfr['sin365'] = np.sin(2 * np.pi * dfr.index.dayofyear / 365.25)
    dfr['cos365'] = np.cos(2 * np.pi * dfr.index.dayofyear / 365.25)
    dfr['sin365_2'] = np.sin(4 * np.pi * dfr.index.dayofyear / 365.25)
    dfr['cos365_2'] = np.cos(4 * np.pi * dfr.index.dayofyear / 365.25)

    st.subheader("modele SARIMAX (p, d, q) = (1, 0, 1) (P, D, Q, s) = (0, 1, 1, 7)  ")

    # Définition du modèle SARIMAX
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    (p, d, q) = (1, 0, 1)        # pour la partie non saisonnière
    (P, D, Q, s) = (0, 1, 1, 7)    # pour la partie saisonnière (fréquence hebdomadaire)
    print(p, d, q, P, D, Q, s)
    # Séparer les données en ensemble d'entraînement et ensemble de test
    nb_j_test = 360
    # train_data = df_D['Comptage_quotidien'][:-(nb_j_test)] # on retire les 2 derniers mois
    # test_data = df_D['Comptage_quotidien'][-(nb_j_test):]
    train_data = dfr.iloc[:-(nb_j_test)]  # on retire la sequence de nb_j_test
    # on prend la sequance  finale de  nb_j_test
    test_data = dfr.iloc[-(nb_j_test):]
    # on repart bien de la serie initiale (car ARIMA var refaire les corrections en interne via les coef p  d q )

    col_exog = [        
                         # "weekend",
                          "vacances",
                          "feries",
                          "Pluie_s",
                          # "Vent_s",
                          "Temperature_s",
                          "Neige_s",
                          "c_joursem_0",
                          "c_joursem_1",
                          "c_joursem_2",
                          "c_joursem_3",
                          "c_joursem_4",
                          "c_joursem_5",
                          "c_joursem_6",
                          "c_mois_1",
                          "c_mois_2",
                          "c_mois_3",
                          "c_mois_4",
                          "c_mois_5",
                          "c_mois_6",
                          "c_mois_7",
                          "c_mois_8",
                          "c_mois_9",
                          "c_mois_10",
                          "c_mois_11",
                          "c_mois_12", 
                         'sin365', 'cos365', 
                        'sin365_2', 'cos365_2', 
                      ]

    # model_S = SARIMAX(train_data['Comptage'],
    #                   exog=train_data[col_exog],
    #                   order=(p, d, q),
    #                   seasonal_order=(P, D, Q, s))
    
    # # Ajustement du modèle
    # model_Sfit = model_S.fit()

    # # Sauvegarde du modèle
    # model_Sfit.save('sarimax_model.pkl')

   
    # Chargement du modèle

    model_Sfit = SARIMAXResults.load('sarimax_model.pkl')

    # Prévision sur l'ensemble de train
    validations = model_Sfit.forecast(steps=len(train_data),
                                      exog=train_data[col_exog])
    
    # Prévision sur l'ensemble de test
    forecast = model_Sfit.forecast(steps=len(test_data),
                                   exog=test_data[col_exog])

    ##########################################"" SARIMAX  * affichage
    # Affichage des résultats
    # plt.figure(figsize=(20,12))
    # plt.plot(train_data.index, train_data['Comptage'], label='datas train', color='royalblue')
    # plt.plot(test_data.index, test_data['Comptage'], label='datas Test', color='darkturquoise')
    # plt.plot(train_data.index, validations, label = 'pred sur entrainement ', color = 'orange')
    # plt.plot(test_data.index, forecast, label='Prévisions', color = 'red')
    # plt.legend()
    # Title =  'Prévision SARIMAX ( pdq:'+str( p)+str(d)+str(q)+') (PDQ: ' +str( P)+str(D)+str(Q)+' s:'+str(s)+')' 
    # plt.title  ( Title )
    # # plt.savefig( Title + '.jpg' )
    # plt.show();
    fig, ax = plt.subplots(figsize=(20, 12))
    
    ax.plot(train_data.index, train_data['Comptage'], label='datas train', color='royalblue')
    ax.plot(test_data.index, test_data['Comptage'], label='datas Test', color='darkturquoise')
    ax.plot(train_data.index, validations, label='pred sur entrainement', color='orange')
    ax.plot(test_data.index, forecast, label='Prévisions', color='red')
    
    ax.legend()
    Title = f'Prévision SARIMAX (pdq:{p}{d}{q}) (PDQ: {P}{D}{Q} s:{s})'
    ax.set_title(Title)
    
    st.pyplot(fig)


    # pickle.dump(model_Sfit , open(Title + '.pickle', 'wb'))
    
    
    y_true = test_data['Comptage']
    y_pred = forecast
    #*************************************** Calcul des métriques
    mae = mean_absolute_error(y_true, forecast)
    mse = mean_squared_error(y_true, forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, forecast)
    
    st.write(f"MAE: {mae}")
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"R²: {r2}")
    
    # Analyse des résidus
    residuals = y_true -forecast
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    st.write(f"Moyenne des résidus : {mean_residuals}")
    st.write(f"Std des résidus : {std_residuals}")
    
    # Test de Ljung-Box
    st.write("Test de Ljung-Box")
    ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    st.write(ljung_box_result)




    
    
    
# Nouvelle page Analyse et conclusion   
       
if page == pages[5]:
    st.header("Analyse des resultats")
    st.subheader("Comparaisons des différents modèles prédictifs") 


    
    st.markdown("Nous avons testé plusieurs modèles." 
                "Tout d'abord les modèles ARIMA, Auto_arima, SARIMA et enfin SARIMAX permettant d'intégrer"
                " des données exogènes ( les données civiles + météorologiques). " 
                "Ainsi, les modèles TBATS, Prophet ont été testé pour les modèles temporels. "
                "Les modèles de regression ont été testé :"
                "regression linéaire, Gradient Boosting et Random Forest. "
                "Le meilleur des modèles adaptés au traitement de cette série temporelle est le modèle ***Sarimax***, "
                "dont la détermination des paramètres nécessite nombre de tests et une approche itérative des plus chronophage.  \n" )
    st.markdown("Toutefois, la comparaison des résultats obtenus, proches de ceux obtenus par les modèles de regression ne prenant "
                "pas en compte les temporalités nous amène à conclure que le trafic cycliste à Paris"
                "repose essentiellement sur les paramètres exogènes (temps, jour de la semaine, période de l’année).  \n")



                
    st.subheader("Conclusions et perspectives d’évolutions")
    st.markdown("Le trafic cycliste à Paris, comme nombre d’activités humaines, ne relève pas d’une cause "
                "définie et identifiée, mais d’une composante de facteurs. Notre modèle ainsi ne prend en "
                "compte que les facteurs les plus évidents, série qu’il conviendrait d’enrichir.  \n"
                )
                
    st.markdown("Ainsi, les travaux d’aménagement et d’embellissement de Paris, les chantiers dans le cadre de "
                "la préparation des JO mériteraient d’être pris en compte, on pourrait le faire en identifiant la " 
                "cartographie des travaux, en faisant un décompte quotidien.  \n "
                )
    st.markdown("Les difficultés croissantes du réseau de transport francilien, dont la presse se fait régulièrement "
                "l’écho, et dont l’origine semble venir du changement de mode de calcul du financement apporté "
                "par la Région Île-de-France, sont une réalité vécue douloureusement par les Parisiennes et les "
                "Parisiens, comme tous les autres usagers des transports publics.  \n "
                )
                
    st.markdown("L’évaluation de ces difficultés, et l’intégration de cette dimension dans la prédiction du trafic "
                "cycliste est une difficulté qui pourrait être résolue par l’examen et l’analyse des données "
                "publiques mises à disposition par les opérateurs de transports franciliens (RATP, SNCF, Veolia)."
                )
    st.markdown("**Usages possibles :**")
    st.markdown("""
                - Mieux programmer les travaux, 
                - Mieux anticiper ainsi le partage de la voirie,
                - Etudier l'accidentologie, afin de mettre des solutions en place pour le reduire.""")

    













