# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

JEU DE DONNEES CREDIT
"""

###############################################################################
###                          LIBRAIRIES A UTILISER                          ###
###############################################################################

# Librairies de manipulation de données
import pandas as pd
import numpy as np
import scipy as sc
from scipy.cluster.hierarchy import dendrogram, linkage

# Librairies de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Librairies de machine learning
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Librairies pour les regex
import re


###############################################################################
###                          DECLARATION VARIABLES                          ###
###############################################################################

# Récupération du fichier
user = "mrahari.TECH"
disk_space = "C:/Users/"
path = disk_space + user +"/Desktop/creditFraud/data/paysim-data.csv"
fichier_credit = pd.read_csv(path)

# Extraction des 100 premières lignes pour faciliter le chargement
fichier_credit = fichier_credit.head(100)

# Récupération du nom des colonnes dans un dataframe
colonne_credit = pd.DataFrame(fichier_credit.columns)

# Informations sur la taille du dataframe
n = fichier_credit.shape
n_lignes = n[0]
n_colonnes = n[1]

### STEP : Unité de temps dans le monde réel : 1 pas i vaut 1 heure de temps. 
# max(step) = 744 (30 jours de simulation)
### TYPE : Moyen de paiement utilisé
### AMOUNT : Montant d'une transaction dans la monnaie locale
### NAMEORIG	: Consommateur qui a engagé la transaction
### OLDBALANCEORIG : Bilan initial avant la transaction du consommateur
### NEWBALANCEORIG	: Nouveau bilan après la transaction du consommateur
### NAMEDEST	 : Personne/organisme qui recoit la transaction
### OLDBLANCEDEST : Bilan initial avant la transaction du receveur
### NEWBALANCEDEST :	 Nouveau bilan après la transaction du receveur
### ISFRAUD : Transactions tests simulées qui ont abouties ou non à une fraude.
# La fraude est faite par les receveurs.
### ISFLAGGEDFRAUD : Test d'un modèle de la fraude ou non.


###############################################################################
###                          DECLARATION FONCTIONS                          ###
###############################################################################

# Fonction pour récupérer la première lettre d'un champ dans un dataframe
def recuperer_premiere_lettre(cellule): 
    return re.sub(r"[0-9]", "", cellule)


###############################################################################
###          PARTIE RECHERCHE D'INFORMATIONS SUR LE JEU DE DONNEES          ###
###############################################################################

# Regarder les informations sur les données du jeu
fichier_credit.info()

# Vérifier qu'il n'y a pas de valeurs nulles
fichier_credit.isnull().values.any()

# Description du fichier
description = fichier_credit.describe()

# On regarde les différents types de transaction
transactions = fichier_credit.type.drop_duplicates().values

# On regarde quelle type de transaction sont sujettes aux fraudes.
print('\n The types of fraudulent transactions are {}'.format(\
list(fichier_credit.loc[fichier_credit.isFraud == 1].type.drop_duplicates().values)))

# On compte le nombre de transaction frauduleuse pour les transferts
nbFraudTransfert = len(fichier_credit.loc[(fichier_credit.isFraud == 1) & 
                                   (fichier_credit.type == 'TRANSFER')])
print('Nb de personnes ayant fait une fraude sur les transferts : ' 
      + str(nbFraudTransfert) + '')
# On compte le nombre de transaction frauduleuse pour le cash
nbCashTransfert = len(fichier_credit.loc[(fichier_credit.isFraud == 1) & 
                                   (fichier_credit.type == 'CASH_OUT')])
print("Nb de personnes ayant fait une fraude sur la monnaie : " 
      + str(nbCashTransfert) + "")

# Nombre de personnes ayant fraudés
print((fichier_credit['isFraud']==1).value_counts())

# Nombre de personnes ayant fraudés en fonction du type de paiement
print(pd.crosstab(fichier_credit['isFraud'],fichier_credit['type']))


###############################################################################
###       PARTIE VISUALISATION AVANT TRANSFORMATION DU JEU DE DONNEES       ###
###############################################################################

### On remarque une relation linaire entre la colonne OLDBALANCEORG et 
# NEWBALANCEORIG qui s'explique par les opérations entre ces deux colonnes.
### Cependant à première vue il n'y a pas de relation linéaire entre les deux 
# variables OLDBLANCEDEST et NEWBALANCEDEST
plt.plot(fichier_credit.iloc[:,4], fichier_credit.iloc[:,5])
plt.xlabel('OLDBALANCEORIG')
plt.ylabel('NEWBALANCEORIG')

# sns.set(style="ticks")
# df = sns.load_dataset("fichier_credit")
# sns.pairplot(df, hue="species")

# Z = linkage(fichier_credit, 'ward')
# plt.scatter(fichier_credit[:, 4], 
#             fichier_credit[:, 5], 
#             c=fichier_credit.isFraud)


# Création d'histogramme
fichier_credit.hist(column='isFraud')
fichier_credit.hist(column='type',by='isFraud')

# Comparaison des distributions avec un boxplot
fichier_credit.boxplot(column='type',by='isFraud')

# Création d'un nuage de point
fichier_credit.plot.scatter(x='type',y='isFraud',c='isFraud')

# Camembert sur la représentation des types
fichier_credit['type'].value_counts().plot.pie()

# Equivalent de pairs de R - affichage des données
# Cela n'a d'intérêt que pour les variables quantitatives 
pd.tools.plotting.scatter_matrix(
        fichier_credit.select_dtypes(exclude=['object']))


###############################################################################
###                   PARTIE ENRICHISSEMENT ET NETTOYAGE                    ###
###############################################################################

a = fichier_credit

# Création d'une colonne pour récupérer la première lettre du nom de l'envoyeur
fichier_credit['FirstnameOrig'] = fichier_credit['nameOrig'].apply(
        recuperer_premiere_lettre)

# Création d'une colonne pour récupérer la première lettre du nom du receveur
fichier_credit['FirstnameDest'] = fichier_credit['nameDest'].apply(
        recuperer_premiere_lettre)


###############################################################################
###       PARTIE VISUALISATION APRES TRANSFORMATION DU JEU DE DONNEES       ###
###############################################################################



###############################################################################
###                         SELECTION DES VARIABLES                         ###
###############################################################################



###############################################################################
###                             CHOIX DU MODELE                             ###
###############################################################################

# Méthode Régression Linéaire
lr = LinearRegression(normalize=True)
# Méthode KNN - voisins = 
knn = KNeighborsClassifier(n_neighbors=5)


###############################################################################
###                           VALIDATION DU MODELE                          ###
###############################################################################

# Découpage du jeu de données en deux parties
X_train, X_test, y_train, y_test = train_test_split(
        fichier_credit.data, fichier_credit.isFraud, test_size=0.4)

# Application du modèle sur le modèle d'entrainement
knn.fit(X_train, y_train)

# Utilisation des prédictions sur le jeu de données test
y_pred = knn.predict(X_test)

### Création d'une matrice de confusion pour afficher la relation entre les
# prédictions et le réel
confusion_matrix(y_test, y_pred)




###############################################################################
###                    AFFICHAGE DES RESULTATS DU MODELE                    ###
###############################################################################




###############################################################################
###                                   TEST                                  ###
###############################################################################









