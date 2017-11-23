# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

JEU DE DONNEES CREDIT
"""

###############################################################################
###                          LIBRAIRIES A UTILISER                          ###
###############################################################################

### Pour installer une libairie inexistante sur Anaconda, aller dans l'invite 
# de commande d'Anaconda et écrire :
# pip install nom_de_la_librairie

# Librairies de manipulation de données
import pandas as pd
import numpy as np
# from scipy.cluster.hierarchy import dendrogram, linkage

# Librairies de visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import prince

# Librairies de construction de graphes
# import networkx as nx

# Librairies de selection de variables
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

# Librairies de machine learning
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Librairies pour les regex
import re


###############################################################################
###                          DECLARATION VARIABLES                          ###
###############################################################################

# Récupération du fichier
fichier_credit = pd.read_csv("C:/Users/cvancauwenberghe/Downloads/credit.csv")

# Extraction des 100 premières lignes pour faciliter le chargement
# fichier_credit = fichier_credit.head(100)

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
### ISFRAUD : Fraude effectuée sur les comtpes des émetteurs et recepteurs.
# Dans la suite du document, le mot "fraude" va être utilisé pour cette colonne
### ISFLAGGEDFRAUD : Mis à 1 si la fraude est supérieure à 200.000€


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

# Voir un extrait du fichier
fichier_credit.head()

# Description du fichier
description = fichier_credit.describe()

# On regarde les différents types de transaction
transactions = fichier_credit.type.drop_duplicates().values

# On regarde quelle type de transaction sont sujettes aux fraudes.
print('\n Les types de transactions frauduleuses sont {}'.format(\
list(fichier_credit.loc[fichier_credit.isFraud == 1].type.drop_duplicates().values)))

# On compte le nombre de transaction frauduleuse pour les transferts
nbFraudTransfert = len(fichier_credit.loc[(fichier_credit.isFraud == 1) & 
                                   (fichier_credit.type == 'TRANSFER')])
print('Nb de personnes ayant fait une fraude sur les transferts : ' 
      + str(nbFraudTransfert) + '')
# 4116 cash out ont subi une fraude

# On compte le nombre de transaction frauduleuse pour le cash
nbCashTransfert = len(fichier_credit.loc[(fichier_credit.isFraud == 1) & 
                                   (fichier_credit.type == 'CASH_OUT')])
print("Nb de personnes ayant fait une fraude sur la monnaie : " 
      + str(nbCashTransfert) + "")
# 4097 transferts ont subi une fraude

# Récupération des lignes où il y a eu fraude
ligne_fraude = fichier_credit.loc[(fichier_credit.isFraud == 1)]
# Moyenne du montant lorsqu'il y a fraude : 1 467 967 €
moy_montant_fraude = ligne_fraude['amount'].mean()
# Max du montant lorsqu'il y a fraude : 10 000 000 €
max_montant_fraude = max(ligne_fraude['amount'])
# Std du montant lorsqu'il y a fraude : 2 404 252 €
std_montant_fraude = ligne_fraude['amount'].std()

# Nombre de personnes ayant fraudés
print((fichier_credit['isFraud']==1).value_counts())

# Nombre de personnes ayant fraudés en fonction du type de paiement
print(pd.crosstab(fichier_credit['isFraud'],fichier_credit['type']))

print(pd.crosstab(fichier_credit['isFraud'],fichier_credit['amount']))


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

# Z = linkage(fichier_credit, 'ward')
# plt.scatter(fichier_credit[:, 4], 
#             fichier_credit[:, 5], 
#             c=fichier_credit.isFraud)


# Création d'histogramme
fichier_credit.hist(column='isFraud')
fichier_credit.hist(column='type',by='isFraud')

# Distribution loi normale
data = fichier_credit['amount']
plt.hist(data, normed=1, color = "blue")
plt.title('Distribution standard normale sur le montant')
plt.grid()
plt.show()

### Comparaison des distributions avec un boxplot
# Boxplot sur toute les variables contenant des variables quantitatives
sns.boxplot(data=fichier_credit, orient="h", palette="Set2")
# Boxplot sur le montant
sns.boxplot(x=fichier_credit["amount"])
# Boxplot sur le montant en fonction de la fraude
sns.boxplot(x="isFraud", y="amount", data=fichier_credit, order=[0, 1])

# Création d'un nuage de point
# fichier_credit.plot.scatter(x='type',y='isFraud',c='isFraud')

# Camembert sur la représentation des types
fichier_credit['type'].value_counts().plot.pie()

# Equivalent de pairs de R - affichage des données
# Cela n'a d'intérêt que pour les variables quantitatives 
# pd.tools.plotting.scatter_matrix(
#         fichier_credit.select_dtypes(exclude=['object']))

# Fonction de répartition cumulative
hist, bin_edges = np.histogram(fichier_credit['amount'], normed=True)
dx = bin_edges[1] - bin_edges[0]
a = np.cumsum(hist)*dx
plt.plot(bin_edges[1:], a)
plt.xlabel("Montant")
plt.ylabel("Transactions cumulées normées")

### Méthode d'Analyse en Composantes Principales
X = fichier_credit[['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest']]
Y = fichier_credit['isFraud']
target_name = ['turquoise', 'navy']
# définition de la commande
pca = PCA()
# Estimation, calcul des composantes principales
C = pca.fit(X).transform(X)
# Explication du pourcentage de variance de chaque variable
print('Explication du pourcentage de variance de chaque variable: %s'
      % str(pca.explained_variance_ratio_))
# Décroissance de la variance expliquée
plt.plot(pca.explained_variance_ratio_)
# Affichage graphique
plt.boxplot(C[:,0:5])
plt.scatter(C[:,0], C[:,1], c=target_name, label=[0,1])

# Cercle des corrélations
cercle = prince.PCA(X, n_components=2)
# cercle.plot_correlation_circle()


###############################################################################
###                   PARTIE ENRICHISSEMENT ET NETTOYAGE                    ###
###############################################################################

# Création d'une colonne pour récupérer la première lettre du nom de l'envoyeur
fichier_credit['nameOrig'] = fichier_credit['nameOrig'].apply(
        recuperer_premiere_lettre)

# Création d'une colonne pour récupérer la première lettre du nom du receveur
fichier_credit['nameDest'] = fichier_credit['nameDest'].apply(
        recuperer_premiere_lettre)

### Processus de transformation de variables non numériques
le = preprocessing.LabelEncoder()
# Application des labels à la colonne type
le.fit(fichier_credit['type'])
# Vérification des classes de la colonne type
list(le.classes_)
# Transformation des variables non numérique en variables numériques
fichier_credit['type'] = le.transform(fichier_credit['type'])
### Si on veut réinverser le processus 
# g = list(le.inverse_transform(fichier_credit['type_num']))

### Emetteur
# Application des labels à la colonne emetteur
le.fit(fichier_credit['nameOrig'])
# Transformation des variables non numérique en variables numériques
fichier_credit['nameOrig'] = le.transform(fichier_credit['nameOrig'])

### Receveur
# Application des labels à la colonne receveur
le.fit(fichier_credit['nameDest'])
# Transformation des variables non numérique en variables numériques
fichier_credit['nameDest'] = le.transform(fichier_credit['nameDest'])


###############################################################################
###       PARTIE VISUALISATION APRES TRANSFORMATION DU JEU DE DONNEES       ###
###############################################################################

### Histogramme suite à l'ajout de colonne contenant la première lettre
# Histogramme sur les receveurs
fichier_credit.hist(column='nameOrig',by='isFraud')

# Histogramme sur les receveurs
fichier_credit.hist(column='nameDest',by='isFraud')


###############################################################################
###                         SELECTION DES VARIABLES                         ###
###############################################################################

### Il n'y a aucune valeur ajoutée sur les envoyeurs, que ce soit via le 
# matricule en entier ou bien la première lettre.
### 

### Utilisation de la méthode du Chi2
# amount, oldbalanceDest, newbalanceDest sont les colonnes choisies
X_new = SelectKBest(chi2, k=3).fit_transform(X, Y)
### Utilisation de l'information mutuelle
# amount, oldbalanceOrg et newbalanceDest sont les colonnes choisies
r = mutual_info_classif(X, fichier_credit['isFraud'])
### Utilisation de la méthode supprimant les variances basses
# amount, oldbalanceOrg et newbalanceOrig sont les colonnes choisies
sel = VarianceThreshold(threshold=np.var(X))
r2 = sel.fit_transform(X)

### Il semblerait que les colonnes amount et oldbalanceOrg
# Soients les colonnes à selectionner. 
# (a voir en plus avec le cercle des corrélations)
# type également 

### On selectionne donc les variables amount, oldbalanceOrg et type
X_choisies = fichier_credit[['amount', 'oldbalanceOrg', 'type']]


###############################################################################
###                             CHOIX DU MODELE                             ###
###############################################################################

# Méthode Régression Linéaire
lr = LinearRegression(normalize=True)
# Méthode KNN
knn = KNeighborsClassifier(n_neighbors=5)
# Méthode Forets aléatoires
rf = RandomForestClassifier()
# Méthode Kmeans - non-supervisé
kmeans = KMeans(n_clusters=2, random_state=0)


###############################################################################
###                           VALIDATION DU MODELE                          ###
###############################################################################


X_choix = fichier_credit[['amount', 'oldbalanceOrg', 'type', 'isFraud']]

# Découpage du jeu de données en deux parties
X_train, X_test, y_train, y_test = train_test_split(
        X_choix, X_choix.isFraud, test_size=0.4)

### Méthode KNN
# Application du modèle sur le modèle d'entrainement
m_knn = knn.fit(X_train, y_train)
# Utilisation des prédictions sur le jeu de données test
y_pred_knn = m_knn.predict(X_test)
### Création d'une matrice de confusion pour afficher la relation entre les
# prédictions et le réel
cf_knn = confusion_matrix(y_test, y_pred_knn)

### Méthode Régression Linéaire
# Application du modèle sur le modèle d'entrainement
m_lr = lr.fit(X_train, y_train)
# Utilisation des prédictions sur le jeu de données test
y_pred_lr = m_lr.predict(X_test)
### Création d'une matrice de confusion pour afficher la relation entre les
# prédictions et le réel
cf_lr = confusion_matrix(y_test, y_pred_lr.round())

### Méthode Random Forest
# Application du modèle sur le modèle d'entrainement
m_rf = rf.fit(X_train, y_train)
# Utilisation des prédictions sur le jeu de données test
y_pred_rf = m_rf.predict(X_test)
### Création d'une matrice de confusion pour afficher la relation entre les
# prédictions et le réel
cf_rf = confusion_matrix(y_test, y_pred_rf)

### Méthode KMeans
m_kmeans = kmeans.fit(X_choix)
l = kmeans.labels_


###############################################################################
###                            RESULTATS DU MODELE                          ###
###############################################################################

### Le modèle qui fonctionne le mieux est le modèle 
# Régression Linaire/Random Forest





