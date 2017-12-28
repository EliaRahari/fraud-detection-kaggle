########################################################################################################################################################
# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

JEU DE DONNEES CREDIT
Chloé VAN CAUWENBERGHE
"""

########################################################################################################################################################
###                                                               LIBRAIRIES A UTILISER                                                              ###
########################################################################################################################################################


### Pour installer une libairie inexistante sur Anaconda, aller dans l'invite 
# de commande d'Anaconda et écrire :
# pip install nom_de_la_librairie

# Librairies de manipulation de données
import pandas as pd
import numpy as np
from scipy import interp


# Librairies de visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import prince


# Librairies de machine learning
### Librairies des modèles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


### Librairies de validation des modèles
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, precision_score


# Librairies pour les regex
import re


########################################################################################################################################################
###                                                               DECLARATION VARIABLES                                                              ###
########################################################################################################################################################


# Récupération du fichier
fichier_credit = pd.read_csv("C:/Users/cvancauwenberghe/Downloads/credit.csv")


# Mélange des données
fichier_credit.sample(frac=1)
# fichier_credit = fichier_credit.head(10000)

# Nom des entetes de colonnes
temps, paiement, montant, nomOrg, soldeInitOrg = 'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg'
soldeFinalOrg, nomDest, soldeInitDest, soldeFinalDest, isFraud = 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud'
jour, difOrg, difOrgTransfo, difDest,difDestTransfo = 'jour', 'difOrg', 'difOrgTransfo', 'difDest', 'difDestTransfo'


# Variables explicatives
X_exp = fichier_credit[[temps, paiement, montant, nomOrg, soldeInitOrg, soldeFinalOrg, nomDest, soldeInitDest, soldeFinalDest]]


# Variables quantitatives
X_quant = fichier_credit[[montant, soldeInitOrg, soldeFinalOrg, soldeInitDest, soldeFinalDest]]


# Variable cible
Y_cible = fichier_credit[isFraud]

### STEP : Unité de temps dans le monde réel : 1 pas i vaut 1 heure de temps. 
# max(step) = 743 (30 jours de simulation)
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


########################################################################################################################################################
###                                                               DECLARATION FONCTIONS                                                              ###
########################################################################################################################################################


# Fonction pour récupérer la première lettre d'un champ dans un dataframe
def recuperer_premiere_lettre(cellule): 
    return re.sub(r"[0-9]", "", cellule) 
    

# Fonction de validation simple
def validation_simple(modele):
    mat = np.asarray([[0,0],[0,0]])
    # Application du modèle sur le modèle d'entrainement
    mod = modele.fit(X_train, y_train)
    # Utilisation des prédictions sur le jeu de données test
    y_pred_knn = mod.predict(X_test)
    # Création d'une matrice de confusion pour afficher la relation entre les prédictions et le réel
    mat = confusion_matrix(y_test, y_pred_knn)
    print(mat)


# Fonction de validation croisée
def validation_croisee(modele, var_exp, var_cible, xTrainMod, yTrainMod): 
    # Déclaration variables
    cf = np.asarray([[0,0],[0,0]])
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    recall = np.asarray([[0,0],[0,0]])
    precision = np.asarray([[0,0],[0,0]])
    i = 0
    # Création de la boucle pour faire la validation croisée
    for train_index, test_index in skf.split(xTrainMod, yTrainMod):    
        X_train, X_test = var_exp[train_index], var_exp[test_index]
        y_train, y_test = var_cible[train_index], var_cible[test_index]
        mod = modele.fit(X_train, y_train.ravel())       
        y_pred = mod.predict(X_test)
        # Matrice de confusion
        cf = cf + confusion_matrix(y_test, y_pred) 
        # Performance : Recall
        rec = recall_score(y_test, y_pred.round(), average='weighted')
        recall = recall + rec
        # Performance : Precision
        pre = precision_score(y_test, y_pred, average='weighted')
        precision = precision + pre
        # Création de la courbe ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))    
        i += 1
    # Affichage de la matrice de confusion
    print(cf)
    # Affichage de precision
    print(precision)
    # Affichage de rappel
    print(recall)
    # Affichage de la courbe ROC
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)   
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
# Fonction de répartition cumulative
def graph_cumul(nomColonne, titre, labelX, labelY):
    hist, bin_edges = np.histogram(nomColonne, normed=True)
    dx = bin_edges[1] - bin_edges[0]
    a = np.cumsum(hist)*dx
    plt.plot(bin_edges[1:], a)
    plt.grid()
    plt.title(titre)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
      
    
########################################################################################################################################################
###                                               PARTIE RECHERCHE D'INFORMATIONS SUR LE JEU DE DONNEES                                              ###
########################################################################################################################################################


# Voir un extrait du fichier
fichier_credit.head()


# Regarder les informations sur les données du jeu
fichier_credit.info()


# Vérifier qu'il n'y a pas de valeurs nulles
fichier_credit.isnull().values.any()


# Description du fichier
description = fichier_credit.describe()


# On regarde les différents types de transaction
transactions = fichier_credit.type.drop_duplicates().values


# On regarde quelle type de transaction sont sujettes aux fraudes.
print('\n Les types de transactions frauduleuses sont {}'.format(\
list(fichier_credit.loc[fichier_credit.isFraud == 1].type.drop_duplicates().values)))
# Les transactions frauduleuses sont CASH_OUT et TRANSFER


# On compte le nombre de transaction frauduleuse pour les transferts
nbFraudTransfert = len(fichier_credit.loc[(fichier_credit[isFraud] == 1) & (fichier_credit[paiement] == 'TRANSFER')])


# On compte le nombre de transaction frauduleuse pour le cash
nbCashTransfert = len(fichier_credit.loc[(fichier_credit[isFraud] == 1) & (fichier_credit[paiement] == 'CASH_OUT')])
print('Nb de personnes ayant fait une fraude sur les transferts : ' + str(nbFraudTransfert))
print(" et Nb de personnes ayant fait une fraude sur la monnaie : " + str(nbCashTransfert))
# 4116 cash out ont subi une fraude et 4097 transferts ont subi une fraude


# Récupération des lignes où il y a eu fraude
ligne_fraude = fichier_credit.loc[(fichier_credit[isFraud] == 1)]


# Moyenne du montant lorsqu'il y a fraude : 1 467 967 €
moy_montant_fraude = ligne_fraude[montant].mean()


# Max du montant lorsqu'il y a fraude : 10 000 000 €
max_montant_fraude = max(ligne_fraude[montant])


# Std du montant lorsqu'il y a fraude : 2 404 252 €
std_montant_fraude = ligne_fraude[montant].std()


# Nombre de personnes ayant fraudés
print((fichier_credit[isFraud]==1).value_counts())

# Nombre de personnes ayant fraudés en fonction du type de paiement
print(pd.crosstab(fichier_credit[isFraud],fichier_credit[paiement]))


########################################################################################################################################################
###                                           PARTIE VISUALISATION AVANT TRANSFORMATION DU JEU DE DONNEES                                            ###
########################################################################################################################################################


### On remarque une relation linaire entre la colonne OLDBALANCEORG et NEWBALANCEORIG qui s'explique par les opérations entre ces deux colonnes.
### Cependant à première vue il n'y a pas de relation linéaire entre les deux variables OLDBLANCEDEST et NEWBALANCEDEST

plt.plot(fichier_credit.iloc[:,4], fichier_credit.iloc[:,5])
plt.xlabel('OLDBALANCEORIG')
plt.ylabel('NEWBALANCEORIG')

# Création d'histogramme
fichier_credit.hist(column=isFraud)
fichier_credit.hist(column=paiement,by=isFraud)


# Distribution loi normale
data = fichier_credit[montant]
plt.hist(data, normed=1, color = "blue")
plt.title('Distribution standard normale sur le montant')
plt.grid()
plt.show()


### Comparaison des distributions avec un boxplot
# Boxplot sur toute les variables contenant des variables quantitatives
sns.boxplot(data=fichier_credit, orient="h", palette="Set2")


# Boxplot sur le montant
sns.boxplot(x=fichier_credit[montant])


# Boxplot sur le montant en fonction de la fraude
sns.boxplot(x=isFraud, y=montant, data=fichier_credit, order=[0, 1])


# Camembert sur la représentation des types
fichier_credit[paiement].value_counts().plot.pie()


# Fonction de répartition cumulative
graph_cumul(fichier_credit[montant], "Transactions cumulées normées - général", "Montant", "Transactions cumulées normées")


### Méthode d'Analyse en Composantes Principales
target_name = ['turquoise', 'navy']


# définition de la commande
pca = PCA()


# Estimation, calcul des composantes principales
C = pca.fit(X_quant).transform(X_quant)


# Explication du pourcentage de variance de chaque variable
print('Explication du pourcentage de variance de chaque variable: %s'
      % str(pca.explained_variance_ratio_))


# Décroissance de la variance expliquée
plt.plot(pca.explained_variance_ratio_)


# Affichage graphique
plt.boxplot(C[:,0:5])
plt.scatter(C[:,0], C[:,1], c=target_name, label=[0,1])


# Cercle des corrélations
cercle = prince.PCA(X_quant, n_components=2)
cercle.plot_correlation_circle()


# Visualisation de la matrice de corrélation
corr = (X_quant.corr())
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)


#Etude sur la variable temporelle "steps"
distinct_step = fichier_credit[temps].unique()


#Sommes échangées chronologiquement
amount_vs_time = fichier_credit.groupby(temps).amount.sum().to_frame()
amount_vs_time["date"] = amount_vs_time.index
amount_vs_time.plot(x="date", y=montant)


########################################################################################################################################################
###                                                        PARTIE ENRICHISSEMENT ET NETTOYAGE                                                        ###
########################################################################################################################################################


# Concaténation des lignes TRANSFER et CASH_OUT 
a = fichier_credit.loc[(fichier_credit[paiement] == "TRANSFER")]
b = fichier_credit.loc[(fichier_credit[paiement] == "CASH_OUT")]
fichier_credit = pd.concat([a, b])


# Création de petits dataframes pour avoir un apercu des colonnes rajoutées
infoAmount = fichier_credit[[montant, paiement, isFraud]]
amountWithFraud = infoAmount.loc[(infoAmount.isFraud == 1)]
amountWithFraudTransfer = amountWithFraud.loc[(amountWithFraud.type == "TRANSFER")]
amountWithFraudCashOut = amountWithFraud.loc[(amountWithFraud.type == "CASH_OUT")]


# 1. Mise en place de pallier pour voir des résultats sur les montants
conditions2 = [
    (amountWithFraud[montant] <= 10000),
    (amountWithFraud[montant] > 10000) & (amountWithFraud[montant] <= 50000),
    (amountWithFraud[montant] > 50000) & (amountWithFraud[montant] <= 100000),
    (amountWithFraud[montant] > 100000) & (amountWithFraud[montant] <= 500000),
    (amountWithFraud[montant] > 500000) & (amountWithFraud[montant] <= 1000000),
    (amountWithFraud[montant] > 1000000) & (amountWithFraud[montant] <= 5000000),
    ]


# 2. Mise en place de pallier pour voir des résultats sur les montants
choices2 = []    
for k in range(1, 7):
    choices2.append(k)
 

# 3. Mise en place de pallier pour voir des résultats sur les montants   
amountWithFraud['cat_montant'] = np.select(conditions2, choices2, default='0')


# Histogramme sur le montant
amountWithFraud.hist(column='cat_montant',by=paiement)


# 1. Transformation des heures en jour
conditions = [fichier_credit[temps] <= 24]
j = 24
for i in range(0,30):
    conditions.append((fichier_credit[temps] > j) & (fichier_credit[temps] <= j + 24))
    j = j + 24


# 2. Transformation des heures en jour
choices = []    
for k in range(1, 32): choices.append(k)
fichier_credit[jour] = pd.Series(np.select(conditions, choices, default='0'))


# On récupère sur les lignes de fraudes
jour2 = fichier_credit.loc[(fichier_credit.isFraud == 1)]


# Affichage par jour
d = jour2[[isFraud, jour]]
d.hist(column=jour)


# Graphique de fonction de répartition cumulée
graph_cumul(amountWithFraudCashOut[montant], "Transactions cumulées normées avec fraude sur CASH OUT", "Montant", "Transactions cumulées normées")


# Ajout d'une colonne sur l'erreur de montant avec le montant, le solde initial et final
fichier_credit[difOrg] = fichier_credit[soldeInitOrg] + fichier_credit[montant] - fichier_credit[soldeFinalOrg]
fichier_credit[difDest] = fichier_credit[soldeInitDest] + fichier_credit[montant] - fichier_credit[soldeFinalDest]


# Valeur absolue de la différence de montant
fichier_credit[difOrg] = abs(fichier_credit[difOrg])
fichier_credit[difDest] = abs(fichier_credit[difDest])


grades = []
# Catégorie de différence de montant pour l'envoyeur
for row in fichier_credit[difOrg]:
    if row <= 100000 : grades.append(0)
    elif (row > 100000) & (row <= 500000) : grades.append(1)
    else: grades.append(2)
fichier_credit['cat_difOrg'] = pd.DataFrame(grades)
fichier_credit['cat_difOrg'].value_counts().plot.pie()


grades = []
# Catégorie de différence de montant pour le destinataire
for row in fichier_credit[difDest]:
    if row <= 10000 : grades.append(0)
    elif (row > 10000) & (row <= 100000) : grades.append(1)
    else: grades.append(2)
fichier_credit['cat_difDest'] = pd.DataFrame(grades)
fichier_credit['cat_difDest'].value_counts().plot.pie()


testDifOrg, testDifDesti = [], []
# Création d'une liste pour enregistrer les données pour l'envoyeur
for row in fichier_credit[difOrg]:
    if row == 0 : testDifOrg.append(0)
    else : testDifOrg.append(1)
fichier_credit[difOrgTransfo] = testDifOrg
    

# Création d'une liste pour enregistrer les données pour le destinataire
for row in fichier_credit[difDest]:
    if row == 0 : testDifDesti.append(0)
    else : testDifDesti.append(1)
fichier_credit[difDestTransfo] = testDifDesti


# Création d'une colonne pour récupérer la première lettre du nom de l'envoyeur
fichier_credit[nomOrg] = fichier_credit[nomOrg].apply(recuperer_premiere_lettre)


# Création d'une colonne pour récupérer la première lettre du nom du receveur
fichier_credit[nomDest] = fichier_credit[nomDest].apply(recuperer_premiere_lettre)


# Histogramme sur l'ajout de colonne contenant la première lettre pour les receveurs
fichier_credit.hist(column=nomOrg,by=isFraud)


# Histogramme sur l'ajout de colonne contenant la première lettre pour les destinataires
fichier_credit.hist(column=nomDest,by=isFraud)
# Seul les clients destinataires sont victimes de fraudes, pas les marchands destinataires


# Le jeu de données à analyser au final est donc :
fichier_credit = fichier_credit[[temps, montant, soldeInitOrg, soldeFinalOrg, soldeInitDest, soldeFinalDest, difOrg, difOrgTransfo, difDest, difDestTransfo, isFraud, jour]]
X = fichier_credit[[temps, montant, soldeInitOrg, soldeFinalOrg, soldeInitDest, soldeFinalDest, difOrg, difOrgTransfo, difDest, difDestTransfo, jour]]


########################################################################################################################################################
###                                                   SELECTION DES VARIABLES ET CHOIX DES MODELES                                                   ###
########################################################################################################################################################


# Déclaration variable cible 
Y_mod = fichier_credit[isFraud]


# Découpage du jeu de données en deux parties
X_train, X_test, y_train, y_test = train_test_split(X, Y_mod, test_size=0.4)


# Vérification du choix de K pour la méthode knn
errors = []
K_max = 10
for k in range(1,K_max):
    knn = KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(X_train, y_train).score(X_test, y_test)))
plt.plot(range(1,K_max), errors, 'o-')
plt.show()


# Choix du k pour la méthode KNN 
k = 2


### Vérification des paramètres pour la méthode Random Forest
# Déclaration des paramètres que nous voulons tester
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# Déclaration de la méthode Random Forest - on déclare 30 arbres comme dans la norme
rf = RandomForestClassifier(n_estimators=30)


# Utilisation de la méthode GridSearchCV
grid_search = GridSearchCV(rf, param_grid=param_grid)
rf.fit(X_train, y_train)


# Affichage des meilleurs critères
print(rf.criterion)
# gini
print(rf.max_depth)
# None
print(rf.max_features)
# auto
print(rf.n_classes_)
# Importance des variables
print(rf.feature_importances_)


# Déclaration des variables
features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)


# Affichage de l'importance des variables
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# Méthode Random Forest
rf = RandomForestClassifier(n_estimators=30, criterion = "gini", max_depth = None, max_features = "auto")


### Modèles utilisés au final
# Méthode KNN
knn = KNeighborsClassifier(n_neighbors = k)


# Méthode Régresssion logistique
lr = LogisticRegression()


### Utilisation de l'information mutuelle
# temps, difOrgTransfo, difDestTransfo
r = mutual_info_classif(X, fichier_credit[isFraud])


# temps, soldeFinalOrg, difOrgTransfo
rfe = RFE(lr, 3)
rfe = rfe.fit(X, fichier_credit[isFraud])
print(rfe.support_)
print(rfe.ranking_)


# Jeu pour chaque modèle
XModKNN = fichier_credit[[difDestTransfo, difOrgTransfo, jour]]
XModRF = fichier_credit[[soldeInitOrg, soldeFinalDest, difOrg]]
XModLR = fichier_credit[[temps, jour, difOrgTransfo]]


########################################################################################################################################################
###                                                               VALIDATION DU MODELE                                                               ###
########################################################################################################################################################


# Validation simple
validation_simple(knn)
validation_simple(rf)


### Validation croisée avec KFold
# Choix du nombre de découpage
# kf = KFold(n_splits = 10)
skf = StratifiedKFold(n_splits=10)


Y_kf = pd.DataFrame(Y_mod).values


# Validation croisée   
validation_croisee(knn, XModKNN.values, Y_kf, XModKNN, Y_mod)
validation_croisee(rf, XModRF.values, Y_kf, XModRF, Y_mod)
validation_croisee(lr, XModLR.values, Y_kf, XModLR, Y_mod)



########################################################################################################################################################
###                                                                RESULTATS DU MODELE                                                               ###
########################################################################################################################################################


### Le modèle qui fonctionne le mieux est le modèle RF
# KNN
# [[2742292   19904]
# [   5133    3080]]

# 38% de détection

# RF - 20
#[[2761376     820]
# [   2467    5746]]

# 70% de détection

# RL
#[[2762182      14]
# [   8159      54]]


########################################################################################################################################################