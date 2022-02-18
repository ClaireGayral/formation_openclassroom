## Projet 8 : Participez à une compétition Kaggle !

*0bjectif* : Prédiction du type de couverture forestière

*Fouille des données* : analyse exploratoire classique, feature selection,  

*Modélisation* : Différentes méthodes de classification très classiques (Naive Bayes, Nearest Neighbor, SVM, Decision Tree)

*Origine des données* : [Compétition kaggle "Forest Cover Type Prediction"](https://www.kaggle.com/c/forest-cover-type-prediction/data)

L'AUC pour différents prétraitements est le suivant : 
<p align="center">
  <img src="https://github.com/ClaireGayral/formation_openclassroom/blob/master/images/P8_compare_classif_models_scores.png" alt="Sublime's custom image"/>
</p>

Les lignes 0 à 3 correspondent respectivement à : 
* dataset 0 : le jeu de données initial, seules les variables constantes ont été retirées
* dataset 1 : les variables catégorielles et numériques non importantes sont retirées,
* dataset 2 : les variables catégorielles sont sélectionnées, et les variables numériques sont prétraitées
* dataset 3 : les variables catégorielles sélectionnées sont projetées dans un espace orthonormé adapté (de
dimension 20), et les variables numériques sont prétraitées

Les datasets 4, 5, 6 et 7 correspondent aux datasets précédents orthonormalisés par PCA.
