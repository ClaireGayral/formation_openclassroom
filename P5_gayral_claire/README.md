## Projet 5 : Catégorisez automatiquement des questions

*0bjectif* : Développez un système de suggestion de tag pour les questions et postes de Stackoverflow

*Fouille des données* : Différents prétraitements propres au NLP (ex stopwords ou radicalisation), différentes représentations (bag of words, tf-idf)

*Modélisation* : 
* Non supervisée : 
    * NMF utilisée pour regrouper les mots par sémantique, donnée par leur distribution dans les postes. Aussi utilisée comme prédicteur de tags, mais annotation fastidieuse des topics.
    * LDA : donne de meilleurs résultats que la NMF
* Supervisée : Après réduction de dimension (clustering hiérarchique ou choix de quelques tags)
    * Naive Bayes, Gradient Boosting, Random Forest
    * Classification binaire ou multi-classe
* API flask pour les résultats 
  

*Origine des données* : [Outil de requêtes SQL sur la BDD de stackoverflow](https://data.stackexchange.com/stackoverflow/query/new)

En plus des notebooks documentés (mais non nettoyés), il était demandé de rédiger un rapport résumant le travail.



