# Réunion du 09/04/2021 avec Julien et Sonia

## 1. bilan comité de thèse de Sonia

Rufin VanRullen conseille de forcer la sparsité lors de l'entrainement (grace a la loss) comme julien l'a fait par le passé *(cf papier 2016)*

méthode expliquée dans le livre de A.Géron (Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow) dans le chapitre sur les auto encoders, spécifiquement celui sur les sparse auto encoder.

## 2. retour travail effectué

- on peut recoder la fonction preprocess() pour que ce soit bcp plus rapide (cf livre de A.Geron) --> passer par tf.data

### 2.1. choix de la structure de données du calcul de la norme L1

- prendre une couche par bloc  (4 blocs + la couche flatten) 

- la dernière couche de convolution de chaque bloc (2,4,7,10,13) et la dernière dense (dense 2) (dense = fully connected)

- le pooling arrive a la fin de chaque bloc mais en terme biologique, le pooling correspond aux synapses afférentes de la couche d'avant qui viennent sur le nouveau neurone **(reprendre peut-être ce point, je n'ai pas tellement compris)**

- pour chaque couche d'activation, il y a autant de carte d'activation qu'il y a de filtre

- regarder dans le chapitre du livre de Géron consacré aux sparse autoencoder (cf ligne 1) comment est intégrée l'information des couches intermédaires qui sont des matrices 3d.Par exemple, quand la loss calcule la norme l1, c'est sur l'ensemble ou sur un bout? 

- Dans un premier temps calculer la norme l1 de chaque carte d'activation pour chaque filtre et faire la moyenne sur le bloc 

- une carte d'activation est une collection de neurones

- exemple: *block 3 conv 3* : matrice (256 * 256 * 56) neurones, calculer la norme L1 sur tous, indépendament de à quelle carte d'activation ils appartiennent en faisant une sorte de flatten. **(c'est différent de l'avant-avant-dernier point ça non? à reprendre)**

### 2.2. divers

- norme L1: 
    + avantage: dérivable en tout point donc fonctionne bien en descente de gradient
    + incvonénient: avec cette norme, un neurone a 0,001 ne va pas peser dedans car la norme l1 compte le nombre de 0

- regarder la distribution des poids cad des activations pour savoir comment calculer le kurtosis : symétrique ou pas? centré sur quoi? 



















