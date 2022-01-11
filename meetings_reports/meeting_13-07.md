# Réunion du 09/04/2021 avec Julien et Sonia


## python
- mesurer la complexité et l'entropie avec la formule du PNAS que m'a donné julien sur les maps (computation = filter)
- refaire les plots des R2 par couche pour gini avc toutes les combinaisons

## R
- centrer-réduire les métriques pour chaque couche (z transformation) pour pouvoir comparer les tailles d'effet
- regarder quelles variables restant après le fit du modèle (si ce sont toujours les mêmes indépendament des paramètres
- birckhoff2: tester en rajoutant les effets simples, normalement c'est la même chose que si on fait l'interaction avec la complexité
- regarder le sens de l'effet de la complexité (R positif ou négatif)
- revérifier les effets quadratiques, sur tous les paramètres (avec juste gini flatten sur le modèle avec l'interaction)


## divers
- laisser tomber le kurtosis qui a l'air inutile
- pareil pour la norme L0 pour le moment, qui est pas mal, mais donne des résultats similaires au coeff de gini, en moins bien
- matrice de corrélations croisées entre les BDD (éclairicir ce point)

## possibilité pour la suite:

- SCUT donne les notes par évaluateur, regarder si la variance des notes d'une image est corrélée au pic de sparseness dans les couches
Hypothèse: image consensuelle = variance faible = sparseness élevée dès les premières couches