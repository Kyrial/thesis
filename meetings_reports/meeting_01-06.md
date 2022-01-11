# Réunion du 01/06/2021 avec Julien et Sonia

- Lire le papier de Mao

- sur une image, plot de la distribution ds activations par couche (le fair eplusieurs fois, mais par image)
- idem sur un batch de 50 images (comme déja fait) mais préciser les métriques (toutes) dans le titre, avec les corrélations

- pour les corrélations: plotter avant, pour voir si pearson est le mieux, maybe tester spearman

- trouver une métrique de granularité (cf papier de Mao 2017)

- faire un gdoc avec les résultats (mettre la date dans le titre + sauvegarde automatique? )

- rajouter la norme l0 en tant que métrique (ou simplement compter le nombre de 0)

- laisser tomber l1

- métrique sur l'évolution de la métrique sur le diag en baton:
    + faire la différence entre le plus grand baton et le plus petit (ie max et min de la sigmoide tant est qu'il y ai une sigmoide)
    + calculer la pente max sur la partie croissante de la sigmoîde, (ie max d ela dérivée) --> y a t'il une métrique pour faire ça sur une évolution discrète --> biblio