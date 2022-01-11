# Réunion du 16/04/2021 avec Julien et Sonia

## 1.Bilan du travail effectué

- pas de corrélation entre les bases de données mergées et la beauté avec la mesure de la norme L1 sur les couches flatten


## 2.Travail a faire pour la suite

- regarder spécifiquement pour chaque bdd
- indice de Gini
- uttiliser alternativement a la norme l1 la formule de ce papier: https://www.researchgate.net/publication/11783584_Characterizing_the_sparseness_of_neural_codes
- essayer aussi ces autres mesures: https://www.sciencedirect.com/science/article/pii/S0959438804001035?casa_token=mAxGwggQKFAAAAAA:W-9HweTz1vBHv3Nspmf8zxU2PQiC_B-aiMiL0PZQOWqG7pSAdtkUtiXZF5ma7W_E2aDWXg7UoU3M
- faire des graphes des résultats
- regarder certes la L1 sur les couches flatten, mais aussi sur la moyenne de chaque channel indépendament (cela permettrait de s'affranchir d'une éventuelle pondération des channels)
- se renseigner sur le type de normalisation de vgg : si introuvable, faire l'histogramme de la distribution des activations par channel sur quelques images

