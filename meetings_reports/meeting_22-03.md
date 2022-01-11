# Réunion 1
22/03/2021
Julien/Nicolas

**Quelles sont les variables caractérisant l'efficacité du traitement de l'info corrélées avec des notes de beauté ?**

##1.Approches

###1.1.Mesurer la sparsité moyenne du réseau

- moyenne par couche (on a un hypercube donc mesuré sur plusieurs métriques par rapport à l'hypercube des features)
    → pour obtenir des valeurs, on réduit à deux dimensions, profondeur en x, R*L en y
- pour chaque couche pour l'ensemble
- de l'ensemble

###1.2.Augmentation de la fluence comme du "suspens"

- plot de la sparsité en fonction de la couche: la sparsité va augmenter 
- mais comment l'augmente-t- elle? Est ce que le type d'accélération est corrélé à la beauté? genre l'accélération = gradient


###1.3.Sparsité contextuelle

- relative à la sparsité moyenne de la population 

##2.BDD

###2.1.Portraits (tableaux)

- Jen Aesthetic subjective database (art divers, pas de portraits), un papier est lié
- MART : spécifiquement des peintures abstraites avec des jugements de beauté dessus

###2.2.Visages (photos)

- chicago faces database, ne prendre que les photos "neutres" (pas de sourires etc...), et un csv avec des caractéristiques (féminité, confiance et attractivité)
- BeautyNet: modèle de prédiction de la beauté: quel jeu de données ? CASIA WEB DATABASE, LSFBD (visages asiatiques)
- SCUT-FBP: a vérifier

##3.Réseau

- VGG ou Resnet (150 couches) assez simples mais suffisants 
- VGG16 sans boucle de récurrence

##4.Tensorflow

###4.1.Tips

- Problème de Tensorflow: différentes manières de faire les choses: genre soit natif, sois tf.data, sois keras → ne pas coller plein de bouts de code qui viennent d'endroits différents mais prendre tout le code d'un endroit et l'adapter tranquillement. 

- addons: trucs pas encore inclus par TF mais plus ou moins reconnues par la communauté

###4.2.Embedding

embedding = vecteur de l’avant dernière couche (la couche avant le classifier)

pour avoir l'embedding, faire top = false sur tf (top étant la couche de classification)

conseil: partir de facenet pour extraire les embedding de nouveaux visages

keywords: extract embedding deep neural networks

attention: terme embedding a un double sens: utilisé aussi en analyse textuelle, rien a voir. 
Le bon embedding est aussi appelé encoding.  

se renseigner sur les normalisations L1 et L2 sur les embeddings

quand top=false, model.predict donne l'embedding

→ voir comment extraire les activations des couches intermédiaires car model.predict ne donnera que l'embedding, cad le vecteur final. 

###4.3.Triplet loss

Différent de la classification, ici c'est de la face re identification: tu lui donne deux visages sur lesquels le réseau n'a pas été entraîné, on veut savoir si c'est la même personne. 

Triplet loss: le modèle a la base ne prend pas une ilage et son label, mais 3 images sous la forme de deux paires d'images, deux identiques (deux images diff mais même personne) , deux différentes 

→ Est-ce que si on fait ça, il y a une influence sur le lien entre beauté et paramètre étudié? 

##5.Bibliographie:

bot telegram pour sci hub

The unreasonable effectiveness of deep features as a perceptual as a perceptual metric, R Zhang

##6.Divers

L’attractivité diffère de la beauté qui n'en est qu'un aspect

swiss roll deep learning (ressemble a umap)
