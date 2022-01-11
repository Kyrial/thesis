# Lab notebook

## 22/03/2021

- réunion avec Julien: mise en place du stage, cf CR de réunion dans /meetings

## 23/03/2021

- convention signée par le CNRS: ne manque plus que la fac
- création de l'environnement de travail, un github

### BDD
- Jen: demande envoyée, en attente d'une réponse
- Chicago: demande envoyée, en attente d'une réponse
- MART: introuvable
- SCUT-FBP: téléchargée (227 MO)

### Sonia
- Architecture VGGface: https://github.com/rcmalli/keras-vggface
- son git: https://github.com/soniamaitieo/vggface_tripletloss
- plein d'autres trucs: cf mails du 23/03

### Alexandre
- coefficient de Gini pour mesuré la sparsité
- Hurley and Richard, 2008, IEEE --> introuvable

## 24/03/2021

### BDD
- Jen: 
	+ reçue, téléchargée
	+ liens de téléchargement des images, pas les images directement, à reprendre
	+ quelle différence entre les traits "beauty" et  "asesthetic quality" ? regarder biblio
	
- CFD: 
	+ reçue, téléchargée

- SCUT: 
	+ reçue, téléchargée

##25/03/2021

### BDD
- MART: fournie en local par Julien, introuvable sur internet

### Data management:

- création d'un répertoire dédié dans /code
		
- **CFD**: 
	+ *IMAGES*:supprimer les non neutres (N), regrouper toutes les images  dans le répertoire /data/redisigned/CFD/images (script: */code/data_management/data_management_pictures_CFD.py*) **DONE**
	+ *LABELS*  récuperer 2 colonnes:  ID,  Attractive, seulement pour les personnes se trouvant dans les images séléctionnées au point précédent (script: */code/data_management/				data_management_labels_CFD.py*) **DONE**
	
###26/03/2021

**SCUT-FBP**: nouvelle base de données trouvée, mise a jour avec 5500 images (majorité d'asiatiques, environ 10% de caucasiens dont Emma Watson)
	
###Data management:
	
- **SCUT-FBP**: 
	+ *IMAGES*:copier/coller dans le  dans le répertoire /data/redisigned/SCUT_FBP/images (script: */code/data_management/data_management_pictures_SCUT_FPB.py*) **DONE**
	+ *LABELS*  faire la moyenne des notes de tous les évaluateurs pour chaque image, créer un nouveau csv disponible dans*/data/redesigned/SCUT_FBP/labels_SCUT_FBP.csv. (script: */code/data_management/SCUT_FBP/data_management_labels_SCUT_FBP.py) **DONE**
	tw
	
##29/03/2021	

	- lecture d'articles, pc trop lent
##30/03/2021

Journée scientifique IA et biodiversité: https://www.labex-cemeb.org/fr/actualites/journee-scientifique-ia-et-biodiversite-30-mars-2021
--> similarity search comme plantnet? 

		
###Data management:
	
- **MART** : quel critère d'évaluation?  sur combien est-ce noté? 
--> *"The judgements were given according to a Likert scale of 1-to-7-points, where number 1 meant a highly negative emotion and 7 meant a highly positive emotion"*
à standardiser sur 5


	
	(...)
			
- **lire ça** :  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3330757/ *(irm sur perception esthétique)*
- **et ça** : https://www.nature.com/articles/s41593-021-00822-8
- **et ça**  :  https://www.sciencedirect.com/science/article/pii/S1077314220300333





# 01/04/2021

## réunion avec julien:

- étude de connectivité fonctionelle cérébrale (functionnal connectivity) qui mesure a quel point l'activation d'une région est éterminée par l'activité d'une autre région (= fkux d'infirmation dans e cerveau)  --> denrière veille: il y a 6 mois donc a voir

-  MEG: bonne réslution temporelle mais  mauvaise spatiale; IRM: inverse

- publi intéressants en neuroscieces sur l'efficacité de codage


## imageNet

récupération d'un réseau avec les poids d'imagenet: VGGface, suivant ceci: https://github.com/rcmalli/keras-vggface


vggface tuto: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/



#  02/04/2021

- étude des exemples de keract (/keract/examples/vgg16.py ET heat_map.py

--> il faut maintenant l'appliquer a notre BDD et tester comment récupérer les valeurs d'activation
--> chercher aussi a quoi correspondent précisément les *gradients* de keract


# 06/04/2021

- création d'un groupe zotero pour la biblio
- lecture doc keract
- problème environnement python vscode


#  07/04/2021

- finalisation démarches carte accès campus
- résolution problème  vs code
- prise en main de keract: 
	+ heatmap des données d'activation sur une image test --> très lent mais fonctionnel
	+ autre fonction de heatmap liée a skl, ne marche pas: pourquoi? qu'est elle sensée faire exactement? 


# 08/04/2021

- récupération carte accès campus
-lecture de l'introduction (1.) de la publication sur la comlexité visuelle
- publi complexité visuelle: technique utilisée pour réduire les biais des notes: *"In order to minimize the potential bias from individual ground-truth contributors and limited rating scales, we obtained the ground-truth labels using a forced-choice pairwise crowdsourcing methodology. Labels were obtained from 1,687 contributors on more than 37,000 pairs of images. The pairwise scores were then converted to absolute scores on a [0,100] rating scale using the Bradley–Terry (Bradley and Terry, 1952) and matrix completion (Candès and Recht, 2009) methods."*
- calcul des activations: array a 4 dimensions, discuter demain lors de la réunion de quelle approche avoir pour le calcul de la norme l1 et d'autres métriques (quelle structure de données? )
- passage possible (testé) en keras sans keract: performances et résultats équivalents, code a peine plus long. 

# 09/04/2021

- **9h**: réunion avec sonia et julien, bilan dans /meetings_reports/meeting_09_04.md

- codage de la norme L2 d'une couche quelconque sur une image test

# 12/04/2021

- nettoyage "a la main" des labels CFD
- renommage adapté aux label des images CFD
- séléction des couches d'intérêt pour le calcul de norme (cf consignes julien)

# 13/04/2021

- détéction des facteurs de lenteur du programme: import de keract et chargement du modèle --> rien qui n'impact directement l'extraction des activations donc pas de soucis (en théorie) dans une boucle --> on verra lors du test

- comment gérer l'allumage du CPU les jours ou je suis en télétravail? : le laisser tout le temps allumé

- bouclage sur toutes les images (a faire avec le CPU):
	+ adaptation de la base de données: les labels sont ils exactement identiques auxnoms des fichiers? (fait pour CFD, fait pour MART, déja le cas pour SCUT-FBP)
	+ créer un dossier avec un csv avec toutes les notes, indépendament de l'origine (pas de dossier avec toutes les images, elles seront récupérées dans les 3 rpertoires des bdd pour éviter les doublons)

- mail envoyé au Dr Saraee pour avoir les scripts de sa publi sur la compexité visuelle
- Réunion avec sonia et julien: discussion sur la méthode utilisée dans la publi sur la complexité
- comment agréger toutes les normes d'une même image? : ne pas les agréger, raisonner couche par couche
- ticket envoyé a yoann pour avoir les id root du gpu situés dans le cahier de l'ancien stagiaire qui l'a installé 

# 14/04/2021

- passage en fonctionnel (en procédural en l'occurence) du code l1_norm_activation de manière a réuttiliser les mêmes fonctions sur les différents jeux de données pour les aggréger

- choix d'un dataframe pandas en tant que structure de données dédiée au calcul des corrélation norme l1 des couches/score de beauté car il s'avère
que c'est plus optimal qu'un dict dans ce cas précis, en terme de rapidité (je ne m'y attendait pas mais cette discussion argumente en ce sens: https://www.javaer101.com/fr/article/3415827.html)

# 15/04/2021

- finalisation du code des corrélations L1/score d'attractivité
- calcul des corrélations sur toute les bases de données mergées: elles tournent autour de 0 donc rien d'intéressant

# 16/04/2021

- réunion avec Julien et Sonia pour discuter des résultats: cf compte rendu de réunion dispo dans le dossier éponyme(qu'il faut que je remette au propre d'ailleurs)

- calcul des corrélations sur chaque base de données individuellement: rien d'intéressant sur scut_fbp et mart, R = 0,3 sur CFD

# 19/04/2021

- finalisation de la connexion a distance: réinstallation de Debian sur le pc gpu afin de récupérer les id root (dispos sur un post it sur le pc, dans les notes de mon téléphone ou dans un mail envoyé a Julien et Sonia)

- le mandrill était en fait séquencé

# 20/04/2021

- résolution d'un problème avec la connexion a distance: il faut créer des paramètres pré enregistrés avec un nom (ici CEFE) sur putty, au lieu de re rentrer les paramère a chaque fois, auquel cas ça ne fonctionnne pas. 

- recherche de thèse:
	+ téléchargement du dossier récap des ANR 2020-2021
	+ epfl: prochain sujets de thèse disponibles fin septembre. 
	+ sfbi: toujours que dalle

-envoie d'un mail aux auteurs de CFD pour aveoir le sexe du jury

# 21/04/2021

- retour négatif des auteurs de CFD (ils n'ont pas collecté cette donnée)
- journée peu productive

# 22/04/2021

- finalisation de la publi sur les approches Sheringtoniennes et Hopfieldiennes: concepts très intéressants mais bcp de blabla. 
- début de la lecture de la publi sur les putatives ratio
- relance de l'autrice de la publi sur la complexité des images pour obtenir son code. : nouveau mail aux co autrices jalal@cs.wisc.edu betke@bu.edu

# 23/04/2021

- lectures 2 publis:
	+ kriegeskorte
	+ Putative ratio: pas convaincant, ne considère pas le images en général mais seulement les rations l/L de quelques rectangles

- création de sous dossier sur le repo zotero

- écoute de ce podcast: https://avisdexperts.ch/videos/view/9379

# 26/04/2021
 semil
- réponses de la co-autrice (Mona Jalal) de la publi sur la complexité --> la première autrice serait en train de réécrite le code et me le renverra via github (a suivre) 
- début de la lecture de la publi sur les matrices de dissimilarité: toolbox matlab dispo, recodage python plus simple? 

# 27/04/2019

- lecture publi matrice dissimilarité --> en fait peu d'intérêt
- lecture tuto matlab: http://www.lps.ens.fr/~krzakala/matlab.pdf
- réu avec sonia et julien 
- code d'une fonction pour calculer la moyenne des normes L1 des valeurs d'activation deschannels d'une couche
- adaptation globale du code pour passer en paramètre si l'on veut un calcul en flatten ou en moyenne de channel pour chaque couche


# 28/04/2021

- journée peu productive

# 29/04/2021

- importer les images de jen: dev d'un code fonctionnel mais le réseau du CNRS bloque: le faire tourner a la maison
- création de paramètres au début du programme facilitant son utilisation (poids, model, bdd etc... )

# 30/04/2021
 
- optimisation du code, petites modifications par ci par là (amélioration du log, etc...)

# 03/05/2021

- téléchargement de JEN en ayant géré les échecs (65 en l'occurence, c'est acceptable sur 1600 images)
- amélioration des paramètres du code des activations
- erreur avec VGGFace a comprendre et corriger

# 04 au 10/05/2021

- vgg face implémenté: même architecture mais implémentation difféente (quelle idée) ce qui expliquait les soucis
- il y avait une erreur de dimension concernant les channels: erreur corrigée
- suite a cette correction, les activations avec les méthodes channel et flatte sont en fait les mêmes: les channels ne sont donc pas pondérés pour vgg
--> n'uttiliser que flatten du coup (pour vgg)
- implémentation de Treve-Rolls: fonctionnel mais hyper lent ce qui la rend inutilisable sur les vraies bdd hors test:
	+ essayer avec reduce(), conseillé par alexandre
	+ regarder sur 30 images si TR corrèle avec L1, sinon ça vaut pas le coup selon julien (moi je trouve que si
	)
- impossible d'utiliser un disque dur externe (pour importer les donées) sur le pc gpu --> peut être que NS sait
- discussion avec une étudiante en psycho qui avait une bdd potentiellement intéressante: en fait non. On retiendra qu'il y a un master IA-psycho à Chambéry. 
- résultats sur la norme L1 avec CFD (pas avec TR)
-JEN: images trop grosses, preprocess impossible, régler ça

# 12/05/2021

- graphe des corrélations entre score et norme L1 pour toutes les bases de données et toutes les couches:
	+ avec le R (corrélation)
	+ avec aussi le R^2
--> résultats particulièrement intéressants avec CFD, moins avec les autres

# 13/05/2021

- développement d'une fonction treve-rolls alternative avec reduce(), mais les tests semblent incohérents, vérifier qu'elle fait bien ce qui est demandé
- création d'une bdd "bigtest" contenant 50 images de  
- calcul de la corrélation entre TR et L1 sur bigtest

# 14/05/2021

- idée pour faire fonctionner treve rolls avec reduce: tricker la récursivité en ne l'appliquant pas au terme A, et en le faisant en dehors de reduce pour l'index 0 du vecteur
- calcul de TR sur CFD avec imagenet et vgg face: résultats surprenants

# 17/05/2021

- découverte d'un dysfonctionnement de git
- calcul de TR sur MART avec imagenet et vggface: résultats encore plus surprenants
- graphes:
	+ **L1** Graphe sur toutes les bdd et tous les poids (un graphe rvalue, un graphe r2)
	+ **TR** Graphe sur toutes les bdd (que mart et cfd) et tous les poids (un graphe rvalue, un graphe r2)
	+ **comparaison** 2 graphe par bdd (r et r2) pour comparer L1 et TR

# 18/05/2021

- résolution du pb git: création d'un nouveau repo, perte des commits du coup. 
- souci avec le pc gpu: souris/clavier non detectés + xfce met super longtemps à se lancer
- création du fichier de labels pour jen
- calcul des normes L1 avec jen
- réunion avec Julien et Sonia
- calcul vite fait de gini sur cfd: résultats bizarres
- début d'un programme pour représenter les distributions

# 19/05/2021

 - préparation et envoi d'une candidature à une thèse

 # 20/05/2021

- PC GPU: la souris et le clavier ne sont reconnus qu'en root sous xfce, pas en utilisateur normal
- histogramme des distribution des activations sur quelques images (infaisable sur une bdd toute entière, processus interrompu)

# 21/05/2021

- migration sur le pc de Sonia, adaptation de l'environnement (packages, logiciels etc)
- souci avec les fichiers de fonctions: pas reconnus par vscode --> exécution via le terminal

# 25/05/2021

- en plus de channel et flatten, création d'une fonction de calcul sur L*l pour chaque carte d'activation
- avec cette fonction, les résultats ne changent pas avec la norme L1 mais changent complètement avec l'indice de Gini
- lancement pendant la nuit de cette fonction sur CFD, résultats a regarder le lendemain (ça a pris 18h)

# 26/05/2021

- corrélations des 4 métriques entre elles (6 mesures) et graphes
- choix de supprimer Treve-Rolls car très corrélé avec Gini

# 27/05/2021

- Comme Gini est différent selon le mode de calcul, faire 3 gini ? (flatten, channel, filter)
- standardiser en fonction des distributions? distribution sur toue une bdd impossible mais batch possible

# 28 et 31/05/2021

- préparation de la présentation
- présentation (ppt : https://onedrive.live.com/redir?resid=6D0185142D469266!748&ithint=file%2cpptx&ct=1622539777367&wdOrigin=OFFICECOM-WEB.MAIN.MRU )

# 01/05/2021

- réunion avec julien et sonia 

# 03/05/2021

- récupération du PC
- début de préparation d el'environnement
- calcul de la norme L0 sur CFD: résultats bizarres
- implmentation de vgg place et test sur JEN

# 04/06/2021

- découverte d'un site de mesure online de l'attractivié a partir d'un cnn entrainé sur scut: https://www.hownormalami.eu/
- idée: regarder si la distribution des métriques toute couche confondue est bimodale (ce qui pourrait expliquer la sigmoIde)


# 07/06/2021

- résultats JEN vgg face: gini robuste au changement de database contrairement a L1

# 08/06/2021

- régression logistique sur (x: le n de la couche, y: les valeurs des métriques

# 09/06/2021  

pas de rédaction du cahier de labo --> 10 pompes

# 10/06/2021

pas de rédaction du cahier de labo --> 10 pompes

# 11/06/2021

pas de rédaction du cahier de labo --> 10 pompes

# 14/06/2021

pas de rédaction du cahier de labo --> 10 pompes

# 15/06/2021

- Lecture et préparation de questions sur l'article Charpentier et al 2021 PNAS
- Lecture du papier de Mao sur la granularité: pas tellement intéressant
- découpage des fonctions en plus petites fonctions
- découverte de la différence entre objets muables et non muables: c'est bien après 2 ans de python

# 16/06/2021

- finalisation du découpages en fonction et en sous-packages, suppression des imports inutiles, 
- commentaires et structuration du code: un travail long et fastidieux

# 17/06/2021

biliographie sur les travaux antérieurs du projet mandrill

# 18/06/2021

- stockage les dictionnaires des métriques dans des jsons avant de calculer les moyennes pour avoir une archive et faire différnetes choses
avec sans tout recalculer a chaque fois
- séparation du programme en deux mains:
	+ un pour calculer les métriques sur les couches (super long mais une fois que ce sera fait, d'ici une diaine de jours, ça sera bon)
	+ un pour tester le modèle, plus exporatoire et "bricolage", qui uttilise les résultats du programme précédent 
	+ plot des corrélations: pearson semble effectivement plus adapté que spearman

# 21/06/2021

- lecture de quelques papiers de MC conseillés par Sonia

# 22/06/2021

- rendez vous avec Marie Charpentier pour mieux comprendre la problématique et les enjeux
- discussion avec julien sur l'idée de faire du reinforcment learning
- "régularisatio" entre 0 et 1 des métriques qui ne le sont pas, pour la régression logistique (fonction compress_metrics())

# 23/06/2021

- matinée en congés, après midi: lecture rapide de quelques publis

# 24/06/2021
- (1)recodage de la partie "analyse des métriques et modèle" en R. 
- métriques de SCUT-FBP enregistrées en csv et pas en json: erreur de ma part ou bug de la fonction a cause de la grande quantité de données ?

# 25/06/2021

- avancées (fastidieuses) sur la conversion du programme python en R

# 28/06/2021

- avancées sur la conversion du programme python en R
- test de quelques modèles prenant parfois en compte les effets d'interaction et les effets quadratiques --> résultas intéressants

# 29/06/2021

- abscisse du point d'inflexion: aucune corrélation 
- discussion sur la thèse 
- point sur l'organisation sur la suite du stage


# 30/06/2021

- lecture de papiers

# 01/07/2021

- la corrélation avec le coeff directeur du point d'inflexion de la sigmoide est assez négative (- 0,32) avec Gini/CFD/VGGFaces mais pas du tout avec L0/CFD /VGGfaces(0,002). Avec  Gini/CFD/Imagenet, la corréltion est inversée (0,28) --> comment interpréter ça ? 

- matrices de corrélation par paires de couches sur CFD et SCUT et MART/vgg sur R

- Création d'un document récapitulant TOUS les résultats sur le drive

# 02/07/2021

- conférence Hanna Kokko 11h15 https://umontpellier-fr.zoom.us/j/85730904192?pwd=MFNkR0ErdkZIM2V4SW1rQ0xIZ28xZz09
- conférence esthétique musique 

# 05/07/2021

- avancées sur le programme R
- talk sur les crocodiles
- réunion avec sonia pour lui présenter les dernières avancées

# 06/07/2021

- avancées sur le programme R:
	+ import de la métrique de la pente maximale par image  a partir d'un JSON généré par Python (super lent--> faisable? )
	+ flexibilisation du programme, non sensible à la variabilité du nom des couches désormais

- commande du pc pour la thèse: 14 pouces, I7, Linux
- réparation de vggplaces: attention: bug pour le model de smalltest car vmax < 1 mais on s'en fout un peut 
- lancement des calculs de métriques avec vggplaces
- le truc sur les points d'inflexion n'était pas adapté

# 07/07/2021

- je bloque un peu sur gompertz, il semblerait qu'il faille définir des paramètres a priori --> on laisse tomber pour le moment
- calcul de la complexité ( = moyenne des couches) --> assez long temps de calcul, a peu près comme gini
- on prend que la couche 3 du block 4 pour la complexité, comme dans la publi? 

# 08/07/2021
# 09/07/2021

# 12/07/2021

- modèle sur les couches avec toutes les combinaisons de paramètres, stockage du R2 dans toutes 


# 13/07/2021

- réunion avec Julien et Sonia : cf compte rendu dans /meeting

# 15/07/2021

- test du calcul d'entropie: temps de calcul irréaliste (nécéssité d'une machine beaucoup plus puissante, ou alors de parraléliser mais je ne sais pas faire)
- z transformation (centrage/reduction) avant le modèle, pour comparer les tailles d'effets

# 16/07/2021

# 19/07/2021
# 20/07/2021
# 21/07/2021
# 22/07/2021
# 23/07/2021

# 26/07/2021
# 27/07/2021
# 28/07/2021
# 29/07/2021
# 30/07/2021

# 02/08/2021
# 03/08/2021
# 04/08/2021
# 05/08/2021
# 06/08/2021

# 09/08/2021
# 10/08/2021
# 11/08/2021
# 12/08/2021
# 13/08/2021

# 16/08/2021
# 17/08/2021
# 18/08/2021
# 19/08/2021
# 20/08/2021

# 23/08/2021
# 24/08/2021
# 25/08/2021
# 26/08/2021
# 27/08/2021

# 30/08/2021
# 31/08/2021
# 01/09/2021
# 02/09/2021
# 03/09/2021

# 06/09/2021
# 07/09/2021
# 08/09/2021
# 09/09/2021
# 10/09/2021

# 13/09/2021
# 14/09/2021
# 15/09/2021
# 16/09/2021
# 17/09/2021

# TODO

## coder:	

### métriques:

- refaire les plots des R2 par couche pour gini avc toutes les combinaisons
- matrice de corrélations croisées entre les BDD (éclairicir ce point)

### model:		

- regarder quelles variables restant après le fit du modèle (si ce sont toujours les mêmes indépendament des paramètres
- birckhoff2: tester en rajoutant les effets simples, normalement c'est la même chose que si on fait l'interaction avec la complexité
- regarder le sens de l'effet de la complexité (R positif ou négatif)
- revérifier les effets quadratiques, sur tous les paramètres (avec juste gini flatten sur le modèle avec l'interaction)

## faire:
	- écrire les fonctions du programme sous forme de formules mathématiques
	- séléction du modèle: matrices de corrélation sur toutes les variables, quand deux sont a plus de 0,7, on concerve celle qui a le meilleur R2

## autres:
	- papier fondateur fluence: Ralf Reber & Piotr Winkelman


## Partie 2 du stage:  

	- SCUT donne les notes par évaluateur, regarder si la variance des notes d'une image est corrélée au pic de sparseness dans les couches
	Hypothèse: image consensuelle = variance faible = sparseness élevée dès les premières couches

	- appliquer les conseils de Rufin, cf livre de géron, cf réunion du 08/04


# THESE

	- Damien Farine est un chercheur qui travaille sur les réseaux sociaux animaux: https://sites.google.com/site/drfarine/publications 
	- interview d'eveillard sur les réseaux sociaux de plancton


































