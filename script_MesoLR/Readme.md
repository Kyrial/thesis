#MesoLR

## Execution scripts


## connexion Compte Meso@LR 

Cluster : muse-login.hpc-lr.univ-montp2.fr

Login : tieos

Mot de passe : a demander auprès de Sonia

Account (nom du groupe) : swp-gpu

Partition(s) : muse-visu

commande connexion ssh:

```ssh tieos@muse-login.hpc-lr.univ-montp2.fr```

## Arborescence
Racine:

`/home`

----

Emplacement des scripts d'execution et des output afin de faire tourner les programmes:

`/home/scratch`


#### Noeud Visu (128 Ga ram)

Emplacement des programmes et données (en lecture et écriture)

`/home/work_swp-gpu`

#### Noeud Visu (3 Terra ram)

Emplacement programmes et données ( ACCES EN LECTURE UNIQUEMENT )

`/home/work_cefe_swp-smp`

----

Emplacement des données (ACCES EN ECRITURE UNIQUEMENT)

```/lustre/[USER]/work_cefe_swp-smp/```

donc, si tieos est user:

```/lustre/tieos/work_cefe_swp-smp/```
## commande mesoLR

Execution de scripts

`Sbatch file arg1 arg2`

arg1 = BDD

arg2 = methode (average, pca etc.)

----

annule l'execution du scrips

`scancel id`

----

affiche la liste des scrips en cours d'execution

`squeue`

----

affiche le nombre d'heure utilisé

`mon_nbre_heures`

ensuite spécifier le groupe :

`cefe_swp-smp` (noeud 3To) ou `swp-gpu` (noeud de 126 Go) 

----



# Transfert de fichier par SFTP

permet de transferer entre la machine local et le MesoLR tout type de dossiers et fichiers

## commandes

commande connexion ssh:

```sftp tieos@muse-login.hpc-lr.univ-montp2.fr```

----

afficher le chemin du repertoire courant du mesoLR

`pwd`

----

afficher le chemin du repertoire courant de la machine local

`lpwd`

### Envoie de données au MesoLR

##### envoie d'un fichier

`put path_file`

path_file = chemin relatif par rapport au répertoire courant de la machine local

##### envoie d'un dossier:

le dossier doit etre créer préalablement,

création d'un dossier:

`mkdir name`

name = nom du dossier

#####

envoie d'une arborescence de fichier 

`put -r path_file`

ajout de -r pour forcer la récurcivité dans l'arborescence.

#### recuperer des données du MesoLR 

`get path_file`

meme principe que pour l'envoie, spécifier `-r` si arborescence de dossier, et créer le dossier s'il s'agit d'un dossier





