# Scripts MesoLR

## Execution scripts
`Sbatch file arg1 arg2`

arg1 = BDD
arg2 = methode (average, pca etc.)

## connexion Compte Meso@LR 
Cluster : muse-login.hpc-lr.univ-montp2.fr
Login : tieos
Mot de passe : a demander auprès de Sonia
Account (nom du groupe) : swp-gpu
Partition(s) : muse-visu
commande connexion ssh:
```ssh tieos@muse-login.hpc-lr.univ-montp2.fr
```

## Arborescence
Racine:
`/home`

Emplacement des scripts d'execution afin de faire tourner les programmes:
`/home/scratch`

##### Noeud Visu (128 Ga ram)

Emplacement des programmes et données (en lecture et écriture)
`/home/work_swp-gpu`

##### Noeud Visu (3 Terra ram)

Emplacement programmes et données ( ACCES EN LECTURE UNIQUEMENT )
`/home/work_cefe_swp-smp`

Emplacement des données (ACCES EN ECRITURE UNIQUEMENT)
```/lustre/[USER]/work_cefe_swp-smp/```
donc, si tieos est user:
```/lustre/tieos/work_cefe_swp-smp/```
## commende mesoLR

annule l'execution du scrips
`scancel id`


affiche la liste des scrips en cours d'execution
`squeue`


