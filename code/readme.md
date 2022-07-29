# code's readme




## main_multiGauss.py      Mode d'emploi:

prototype permettant, a partir des activation extraite, de générer les LLH

##executions avec parametre:

`main_multiGauss.py mesoLR CFD_ALL average modele_Fairface`

###Argument 1: [utile uniquement pour le mesoLR],

    "`mesoLR`" indique que le noeud standard est utilisé
    "`mesoLR-3T`" indique que le noeud de 3To est utilisé
 
----
    
###Argument 2: "BDD"

    les données doivent etre dans le repertoire results/

    macro: "`CFD_ALL`":
	
    calcule les llh pour tout les sous ensemble de CFD (ceux ci doivent tous etre stoqué dans result/ ) 
---- 

###Argument 3 "Method" 

    indique la méthode a utilisé : `pca`, `average`, `featureMap` etc.

----

###Argument 4 "model"
    indique le nom de la bdd a utilisé comme modèle de reference

    macro: "`modele_Fairface`":
	
    associe le modèle correspondant a chaque sous ensemble de CFD
	

----
