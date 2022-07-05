1) SPARSITE + COMPLEXITE:

	1)1) PAS DE CV, MODEL POUR CHAQUE COUCHE
	
	-r_model: 
	Pour chaque couche séparément, sans validation croisée, modèle multiple de la fluence + complexité + interaction + effet quadratique fluence + effet quadratique complexité
	
	
	1)2) LOOCV (glm, ridge et lasso)
	
	-r_model_LOOCV_glm:
	C'est la qu'il y a les plots des estimates !! (mais attention c'est bien des modèles séparés par couche)
	Sur chaque couche, LOOCV de la complexité + gini + interaction entre les deux + effets quadratiques des deux
	
	-r_model_LOOCV_glmsans_effet_quadra:
	idem sans les effets quadratiques

	-r_model_LOOCV_glmsans_effet_quadra_ni_complexite:
	idem sans les effets quadratiques ET la complextié
	
	- r_model_LOOCV_ridge:
	Ridge en LOOCV (ou lasso) sur toutes les couches, sparsité + complexité
	
	
	1)3) 10-FOLD CV, ridge et lasso
	
	- r_model_10f-cv_ridgelasso_bylayer_spar_comp
	Modèle sparsité+complexité par couche 
	
	- r_model_10-fold_ridge_lasso:
	modèle complet, 10-fold, ridge et lasso et elasticnet, tailles d'effets
	
	
	1)4) CROISEMENTS DE BDD pour toutes les couches en même temps
	
	- r_model_ridge_2db:
	r2 entre un modèle entrainé sur une db et testé sur une autre (r2 entre la bveauté vraie et la beauté prédite), y = pour toutes les couches, beauté  complexité

	- r_model_ridge_3db:
	idem entre deux db concaténes en train pour prédire une autre

	- r_model_ridge_4db:
	idem avec deux db concaténées en train et 2 db concaténées en test
		
	

2) ACP:

	- r_model_10fold_ridge_ACP:
	modèles 10-fold sur les ACP sur les composantes de chaque couche pour 80% de variance, output: BIC AIC R2 pour toutes les couches
	
	-r_model_10fold_ridge_ACP_SCUT:
	idem mais sans boucle, spécialement pour scut car sinon ça marche pas
	
	- r_model_10fold_ridge_acp_several_variance_treshold
	modèles 10-fold sur les ACP sur les composantes de chaque couche pour des seuils de 20 à 80% de variance expliquée, output: graphes d'AIC/BIC min, R2max (parmis chaque couche) pour chaque % de 	variance

	- r_model_LOOCV_ridge_ACP:
	ridge sur les   ACP pour chaque couche, en LOOCV donc infaisable
	
	- test_curve_variance_pca:
	Plot de la variance cumulée des composnates de l'acp pour voir si la forme est cohérente (log), a permis de débuguer le pb d'arrondi, en partie

3) AUTRE:

	r_model_internship:
	models divers, complexité, points d'inflexion, birckoff, acp sur gini des dernières couches de conv, etc (des trucs de la fin du stage) etc




















