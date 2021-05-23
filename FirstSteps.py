#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:54:07 2021

@author: xavieraguas
"""

from MovieLens import MovieLens
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics


ml = MovieLens()

#Carga dataset
data = ml.loadMovieLensLatestSmall()

#Peliculas mas votadas -> Sirve para calcular Novedad
rankings = ml.getPopularityRanks()


# Similitud entre items -> Sirve para calcular Diversidad
fullTrainSet = data.build_full_trainset()
bigTestSet1 = fullTrainSet.build_anti_testset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}   # compute  similarities between items
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)


# Construccion del sistema de recomendacion
# 75% Training 25% testing
trainSet, testSet = train_test_split(data, test_size=.25, random_state=1)
algo = SVD(random_state=10)
algo.fit(trainSet)


#Prediciones
predictions = algo.test(testSet)



# Evaluation
print("RMSE: ", RecommenderMetrics.RMSE(predictions))
print("MAE: ", RecommenderMetrics.MAE(predictions))

#Reserva una calificación por usuario para realizar pruebas
LOOCV = LeaveOneOut(n_splits=1, random_state=1)

rank = 0
for trainSet, testSet in LOOCV.split(data):
    
    print(len(testSet))
    # Entrenar modelo sin ratings excluidos
    algo.fit(trainSet)
    
    # Prediccion para ratings excluidos
    leftOutPredictions = algo.test(testSet)
    
    # Construccion del dataset que no tiene calificaciones realizadas por los usuarios a las peliculas
    bigTestSet = trainSet.build_anti_testset()
    allPredictions = algo.test(bigTestSet)
    
    
    # Calcula el top 10 para cada usuario
    topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n=10)
    
    # Revisar con que frecuencia se recomienda una pelicula que el usuario califico
    print("\nHit Rate: ", RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions))
    rank += 1
    print(rank)
    
    
    
    
    
    