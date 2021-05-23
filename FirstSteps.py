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
sim_options = {'name': 'pearson_baseline', 'user_based': False}   # compute  similarities between items
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)
