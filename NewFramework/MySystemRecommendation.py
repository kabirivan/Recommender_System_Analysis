#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:47:05 2021

@author: xavieraguas
"""

from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator


import random
import numpy as np


def LoadMovieLensData():
    ml = MovieLens()
    print("Carga de ratings de las peliculas...")
    data = ml.loadMovieLensLatestSmall()
    print("\nCalculo de las peliculas que tienen mas calificaciones --> Novedad...")
    rankings = ml.getPopularityRanks()
    return (data, rankings)


np.random.seed(0)
random.seed(0)


# Datos generales para el calculo de la eficiencia de los algoritmos de recomendacion
(evaluationData, rankings) = LoadMovieLensData()


# Evaluador de algoritmos
evaluator = Evaluator(evaluationData, rankings)