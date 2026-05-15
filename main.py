"""
Модуль расчета одномерных стационарных температурных полей в телах простой
геометрической формы (пластина, цилиндр) для случаев q_v = 0 q_v != 0

@Автор: <Барышев Савва>
Назначение: Расчет одномерных стационарных температурных полей в телах простой
геометрической формы
Версия: 1.0

Module for calculating one-dimensional steady heat conduction
problems for plates and cylinders for q_v = 0 q_v != 0

@author: <Savva Baryshev>
Purpose: Calculation of one-dimensional steady heat conduction problems for  simple geometric bodies
Version: 1.0
"""


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate
from math import *


class Init_Parameters:
    # Класс начальных параметров системы

    def __init__(self):
        # тепловые параметры системы:
        self.q_v0 = 0*10e-6 # референсное тепловыделение, Вт*м²
        self.q_v = 50*10e-6 # тепловыделение твэла, Вт*м²
        self.t_c = 1800 # температура поверхности сердечника,  ̊C
        self.t_m = 2800  # температура плавления твэла,  ̊C
        self.t_c = 1800  # температура наружной поверхности оболочки,  ̊C
        self.lambda_f = 5.0 # коэффициент теплопроводности твэла, Вт/(м*град)
        self.lambda_s = 40.0 # коэффициент теплопроводности оболочки, Вт/(м*град)
        self.lambda_a = 0.052 # коэффициент теплопроводности воздуха, Вт/(м*град)
        self.lambda_he = 0.28 # коэффициент теплопроводности He, Вт/(м*град)

        # геометрические параметры системы:
        self.d = 15*10e-3 # толщина пластины () или диаметр сплошного цилиндра
        self.c = 0.05*10e-3 #


