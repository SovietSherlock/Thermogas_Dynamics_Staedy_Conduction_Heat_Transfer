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
import sympy as sp
import matplotlib.pyplot as plt
import math
from scipy import integrate


class Init_Parameters:
    # Класс начальных параметров системы

    def __init__(self):
        # тепловые параметры системы:
        self.q_v0 = 0e-6 # мощность тепловыделения оболочки, Вт*м²
        self.q_v = 50e-6 # мощность тепловыделения твэла, Вт*м²
        self.t_c = 1800 # температура поверхности сердечника, ̊C
        self.t_m = 2800  # температура плавления твэла, ̊C
        self.t_02_r = 1200  # референсная температура наружной поверхности оболочки, ̊C
        self.lambda_f = 5.0 # коэффициент теплопроводности твэла, Вт/(м*град)
        self.lambda_s = 40.0 # коэффициент теплопроводности оболочки, Вт/(м*град)
        self.lambda_a = 0.052 # коэффициент теплопроводности воздуха, Вт/(м*град)
        self.lambda_he = 0.28 # коэффициент теплопроводности He, Вт/(м*град)

        # геометрические параметры системы:
        self.delta = 1e-3 # толщина защитной оболочки, м
        self.d = 15e-3 # толщина пластины (2*Delta) или диаметр сплошного цилиндра (2*r_c), м
        self.c = 0.05e-3 # толщина зазора, м

class  Math_Model(Init_Parameters):
    # Класс математических моделей для определения термических параметров системы, при следующих условиях установки оболочки относительно твэла:
    # а) Идеальный контакт между твэлом и оболочкой
    # б) Наличие воздушного зазора между твэлом и оболочкой, толщиной 0,05 мм
    # в) Наличие зазора между твэлом и оболочкой, толщиной 0,05 мм, заполненного He

    def __init__(self):
        super().__init__()
        self.t_01_p_a = None
        self.t_01_c_a = None
        self.t_01_p_h = None
        self.t_01_c_h = None

    def ODE_fuel_rod_plane(self):
        # функция описания математической модели для тепловыделяющей пластины:
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x) + self.q_v/self.lambda_f, 0)
        ics = {
            t.diff(x).subs(x, 0): 0,
            t.subs(x, self.d/2): self.t_c
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_fuel_rod_cylinder(self):
        # функция описания математической модели для тепловыделяющего цилиндра:
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r) + self.q_v/self.lambda_f, 0)
        ics = {
            t.diff(r).subs(r, 0): 0,
            t.subs(r, self.d/2): self.t_c
        }
        return sp.dsolve(eq, t, ics=ics)


    def ODE_shell_plane(self,):
        # функция описания математической модели для оболочки пластины с условием установки а):
        t_02 = self.t_c - (self.q_v*self.delta**2)/self.lambda_f # температура на внешней границе оболочки, ̊C
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, self.d/2): self.t_c,
            t.subs(x, self.d/2 + self.delta): t_02
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_shell_cylinder(self):
        # функция описания математической модели для оболочки цилиндра с условием установки а):
        numerator = (self.q_v/4)*((self.d + 2*self.delta)**2 - self.d**2)
        denominator = (2/math.log((self.d + 2*self.delta)/self.d))*self.lambda_f
        t_02 = self.t_c - numerator/denominator # температура на внешней границе оболочки, ̊C
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, self.d/2): self.t_c,
            t.subs(r, self.d/2 + self.delta): t_02
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_clearance_plane_air(self,):
        # функция описания математической модели для зазора между оболочкой и пластиной с условием установки б):
        self.t_01_p_a = self.t_c - (self.q_v*self.c**2)/self.lambda_a # температура на внешней границе оболочки, ̊C
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, self.d/2): self.t_c,
            t.subs(x, self.d/2 + self.c): self.t_01_p_a
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_clearance_cylinder_air(self):
        # функция описания математической модели для зазора между оболочкой и цилиндром с условием установки б):
        numerator = (self.q_v/4)*((self.d + 2*self.c)**2 - self.d**2)
        denominator = (2/math.log((self.d + 2*self.c)/self.d))*self.lambda_a
        self.t_01_c_a = self.t_c - numerator/denominator # температура на внешней границе оболочки, ̊C
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, self.d/2): self.t_c,
            t.subs(r, self.d/2 + self.c): self.t_01_c_a
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_clearance_plane_helium(self,):
        # функция описания математической модели для зазора между оболочкой и пластиной с условием установки в):
        self.t_01_p_h = self.t_c - (self.q_v * self.c ** 2) / self.lambda_he  # температура на внешней границе оболочки, ̊C
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x, x), 0)
        ics = {
            t.subs(x, self.d / 2): self.t_c,
            t.subs(x, self.d / 2 + self.c): self.t_01_p_h
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_clearance_cylinder_helium(self):
        # функция описания математической модели для зазора между оболочкой и цилиндром с условием установки в):
        numerator = (self.q_v / 4) * ((self.d + 2 * self.c) ** 2 - self.d ** 2)
        denominator = (2 / math.log((self.d + 2 * self.c) / self.d)) * self.lambda_he
        self.t_01_c_h = self.t_c - numerator / denominator  # температура на внешней границе оболочки, ̊C
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r, r) + (1 / r) * t.diff(r), 0)
        ics = {
            t.subs(r, self.d / 2): self.t_c,
            t.subs(r, self.d / 2 + self.c): self.t_01_c_h
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_shell_plane_gas(self,):
        # функция описания математической модели для оболочки пластины с условием установки с зазором:
        # присвоение внутренней переменной валидного значения
        t_01 = self.t_01_p_a if self.t_01_p_a is not None else self.t_01_p_h
        if t_01 is None: # проверка на валидность
            raise ValueError("Пустое значение температуры внутренней стенки оболочки. Проверьте вычисления.")
        t_02 = t_01 - (self.q_v * self.delta ** 2) / self.lambda_s  # температура на внешней границе оболочки, ̊C
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, self.d/2 + self.c): t_01,
            t.subs(x, self.d/2 + self.c + self.delta): t_02
        }
        return sp.dsolve(eq, t, ics=ics)

    def ODE_shell_cylinder_gas(self):
        # функция описания математической модели для оболочки цилиндра с условием установки с зазором:
        t_01 = self.t_01_c_a if self.t_01_c_a is not None else self.t_01_c_h
        if t_01 is None: # проверка на валидность
            raise ValueError("Пустое значение температуры внутренней стенки оболочки. Проверьте вычисления.")
        numerator = (self.q_v / 4) * ((self.d + 2 * (self.c + self.delta)) ** 2 - (self.d + 2 * self.c) ** 2)
        denominator = (2 / math.log((self.d + 2 * (self.c + self.delta)) / (self.d + 2 * self.c))) * self.lambda_s
        t_02 = t_01 - numerator / denominator  # температура на внешней границе оболочки, ̊C
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r, r) + (1 / r) * t.diff(r), 0)
        ics = {
            t.subs(r, self.d/2 + self.c): t_01,
            t.subs(r, self.d/2 + self.c + self.delta): t_02
        }
        return sp.dsolve(eq, t, ics=ics)

