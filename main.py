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
import pandas as pd


class Init_Parameters:
    # Класс начальных параметров системы

    def __init__(self):
        # тепловые параметры системы:
        self.q_v0 = 0e6 # мощность тепловыделения оболочки, Вт*м²
        self.q_v = 50e6 # мощность тепловыделения твэла, Вт*м²
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
        eq = sp.Eq(t.diff(x,x) + sp.nsimplify(self.q_v/self.lambda_f), 0)
        ics = {
            t.diff(x).subs(x, 0): 0,
            t.subs(x, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c)
        }
        return sp.dsolve(eq, ics=ics)

    def ODE_fuel_rod_cylinder(self):
        # функция описания математической модели для тепловыделяющего цилиндра:
        r = sp.Symbol('r')
        R = self.d / 2
        t_expr = self.t_c + (self.q_v / (4 * self.lambda_f)) * (R ** 2 - r ** 2)
        return sp.Eq(sp.Function('t')(r), t_expr)

    def ODE_shell_plane(self):
        # функция описания математической модели для оболочки пластины с условием установки а):
        # Расчет теплового потока и перепада температуры в оболочке
        q = self.q_v * (self.d / 2)  # тепловой поток, Вт/м²
        t_02 = sp.nsimplify(self.t_c - q * self.delta / self.lambda_s)
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c),
            t.subs(x, sp.nsimplify(self.d/2 + self.delta)): t_02
        }
        return sp.dsolve(eq, ics=ics)

    def ODE_shell_cylinder(self):
        # функция описания математической модели для оболочки цилиндра с условием установки а):
        numerator = sp.nsimplify(self.q_v*((self.d/2)**2)*math.log((self.d + 2*self.delta)/self.d))
        denominator = sp.nsimplify(2*self.lambda_s)
        t_02 = sp.nsimplify(self.t_c - numerator/denominator)
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c),
            t.subs(r, sp.nsimplify(self.d/2 + self.delta)): t_02
        }
        return sp.dsolve(eq, ics=ics)

    def ODE_clearance_plane_air(self):
        # функция описания математической модели для зазора между оболочкой и пластиной с условием установки б):
        numerator = sp.nsimplify(self.q_v * self.d * self.c)
        denominator = sp.nsimplify(2 * self.lambda_a)
        self.t_01_p_a = sp.nsimplify(self.t_c - numerator/denominator)
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c),
            t.subs(x, sp.nsimplify(self.d/2 + self.c)): self.t_01_p_a
        }
        return sp.dsolve(eq, ics=ics), self.t_01_p_a

    def ODE_clearance_cylinder_air(self):
        # функция описания математической модели для зазора между оболочкой и цилиндром с условием установки б):
        numerator = sp.nsimplify(self.q_v*((self.d/2)**2)*math.log((self.d + 2*self.c)/self.d))
        denominator = sp.nsimplify(2*self.lambda_a)
        self.t_01_c_a = sp.nsimplify(self.t_c - numerator/denominator)
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c),
            t.subs(r, sp.nsimplify(self.d/2 + self.c)): self.t_01_c_a
        }
        return sp.dsolve(eq, ics=ics), self.t_01_c_a

    def ODE_clearance_plane_helium(self):
        # функция описания математической модели для зазора между оболочкой и пластиной с условием установки в):
        numerator = sp.nsimplify(self.q_v * self.d * self.c)
        denominator = sp.nsimplify(2 * self.lambda_he)
        self.t_01_p_h = sp.nsimplify(self.t_c - numerator/denominator)
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c),
            t.subs(x, sp.nsimplify(self.d/2 + self.c)): self.t_01_p_h
        }
        return sp.dsolve(eq, ics=ics), self.t_01_p_h

    def ODE_clearance_cylinder_helium(self):
        # функция описания математической модели для зазора между оболочкой и цилиндром с условием установки в):
        numerator = sp.nsimplify(self.q_v * ((self.d / 2) ** 2) * math.log((self.d + 2 * self.c) / self.d))
        denominator = sp.nsimplify(2 * self.lambda_he)
        self.t_01_c_h = sp.nsimplify(self.t_c - numerator/denominator)
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, sp.nsimplify(self.d/2)): sp.nsimplify(self.t_c),
            t.subs(r, sp.nsimplify(self.d/2 + self.c)): self.t_01_c_h
        }
        return sp.dsolve(eq, ics=ics), self.t_01_c_h

    def ODE_shell_plane_air(self):
        # функция описания математической модели для оболочки пластины с условием установки б):
        t_01 = sp.nsimplify(self.t_01_p_a)
        numerator = sp.nsimplify(self.q_v * self.d * self.delta)
        denominator = sp.nsimplify(2 * self.lambda_s)
        t_02 = sp.nsimplify(t_01 - numerator/denominator)
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, sp.nsimplify(self.d/2 + self.c)): t_01,
            t.subs(x, sp.nsimplify(self.d/2 + self.c + self.delta)): t_02
        }
        return sp.dsolve(eq, ics=ics)

    def ODE_shell_plane_he(self):
        # функция описания математической модели для оболочки пластины с условием установки в):
        t_01 = sp.nsimplify(self.t_01_p_h)
        numerator = sp.nsimplify(self.q_v * self.d * self.delta)
        denominator = sp.nsimplify(2 * self.lambda_s)
        t_02 = sp.nsimplify(t_01 - numerator/denominator)
        x = sp.Symbol('x')
        t = sp.Function('t')(x)
        eq = sp.Eq(t.diff(x,x), 0)
        ics = {
            t.subs(x, sp.nsimplify(self.d/2 + self.c)): t_01,
            t.subs(x, sp.nsimplify(self.d/2 + self.c + self.delta)): t_02
        }
        return sp.dsolve(eq, ics=ics)


    def ODE_shell_cylinder_air(self):
        # функция описания математической модели для оболочки цилиндра с условием установки б):
        t_01 = sp.nsimplify(self.t_01_c_a)
        numerator = sp.nsimplify(self.q_v * ((self.d / 2) ** 2) * math.log((self.d + 2*(self.c + self.delta)) / self.d))
        denominator = sp.nsimplify(2 * self.lambda_s)
        t_02 =  sp.nsimplify(t_01 - numerator/denominator)
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, sp.nsimplify(self.d/2 + self.c)): t_01,
            t.subs(r, sp.nsimplify(self.d/2 + self.c + self.delta)): t_02
        }
        return sp.dsolve(eq, ics=ics)

    def ODE_shell_cylinder_he(self):
        # функция описания математической модели для оболочки цилиндра с условием установки б):
        t_01 = sp.nsimplify(self.t_01_c_h)
        numerator = sp.nsimplify(self.q_v * ((self.d / 2) ** 2) * math.log((self.d + 2*(self.c + self.delta)) / self.d))
        denominator = sp.nsimplify(2 * self.lambda_s)
        t_02 =  sp.nsimplify(t_01 - numerator/denominator)
        r = sp.Symbol('r')
        t = sp.Function('t')(r)
        eq = sp.Eq(t.diff(r,r) + (1/r)*t.diff(r), 0)
        ics = {
            t.subs(r, sp.nsimplify(self.d/2 + self.c)): t_01,
            t.subs(r, sp.nsimplify(self.d/2 + self.c + self.delta)): t_02
        }
        return sp.dsolve(eq, ics=ics)

    def volumetric_heat_release_plane(self):
        # функция описания мат модели объемной плотности тепловыделения для пластинчатой модели системы с условием установки а):
        numerator = 8*self.lambda_f*(self.t_m - self.t_02_r)
        denominator = self.d**2
        q_e = numerator/denominator
        t_c_r = (q_e*self.d**2)/4*self.lambda_f + self.t_m
        return q_e, t_c_r

    def volumetric_heat_release_cylinder(self):
        # функция описания мат модели объемной плотности тепловыделения для цилиндрической модели системы с условием установки а):
        numerator = 16*self.lambda_f*(self.t_m - self.t_02_r)
        denominator = self.d**2
        q_e = numerator / denominator
        numerator_1 = (q_e*self.d**2*sp.log((self.d + 2*self.delta) / self.d))
        denominator_1 = 2*self.lambda_s
        t_c_r = self.t_02_r - numerator_1 / denominator_1
        return q_e, t_c_r

    def volumetric_heat_release_plane_air(self):
        # функция описания мат модели объемной плотности тепловыделения для пластинчатой модели системы с условием установки б):
        numerator = self.t_m - self.t_02_r
        denominator = (self.d**2/(8*self.lambda_f)) + (self.d*self.c/(2*self.lambda_a)) + (self.d*self.delta/(2*self.lambda_s))
        q_e = numerator / denominator
        t_c_r = (q_e * self.d ** 2) / 4 * self.lambda_f + self.t_m
        t_c0_r = (q_e * self.d * self.c) / 4 * self.lambda_a + t_c_r
        return q_e, t_c_r, t_c0_r

    def volumetric_heat_release_cylinder_air(self):
        # функция описания мат модели объемной плотности тепловыделения для цилиндрической модели системы с условием установки б):
        numerator = self.t_m - self.t_02_r
        denominator = (self.d ** 2 / (16 * self.lambda_f)
                       + (self.d ** 2 / (8 * self.lambda_a)) * math.log(1 + 2 * self.c / self.d)
                       + (self.d ** 2 / (8 * self.lambda_s)) * math.log(1 + 2 * self.delta / (self.d + 2 * self.c)))
        q_e = numerator / denominator
        return q_e

    def volumetric_heat_release_plane_helium(self):
        # функция описания мат модели объемной плотности тепловыделения для пластинчатой модели системы с условием установки в):
        numerator = self.t_m - self.t_02_r
        denominator = (self.d**2/(8*self.lambda_f)) + (self.d*self.c/(2*self.lambda_he)) + (self.d*self.delta/(2*self.lambda_s))
        q_e = numerator / denominator
        t_c_r = (q_e * self.d ** 2) / 4 * self.lambda_f + self.t_m
        t_c0_r = (q_e * self.d * self.c) / 4 * self.lambda_he + t_c_r
        return q_e, t_c_r, t_c0_r


    def volumetric_heat_release_cylinder_helium(self):
        # функция описания мат модели объемной плотности тепловыделения для цилиндрической модели системы с условием установки в):
        numerator = self.t_m - self.t_02_r
        denominator = (self.d ** 2 / (16 * self.lambda_f)
                       + (self.d ** 2 / (8 * self.lambda_he)) * math.log(1 + 2 * self.c / self.d)
                       + (self.d ** 2 / (8 * self.lambda_s)) * math.log(1 + 2 * self.delta / (self.d + 2 * self.c)))
        q_e = numerator / denominator
        return q_e

class Simulation(Math_Model):
    # Класс аналитического решения дифференциальных уравнений при заданных начальных условиях

    def __init__(self):
        super().__init__()
        self.results = {}
        self.x = sp.Symbol('x')
        self.r = sp.Symbol('r')

    def solution_to_numpy(self, solution, coord_var, coord_range, num_points=100):
        # Преобразует sympy решение в numpy массивы
        if solution is None:
            raise ValueError("Решение не найдено")

        rhs = solution.rhs
        free_symbols = rhs.free_symbols

        if coord_var not in free_symbols:
            coord_array_m = np.linspace(coord_range[0], coord_range[1], num_points)
            temp_array = np.full(num_points, float(rhs))
            return coord_array_m, temp_array

        temp_func = sp.lambdify(coord_var, rhs, 'numpy')
        coord_array_m = np.linspace(coord_range[0], coord_range[1], num_points)
        temp_array = temp_func(coord_array_m)
        return coord_array_m, temp_array

    # ==================== ПЛАСТИНА - ИДЕАЛЬНЫЙ КОНТАКТ ====================

    def get_plane_ideal_contact_table(self, num_points=None):
        # Таблица для пластины с идеальным контактом (случай а)

        # Расчёт количества точек для шага 1/16 от d/2
        step_m = (self.d / 2) / 16  # шаг в метрах
        num_points_fuel = 17  # 0, 1/16, 2/16, ..., 16/16 = 17 точек

        fuel_solution = self.ODE_fuel_rod_plane()
        shell_solution = self.ODE_shell_plane()

        x_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.x, (0, self.d / 2), num_points_fuel
        )

        # Для оболочки также используется 17 точек, для соответствия шага
        num_points_shell = 17
        x_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.x, (self.d / 2, self.d / 2 + self.delta), num_points_shell
        )

        x_total_m = np.concatenate([x_fuel_m, x_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_shell[1:]])
        x_total_mm = x_total_m * 1000

        df = pd.DataFrame({
            'x, мм': np.round(x_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })
        self.results['plane_ideal'] = df
        return df

    # ==================== ЦИЛИНДР - ИДЕАЛЬНЫЙ КОНТАКТ ====================

    def get_cylinder_ideal_contact_table(self, num_points=17):
        # Таблица для цилиндра с идеальным контактом (случай а)
        fuel_solution = self.ODE_fuel_rod_cylinder()
        shell_solution = self.ODE_shell_cylinder()

        num_points_layer = 17

        r_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.r, (0, self.d / 2), num_points_layer
        )
        r_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.r, (self.d / 2, self.d / 2 + self.delta), num_points_layer
        )

        r_total_m = np.concatenate([r_fuel_m, r_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_shell[1:]])
        r_total_mm = r_total_m * 1000

        df = pd.DataFrame({
            'r, мм': np.round(r_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })
        self.results['cylinder_ideal'] = df
        return df

    # ==================== ПЛАСТИНА - ВОЗДУШНЫЙ ЗАЗОР ====================

    def get_plane_air_gap_table(self, num_points=17):
        # Таблица для пластины с воздушным зазором (случай б)
        fuel_solution = self.ODE_fuel_rod_plane()
        clearance_solution, t_01 = self.ODE_clearance_plane_air()
        shell_solution = self.ODE_shell_plane_air()

        # 17 точек для каждого слоя (0, 1/16, ..., 16/16 от толщины слоя)
        num_points_layer = 17

        x_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.x, (0, self.d / 2), num_points_layer
        )
        x_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.x, (self.d / 2, self.d / 2 + self.c), num_points_layer
        )
        x_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.x, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points_layer
        )

        x_total_m = np.concatenate([x_fuel_m, x_clearance_m[1:], x_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_clearance[1:], t_shell[1:]])
        x_total_mm = x_total_m * 1000

        df = pd.DataFrame({
            'x, мм': np.round(x_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })
        self.results['plane_air'] = df
        return df

    # ==================== ПЛАСТИНА - ГЕЛИЕВЫЙ ЗАЗОР ====================

    def get_plane_helium_gap_table(self, num_points=17):
        # Таблица для пластины с гелиевым зазором (случай в)
        fuel_solution = self.ODE_fuel_rod_plane()
        clearance_solution, t_01 = self.ODE_clearance_plane_helium()
        shell_solution = self.ODE_shell_plane_he()

        num_points_layer = 17

        x_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.x, (0, self.d / 2), num_points_layer
        )
        x_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.x, (self.d / 2, self.d / 2 + self.c), num_points_layer
        )
        x_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.x, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points_layer
        )

        x_total_m = np.concatenate([x_fuel_m, x_clearance_m[1:], x_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_clearance[1:], t_shell[1:]])
        x_total_mm = x_total_m * 1000

        df = pd.DataFrame({
            'x, мм': np.round(x_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })
        self.results['plane_helium'] = df
        return df

    # ==================== ЦИЛИНДР - ВОЗДУШНЫЙ ЗАЗОР ====================

    def get_cylinder_air_gap_table(self, num_points=17):
        # Таблица для цилиндра с воздушным зазором (случай б)
        fuel_solution = self.ODE_fuel_rod_cylinder()
        clearance_solution, t_01 = self.ODE_clearance_cylinder_air()
        shell_solution = self.ODE_shell_cylinder_air()

        num_points_layer = 17

        r_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.r, (0, self.d / 2), num_points_layer
        )
        r_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.r, (self.d / 2, self.d / 2 + self.c), num_points_layer
        )
        r_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.r, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points_layer
        )

        r_total_m = np.concatenate([r_fuel_m, r_clearance_m[1:], r_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_clearance[1:], t_shell[1:]])
        r_total_mm = r_total_m * 1000

        df = pd.DataFrame({
            'r, мм': np.round(r_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })
        self.results['cylinder_air'] = df
        return df
    # ==================== ЦИЛИНДР - ГЕЛИЕВЫЙ ЗАЗОР ====================

    def get_cylinder_helium_gap_table(self, num_points=17):
        # Таблица для цилиндра с гелиевым зазором (случай в)
        fuel_solution = self.ODE_fuel_rod_cylinder()
        clearance_solution, t_01 = self.ODE_clearance_cylinder_helium()
        shell_solution = self.ODE_shell_cylinder_he()

        num_points_layer = 17

        r_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.r, (0, self.d / 2), num_points_layer
        )
        r_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.r, (self.d / 2, self.d / 2 + self.c), num_points_layer
        )
        r_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.r, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points_layer
        )

        r_total_m = np.concatenate([r_fuel_m, r_clearance_m[1:], r_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_clearance[1:], t_shell[1:]])
        r_total_mm = r_total_m * 1000

        df = pd.DataFrame({
            'r, мм': np.round(r_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })
        self.results['cylinder_helium'] = df
        return df

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    def print_table(self, df, title="Распределение температуры", num_rows=None):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        if num_rows:
            print(df.head(num_rows).to_string(index=False))
            print(f"\n... и еще {len(df) - num_rows} строк")
        else:
            print(df.to_string(index=False))
        print("\n" + "=" * 70)
        print(f"Всего точек: {len(df)}")
        print(f"Диапазон температур: {df['Температура, °C'].min():.1f} ... {df['Температура, °C'].max():.1f} °C")

    def save_table_to_csv(self, df, filename):
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Таблица сохранена в файл: {filename}")

    def run_all_tables(self, num_points=50, save_csv=True, save_dir="tables"):
        """Запуск всех расчетов и вывод таблиц"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("ЗАПУСК РАСЧЕТОВ ТЕМПЕРАТУРНЫХ ПОЛЕЙ")
        print("=" * 80)
        print(f"Таблицы сохраняются в папку: {save_dir}")

        df1 = self.get_plane_ideal_contact_table(num_points)
        self.print_table(df1, "ПЛАСТИНА - ИДЕАЛЬНЫЙ КОНТАКТ (случай а)")
        if save_csv:
            self.save_table_to_csv(df1, "plane_ideal_contact.csv", save_dir)

        df2 = self.get_cylinder_ideal_contact_table(num_points)
        self.print_table(df2, "ЦИЛИНДР - ИДЕАЛЬНЫЙ КОНТАКТ (случай а)")
        if save_csv:
            self.save_table_to_csv(df2, "cylinder_ideal_contact.csv", save_dir)

        df3 = self.get_plane_air_gap_table(num_points)
        self.print_table(df3, "ПЛАСТИНА - ВОЗДУШНЫЙ ЗАЗОР (случай б)")
        if save_csv:
            self.save_table_to_csv(df3, "plane_air_gap.csv", save_dir)

        df4 = self.get_cylinder_air_gap_table(num_points)
        self.print_table(df4, "ЦИЛИНДР - ВОЗДУШНЫЙ ЗАЗОР (случай б)")
        if save_csv:
            self.save_table_to_csv(df4, "cylinder_air_gap.csv", save_dir)

        df5 = self.get_plane_helium_gap_table(num_points)
        self.print_table(df5, "ПЛАСТИНА - ГЕЛИЕВЫЙ ЗАЗОР (случай в)")
        if save_csv:
            self.save_table_to_csv(df5, "plane_helium_gap.csv", save_dir)

        df6 = self.get_cylinder_helium_gap_table(num_points)
        self.print_table(df6, "ЦИЛИНДР - ГЕЛИЕВЫЙ ЗАЗОР (случай в)")
        if save_csv:
            self.save_table_to_csv(df6, "cylinder_helium_gap.csv", save_dir)

        print("\n" + "=" * 80)
        print("✅ ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ")
        print("=" * 80)
        return self.results

        # ==================== РАСЧЕТ КРИТИЧЕСКОЙ МОЩНОСТИ q_e ====================

    def print_qe_results(self):
        # Вывод результатов расчета критической мощности
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ РАСЧЕТА КРИТИЧЕСКОЙ МОЩНОСТИ q_e")
        print("=" * 70)
        print(f"Исходные условия:")
        print(f"  Максимальная допустимая температура сердечника t_m = {self.t_m} °C")
        print(f"  Температура наружной поверхности оболочки t_02_r = {self.t_02_r} °C")
        print(f"  Разница температур: Δt_max = {self.t_m - self.t_02_r} °C")

        print("\n" + "-" * 50)
        print("РЕЗУЛЬТАТЫ РАСЧЕТА q_e:")
        print("-" * 50)

        # Расчет q_e для всех случаев
        qe_values = {
            'Пластина, идеальный контакт': self.volumetric_heat_release_plane(),
            'Пластина, воздушный зазор': self.volumetric_heat_release_plane_air(),
            'Пластина, гелиевый зазор': self.volumetric_heat_release_plane_helium(),
            'Цилиндр, идеальный контакт': self.volumetric_heat_release_cylinder(),
            'Цилиндр, воздушный зазор': self.volumetric_heat_release_cylinder_air(),
            'Цилиндр, гелиевый зазор': self.volumetric_heat_release_cylinder_helium()
        }

        for name, qe in qe_values.items():
            print(f"\n{name}:")
            print(f"  q_e = {qe:.2e} Вт/м³ = {qe / 1e6:.2f} МВт/м³")

        # Сравнение с текущим q_v
        print("\n" + "-" * 50)
        print("СРАВНЕНИЕ С ТЕКУЩИМ РЕЖИМОМ:")
        print("-" * 50)
        print(f"Текущая мощность: q_v = {self.q_v / 1e6:.2f} МВт/м³")

        for name, qe in qe_values.items():
            margin = qe / self.q_v
            if margin > 1.2:
                status = "✅ БЕЗОПАСНО"
            elif margin > 1.0:
                status = "⚠️ ПРЕДЕЛЬНО"
            else:
                status = "❌ ОПАСНО"
            print(f"  {name:35}: запас = {margin:.3f} → {status}")

        return qe_values

    def get_qe_dataframe(self, save_csv=True, save_dir="tables"):
        """Возвращает DataFrame с результатами расчета q_e и сохраняет в CSV"""
        import os
        data = []

        cases = [
            ('Пластина', 'Идеальный контакт', self.volumetric_heat_release_plane()),
            ('Пластина', 'Воздушный зазор', self.volumetric_heat_release_plane_air()),
            ('Пластина', 'Гелиевый зазор', self.volumetric_heat_release_plane_helium()),
            ('Цилиндр', 'Идеальный контакт', self.volumetric_heat_release_cylinder()),
            ('Цилиндр', 'Воздушный зазор', self.volumetric_heat_release_cylinder_air()),
            ('Цилиндр', 'Гелиевый зазор', self.volumetric_heat_release_cylinder_helium())
        ]

        for geometry, contact, qe in cases:
            if qe is not None:
                data.append({
                    'Геометрия': geometry,
                    'Тип контакта': contact,
                    'q_e, Вт/м³': f"{qe:.2e}",
                    'q_e, МВт/м³': round(qe / 1e6, 2),
                    'Запас прочности': round(qe / self.q_v, 3)
                })

        df = pd.DataFrame(data)

        if save_csv:
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, "critical_power.csv")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n✓ Таблица критических мощностей сохранена в: {filepath}")

        return df

    def save_table_to_csv(self, df, filename, subdir="tables"):
        """Сохранение таблицы в CSV файл в указанную подпапку"""
        import os
        # Создаем папку для таблиц, если её нет
        os.makedirs(subdir, exist_ok=True)
        filepath = os.path.join(subdir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✓ Таблица сохранена в файл: {filepath}")


class Plotter(Simulation):
    # Класс вывода графиков распределения температуры

    def __init__(self):
        super().__init__()
        # Настройка параметров шрифта для matplotlib
        self.setup_plot_style()

        self.colors = {
            'ideal': 'steelblue',
            'air': 'firebrick',
            'helium': 'forestgreen'
        }

        self.labels = {
            'ideal': 'Идеальный контакт (а)',
            'air': 'Воздушный зазор (б)',
            'helium': 'Гелиевый зазор (в)'
        }

    def setup_plot_style(self):
        """Настройка стилей оформления графика"""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['axes.titleweight'] = 'normal'
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    def format_axes(self, ax, xlabel, ylabel, title, xlim=None, ylim=None):
        """Форматирование осей графика"""
        ax.set_xlabel(xlabel, fontsize=14, fontname='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=14, fontname='Times New Roman')
        ax.set_title(title, fontsize=14, fontname='Times New Roman')
        ax.tick_params(axis='both', labelsize=12, width=1.5)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def get_label(self, label_type):
        """Возвращает подпись с нужным шрифтом"""
        labels = {
            'x': 'x, мм',
            'r': 'r, мм',
            't': 't, °C',
            't_center': 't_ц, °C',
            't_surface': 't_пов, °C'
        }
        return labels.get(label_type, label_type)

    def plot_plane_distributions(self, save_path=None, show=True):
        """Построение графика распределения температуры в пластине"""
        if 'plane_ideal' not in self.results:
            print("Ошибка: сначала запустите расчеты (run_all_tables)")
            return None

        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        conditions = [
            ('plane_ideal', 'ideal'),
            ('plane_air', 'air'),
            ('plane_helium', 'helium')
        ]

        for result_key, condition in conditions:
            if result_key in self.results:
                df = self.results[result_key]
                ax.plot(df['x, мм'], df['Температура, °C'],
                        color=self.colors[condition],
                        linewidth=1,
                        label=self.labels[condition],
                        alpha=0.9)

        xlabel = self.get_label('x')
        ylabel = self.get_label('t')
        title = 'Распределение температуры в пластине'

        self.format_axes(ax, xlabel, ylabel, title)

        ax.annotate(f'$q_v = {self.q_v / 1e6:.0f}$ МВт/м$^3$',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, fontname='Times New Roman')

        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_cylinder_distributions(self, save_path=None, show=True):
        """Построение графика распределения температуры в цилиндре"""
        if 'cylinder_ideal' not in self.results:
            print("Ошибка: сначала запустите расчеты (run_all_tables)")
            return None

        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        conditions = [
            ('cylinder_ideal', 'ideal'),
            ('cylinder_air', 'air'),
            ('cylinder_helium', 'helium')
        ]

        for result_key, condition in conditions:
            if result_key in self.results:
                df = self.results[result_key]
                ax.plot(df['r, мм'], df['Температура, °C'],
                        color=self.colors[condition],
                        linewidth=1,
                        label=self.labels[condition],
                        alpha=0.9)

        xlabel = self.get_label('r')
        ylabel = self.get_label('t')
        title = 'Распределение температуры в цилиндре'

        self.format_axes(ax, xlabel, ylabel, title)

        ax.annotate(f'$q_v = {self.q_v / 1e6:.0f}$ МВт/м$^3$',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, fontname='Times New Roman')

        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_comparison_plane_cylinder(self, condition='ideal', save_path=None, show=True):
        """Сравнение распределения температуры в пластине и цилиндре"""
        condition_map = {
            'ideal': ('plane_ideal', 'cylinder_ideal'),
            'air': ('plane_air', 'cylinder_air'),
            'helium': ('plane_helium', 'cylinder_helium')
        }

        if condition not in condition_map:
            print(f"Ошибка: неизвестное условие '{condition}'")
            return None

        plane_key, cylinder_key = condition_map[condition]

        if plane_key not in self.results or cylinder_key not in self.results:
            print("Ошибка: сначала запустите расчеты (run_all_tables)")
            return None

        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        df_plane = self.results[plane_key]
        ax.plot(df_plane['x, мм'], df_plane['Температура, °C'],
                color=self.colors[condition],
                linewidth=1,
                label='Пластина',
                alpha=0.9,
                linestyle='-')

        df_cylinder = self.results[cylinder_key]
        ax.plot(df_cylinder['r, мм'], df_cylinder['Температура, °C'],
                color=self.colors[condition],
                linewidth=1,
                label='Цилиндр',
                alpha=0.9,
                linestyle='--')

        xlabel = 'x, r, мм'
        ylabel = self.get_label('t')
        title = f'Сравнение распределения температуры\nУсловие: {self.labels[condition]}'

        self.format_axes(ax, xlabel, ylabel, title)

        ax.annotate(f'$q_v = {self.q_v / 1e6:.0f}$ МВт/м$^3$',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, fontname='Times New Roman')

        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_all_conditions_comparison(self, save_path=None, show=True):
        """Построение двух графиков (пластина и цилиндр) рядом"""
        if 'plane_ideal' not in self.results:
            print("Ошибка: сначала запустите расчеты (run_all_tables)")
            return None

        self.setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        conditions_plane = [
            ('plane_ideal', 'ideal'),
            ('plane_air', 'air'),
            ('plane_helium', 'helium')
        ]

        for result_key, condition in conditions_plane:
            if result_key in self.results:
                df = self.results[result_key]
                ax1.plot(df['x, мм'], df['Температура, °C'],
                         color=self.colors[condition],
                         linewidth=1,
                         label=self.labels[condition],
                         alpha=0.9)

        conditions_cylinder = [
            ('cylinder_ideal', 'ideal'),
            ('cylinder_air', 'air'),
            ('cylinder_helium', 'helium')
        ]

        for result_key, condition in conditions_cylinder:
            if result_key in self.results:
                df = self.results[result_key]
                ax2.plot(df['r, мм'], df['Температура, °C'],
                         color=self.colors[condition],
                         linewidth=1,
                         label=self.labels[condition],
                         alpha=0.9)

        ax1.set_xlabel('x, мм', fontsize=14, fontname='Times New Roman')
        ax1.set_ylabel('t, °C', fontsize=14, fontname='Times New Roman')
        ax1.set_title('Пластина', fontsize=14, fontname='Times New Roman')
        ax1.tick_params(axis='both', labelsize=12)
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax2.set_xlabel('r, мм', fontsize=14, fontname='Times New Roman')
        ax2.set_ylabel('t, °C', fontsize=14, fontname='Times New Roman')
        ax2.set_title('Цилиндр', fontsize=14, fontname='Times New Roman')
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(True, linestyle='--', alpha=0.3)

        fig.suptitle(f'Распределение температуры при $q_v = {self.q_v / 1e6:.0f}$ МВт/м$^3$',
                     fontsize=14, fontname='Times New Roman')

        ax1.legend(loc='best', fontsize=10, frameon=True, fancybox=True)
        ax2.legend(loc='best', fontsize=10, frameon=True, fancybox=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_with_temperature_limit(self, condition='ideal', save_path=None, show=True):
        """Построение графика с отображением предела температуры плавления"""
        condition_map = {
            'ideal': 'plane_ideal',
            'air': 'plane_air',
            'helium': 'plane_helium'
        }

        if condition not in condition_map:
            print(f"Ошибка: неизвестное условие '{condition}'")
            return None

        result_key = condition_map[condition]

        if result_key not in self.results:
            print("Ошибка: сначала запустите расчеты (run_all_tables)")
            return None

        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        df = self.results[result_key]
        ax.plot(df['x, мм'], df['Температура, °C'],
                color=self.colors[condition],
                linewidth=1,
                label=self.labels[condition],
                alpha=0.9)

        # Линия предела температуры плавления
        ax.axhline(y=self.t_m, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.7,
                   label=f'Температура плавления: $t_m = {self.t_m}^\\circ$C')

        xlabel = self.get_label('x')
        ylabel = self.get_label('t')
        title = 'Распределение температуры в пластине с ограничением по температуре плавления'

        self.format_axes(ax, xlabel, ylabel, title)

        max_temp = df['Температура, °C'].max()
        if max_temp > self.t_m:
            ax.fill_between(df['x, мм'], self.t_m, max_temp,
                            alpha=0.3, color='red',
                            label=f'Превышение предела: $t = {max_temp:.0f}^\\circ$C')

        ax.annotate(f'$q_v = {self.q_v / 1e6:.0f}$ МВт/м$^3$',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, fontname='Times New Roman')

        ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_qe_comparison(self, save_path=None, show=True):
        """Построение столбчатой диаграммы сравнения критических мощностей"""
        self.setup_plot_style()

        qe_values = {
            'Пластина\nидеальный': self.volumetric_heat_release_plane(),
            'Пластина\nвоздух': self.volumetric_heat_release_plane_air(),
            'Пластина\nгелий': self.volumetric_heat_release_plane_helium(),
            'Цилиндр\nидеальный': self.volumetric_heat_release_cylinder(),
            'Цилиндр\nвоздух': self.volumetric_heat_release_cylinder_air(),
            'Цилиндр\nгелий': self.volumetric_heat_release_cylinder_helium()
        }

        categories = list(qe_values.keys())
        values = [qe_values[cat] / 1e6 for cat in categories]

        bar_colors = []
        for cat in categories:
            if 'идеальный' in cat:
                bar_colors.append(self.colors['ideal'])
            elif 'воздух' in cat:
                bar_colors.append(self.colors['air'])
            else:
                bar_colors.append(self.colors['helium'])

        fig, ax = plt.subplots(figsize=(20, 10))

        bars = ax.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax.axhline(y=self.q_v / 1e6, color='red', linestyle='--',
                   linewidth=2, alpha=0.8,
                   label=f'Текущая мощность: $q_v = {self.q_v / 1e6:.0f}$ МВт/м$^3$')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                    f'{value:.0f}',
                    ha='center', va='bottom', fontsize=10, fontname='Times New Roman')

        ax.set_ylabel('Критическая мощность $q_e$, МВт/м$^3$', fontsize=14, fontname='Times New Roman')
        ax.set_title('Сравнение критических мощностей для различных условий', fontsize=14, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        # Поворачиваем подписи на оси X для лучшей читаемости
        for label in ax.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(11)

        ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_all(self, save_dir=None, show=True):
        """Построение всех графиков"""
        print("\n" + "=" * 80)
        print("ПОСТРОЕНИЕ ГРАФИКОВ")
        print("=" * 80)

        # Создаем папку для графиков, если указана
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)

        print("\n1. Построение графика для пластины...")
        self.plot_plane_distributions(
            save_path=f"{save_dir}/plane_distributions.png" if save_dir else None,
            show=show
        )

        print("2. Построение графика для цилиндра...")
        self.plot_cylinder_distributions(
            save_path=f"{save_dir}/cylinder_distributions.png" if save_dir else None,
            show=show
        )

        print("3. Построение графика сравнения пластина/цилиндр (идеальный контакт)...")
        self.plot_comparison_plane_cylinder(
            condition='ideal',
            save_path=f"{save_dir}/comparison_ideal.png" if save_dir else None,
            show=show
        )

        print("4. Построение графика сравнения пластина/цилиндр (воздушный зазор)...")
        self.plot_comparison_plane_cylinder(
            condition='air',
            save_path=f"{save_dir}/comparison_air.png" if save_dir else None,
            show=show
        )

        print("5. Построение графика сравнения пластина/цилиндр (гелиевый зазор)...")
        self.plot_comparison_plane_cylinder(
            condition='helium',
            save_path=f"{save_dir}/comparison_helium.png" if save_dir else None,
            show=show
        )

        print("6. Построение совмещенных графиков...")
        self.plot_all_conditions_comparison(
            save_path=f"{save_dir}/all_conditions_comparison.png" if save_dir else None,
            show=show
        )

        print("7. Построение графика с температурой плавления...")
        self.plot_with_temperature_limit(
            condition='ideal',
            save_path=f"{save_dir}/temperature_limit.png" if save_dir else None,
            show=show
        )

        print("8. Построение диаграммы критических мощностей...")
        self.plot_qe_comparison(
            save_path=f"{save_dir}/qe_comparison.png" if save_dir else None,
            show=show
        )

    def plot_gap_detail_plane(self, condition='air', save_path=None, show=True):
        """
        Детальный график распределения температуры в области зазора для ПЛАСТИНЫ
        Увеличенный фрагмент в границах:
        от self.d/2 - 0.0001 м до self.d/2 + self.c + 0.0001 м
        Шаг сетки по координате 0.01 мм

        Parameters:
        -----------
        condition : str
            'air' или 'helium' - какой зазор показать
        save_path : str, optional
            Путь для сохранения графика
        show : bool
            Показывать график
        """
        if condition == 'air':
            result_key = 'plane_air'
            gap_color = self.colors['air']
            line_style = '-'
        elif condition == 'helium':
            result_key = 'plane_helium'
            gap_color = self.colors['helium']
            line_style = '-'
        else:
            print("Ошибка: condition должен быть 'air' или 'helium'")
            return None

        if result_key not in self.results:
            print("Ошибка: сначала запустите расчеты (run_all_tables)")
            return None

        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        # Границы области просмотра в метрах
        x_start_m = self.d / 2 - 0.0001  # на 0.1 мм левее границы твэла
        x_end_m = self.d / 2 + self.c + 0.0001  # на 0.1 мм правее границы зазора

        # Переводим в мм для отображения
        x_start_mm = x_start_m * 1000
        x_end_mm = x_end_m * 1000

        # Создаем массив координат с шагом 0.01 мм (1e-5 м)
        step_m = 1e-5  # 0.01 мм
        x_detailed_m = np.arange(x_start_m, x_end_m + step_m, step_m)
        x_detailed_mm = x_detailed_m * 1000

        # Получаем аналитическое решение для твэла и оболочки
        fuel_solution = self.ODE_fuel_rod_plane()
        clearance_solution, t_01 = self.ODE_clearance_plane_air() if condition == 'air' else self.ODE_clearance_plane_helium()
        shell_solution = self.ODE_shell_plane_air() if condition == 'air' else self.ODE_shell_plane_he()

        # Вычисляем температуры в детальных точках
        t_detailed = np.zeros_like(x_detailed_m)

        for i, x in enumerate(x_detailed_m):
            if x <= self.d / 2:
                # Твэл
                t_func = sp.lambdify(self.x, fuel_solution.rhs, 'numpy')
                t_detailed[i] = t_func(x)
            elif x <= self.d / 2 + self.c:
                # Зазор
                t_func = sp.lambdify(self.x, clearance_solution.rhs, 'numpy')
                t_detailed[i] = t_func(x)
            else:
                # Оболочка
                t_func = sp.lambdify(self.x, shell_solution.rhs, 'numpy')
                t_detailed[i] = t_func(x)

        # Основной график
        ax.plot(x_detailed_mm, t_detailed,
                color=gap_color,
                linewidth=2,
                label=self.labels[condition],
                alpha=0.9)

        # Вертикальные линии границ
        x_fuel_end_mm = self.d / 2 * 1000
        x_gap_end_mm = (self.d / 2 + self.c) * 1000

        ax.axvline(x=x_fuel_end_mm, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=x_gap_end_mm, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

        # Подсветка области зазора
        ax.axvspan(x_fuel_end_mm, x_gap_end_mm, alpha=0.25, color=gap_color)

        # Точки на границах с температурами
        idx_fuel_end = np.argmin(np.abs(x_detailed_mm - x_fuel_end_mm))
        idx_gap_end = np.argmin(np.abs(x_detailed_mm - x_gap_end_mm))

        t_fuel_end = t_detailed[idx_fuel_end]
        t_gap_end = t_detailed[idx_gap_end]

        ax.plot(x_fuel_end_mm, t_fuel_end, 'ro', markersize=8, zorder=5)
        ax.plot(x_gap_end_mm, t_gap_end, 'ro', markersize=8, zorder=5)

        # Аннотации температур
        ax.annotate(f'{t_fuel_end:.1f}°C',
                    xy=(x_fuel_end_mm, t_fuel_end),
                    xytext=(x_fuel_end_mm + 0.03, t_fuel_end + 15),
                    fontsize=10, fontname='Times New Roman',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))

        ax.annotate(f'{t_gap_end:.1f}°C',
                    xy=(x_gap_end_mm, t_gap_end),
                    xytext=(x_gap_end_mm + 0.03, t_gap_end - 25),
                    fontsize=10, fontname='Times New Roman',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))

        # Перепад в зазоре
        delta_T = t_fuel_end - t_gap_end
        ax.annotate(f'ΔT = {delta_T:.1f}°C',
                    xy=((x_fuel_end_mm + x_gap_end_mm) / 2, (t_fuel_end + t_gap_end) / 2),
                    ha='center', fontsize=11, fontname='Times New Roman', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black'))

        # Настройка осей
        ax.set_xlim(x_start_mm, x_end_mm)
        ax.set_xlabel('x, мм', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('t, °C', fontsize=14, fontname='Times New Roman')
        ax.set_title(f'Детальное распределение температуры в области зазора (Пластина)\n{self.labels[condition]}',
                     fontsize=14, fontname='Times New Roman')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Информация о зазоре
        info_text = f'Толщина зазора: {self.c * 1000:.3f} мм\nШаг сетки: 0.01 мм'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, fontname='Times New Roman',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_gap_detail_plane_all(self, save_path=None, show=True):
        """
        Детальный график распределения температуры в области зазора для ПЛАСТИНЫ
        для всех трех условий установки (а, б, в) на одном графике

        Границы: от self.d/2 - 0.00002 м до self.d/2 + self.c + 0.00002 м
        Шаг сетки по координате: 0.005 мм
        """
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        # Границы области просмотра в метрах
        x_start_m = self.d / 2 - 0.00002  # на 0.02 мм левее границы твэла
        x_end_m = self.d / 2 + self.c + 0.00002  # на 0.02 мм правее границы зазора

        # Переводим в мм для отображения
        x_start_mm = x_start_m * 1000
        x_end_mm = x_end_m * 1000

        # Создаем массив координат с шагом 0.005 мм (5e-6 м)
        step_m = 5e-6  # 0.005 мм
        x_detailed_m = np.arange(x_start_m, x_end_m + step_m, step_m)
        x_detailed_mm = x_detailed_m * 1000

        # Условия для построения
        conditions = [
            ('plane_ideal', 'ideal', 'Идеальный контакт (а)'),
            ('plane_air', 'air', 'Воздушный зазор (б)'),
            ('plane_helium', 'helium', 'Гелиевый зазор (в)')
        ]

        for result_key, color_key, label in conditions:
            if result_key not in self.results:
                continue

            # Получаем аналитические решения
            fuel_solution = self.ODE_fuel_rod_plane()

            if result_key == 'plane_ideal':
                clearance_solution = None
                shell_solution = self.ODE_shell_plane()
            elif result_key == 'plane_air':
                clearance_solution, _ = self.ODE_clearance_plane_air()
                shell_solution = self.ODE_shell_plane_air()
            else:  # plane_helium
                clearance_solution, _ = self.ODE_clearance_plane_helium()
                shell_solution = self.ODE_shell_plane_he()

            # Вычисляем температуры в детальных точках
            t_detailed = np.zeros_like(x_detailed_m)

            for i, x in enumerate(x_detailed_m):
                if x <= self.d / 2:
                    # Твэл
                    t_func = sp.lambdify(self.x, fuel_solution.rhs, 'numpy')
                    t_detailed[i] = t_func(x)
                elif x <= self.d / 2 + self.c:
                    # Зазор (только для случаев с зазором)
                    if clearance_solution is not None:
                        t_func = sp.lambdify(self.x, clearance_solution.rhs, 'numpy')
                        t_detailed[i] = t_func(x)
                    else:
                        # Для идеального контакта продолжаем твэл
                        t_func = sp.lambdify(self.x, fuel_solution.rhs, 'numpy')
                        t_detailed[i] = t_func(x)
                else:
                    # Оболочка
                    t_func = sp.lambdify(self.x, shell_solution.rhs, 'numpy')
                    t_detailed[i] = t_func(x)

            # Построение графика
            ax.plot(x_detailed_mm, t_detailed,
                    color=self.colors[color_key],
                    linewidth=2,
                    label=label,
                    alpha=0.9)

        # Настройка осей
        ax.set_xlim(x_start_mm, x_end_mm)
        ax.set_xlabel('x, мм', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('t, °C', fontsize=14, fontname='Times New Roman')
        ax.set_title('Детальное распределение температуры в области зазора (Пластина)',
                     fontsize=14, fontname='Times New Roman')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Информация о параметрах
        info_text = f'Толщина зазора: {self.c * 1000:.3f} мм\nШаг сетки: 0.01 мм'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, fontname='Times New Roman',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_gap_detail_cylinder_all(self, save_path=None, show=True):
        """
        Детальный график распределения температуры в области зазора для ЦИЛИНДРА
        для всех трех условий установки (а, б, в) на одном графике

        Границы: от self.d/2 - 0.00002 м до self.d/2 + self.c + 0.00002 м
        Шаг сетки по координате: 0.005 мм
        """
        self.setup_plot_style()
        fig, ax = plt.subplots(figsize=(20, 10))

        # Границы области просмотра в метрах
        r_start_m = self.d / 2 - 0.00002  # на 0.02 мм левее границы твэла
        r_end_m = self.d / 2 + self.c + 0.00002  # на 0.02 мм правее границы зазора

        # Переводим в мм для отображения
        r_start_mm = r_start_m * 1000
        r_end_mm = r_end_m * 1000

        # Создаем массив координат с шагом 0.005 мм (5e-6 м)
        step_m = 5e-6  # 0.005 мм
        r_detailed_m = np.arange(r_start_m, r_end_m + step_m, step_m)
        r_detailed_mm = r_detailed_m * 1000

        # Условия для построения
        conditions = [
            ('cylinder_ideal', 'ideal', 'Идеальный контакт (а)'),
            ('cylinder_air', 'air', 'Воздушный зазор (б)'),
            ('cylinder_helium', 'helium', 'Гелиевый зазор (в)')
        ]

        for result_key, color_key, label in conditions:
            if result_key not in self.results:
                continue

            # Получаем аналитические решения
            fuel_solution = self.ODE_fuel_rod_cylinder()

            if result_key == 'cylinder_ideal':
                clearance_solution = None
                shell_solution = self.ODE_shell_cylinder()
            elif result_key == 'cylinder_air':
                clearance_solution, _ = self.ODE_clearance_cylinder_air()
                shell_solution = self.ODE_shell_cylinder_air()
            else:  # cylinder_helium
                clearance_solution, _ = self.ODE_clearance_cylinder_helium()
                shell_solution = self.ODE_shell_cylinder_he()

            # Вычисляем температуры в детальных точках
            t_detailed = np.zeros_like(r_detailed_m)

            for i, r in enumerate(r_detailed_m):
                if r <= self.d / 2:
                    # Твэл
                    t_func = sp.lambdify(self.r, fuel_solution.rhs, 'numpy')
                    t_detailed[i] = t_func(r)
                elif r <= self.d / 2 + self.c:
                    # Зазор (только для случаев с зазором)
                    if clearance_solution is not None:
                        t_func = sp.lambdify(self.r, clearance_solution.rhs, 'numpy')
                        t_detailed[i] = t_func(r)
                    else:
                        # Для идеального контакта продолжаем твэл
                        t_func = sp.lambdify(self.r, fuel_solution.rhs, 'numpy')
                        t_detailed[i] = t_func(r)
                else:
                    # Оболочка
                    t_func = sp.lambdify(self.r, shell_solution.rhs, 'numpy')
                    t_detailed[i] = t_func(r)

            # Построение графика
            ax.plot(r_detailed_mm, t_detailed,
                    color=self.colors[color_key],
                    linewidth=2,
                    label=label,
                    alpha=0.9)

        # Настройка осей
        ax.set_xlim(r_start_mm, r_end_mm)
        ax.set_xlabel('r, мм', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('t, °C', fontsize=14, fontname='Times New Roman')
        ax.set_title('Детальное распределение температуры в области зазора (Цилиндр)',
                     fontsize=14, fontname='Times New Roman')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Информация о параметрах
        info_text = f'Толщина зазора: {self.c * 1000:.3f} мм\nШаг сетки: 0.01 мм'
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, fontname='Times New Roman',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def plot_all(self, save_dir="./graphics", show=True):
        """Построение всех графиков"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("ПОСТРОЕНИЕ ГРАФИКОВ")
        print("=" * 80)
        print(f"Графики сохраняются в папку: {save_dir}")

        # 1-8. Основные графики...
        print("\n1. Построение графика для пластины...")
        self.plot_plane_distributions(
            save_path=os.path.join(save_dir, "plane_distributions.png"),
            show=show
        )

        print("2. Построение графика для цилиндра...")
        self.plot_cylinder_distributions(
            save_path=os.path.join(save_dir, "cylinder_distributions.png"),
            show=show
        )

        print("3. Построение графика сравнения пластина/цилиндр (идеальный контакт)...")
        self.plot_comparison_plane_cylinder(
            condition='ideal',
            save_path=os.path.join(save_dir, "comparison_ideal.png"),
            show=show
        )

        print("4. Построение графика сравнения пластина/цилиндр (воздушный зазор)...")
        self.plot_comparison_plane_cylinder(
            condition='air',
            save_path=os.path.join(save_dir, "comparison_air.png"),
            show=show
        )

        print("5. Построение графика сравнения пластина/цилиндр (гелиевый зазор)...")
        self.plot_comparison_plane_cylinder(
            condition='helium',
            save_path=os.path.join(save_dir, "comparison_helium.png"),
            show=show
        )

        print("6. Построение совмещенных графиков...")
        self.plot_all_conditions_comparison(
            save_path=os.path.join(save_dir, "all_conditions_comparison.png"),
            show=show
        )

        print("7. Построение графика с температурой плавления...")
        self.plot_with_temperature_limit(
            condition='ideal',
            save_path=os.path.join(save_dir, "temperature_limit.png"),
            show=show
        )

        print("8. Построение диаграммы критических мощностей...")
        self.plot_qe_comparison(
            save_path=os.path.join(save_dir, "qe_comparison.png"),
            show=show
        )

        # 9. Детальный график зазора для пластины (все условия)
        print("9. Построение детального графика области зазора для пластины...")
        self.plot_gap_detail_plane_all(
            save_path=os.path.join(save_dir, "gap_detail_plane_all.png"),
            show=show
        )

        # 10. Детальный график зазора для цилиндра (все условия)
        print("10. Построение детального графика области зазора для цилиндра...")
        self.plot_gap_detail_cylinder_all(
            save_path=os.path.join(save_dir, "gap_detail_cylinder_all.png"),
            show=show
        )

        print("\n✅ Все графики построены!")
        print(f"✅ Все графики сохранены в папку '{save_dir}'")


# Основной блок запуска
if __name__ == "__main__":
    sim = Simulation()

    # 1. Запуск расчетов (таблицы сохранятся в папку "tables")
    results = sim.run_all_tables(num_points=50, save_csv=True, save_dir="tables")

    # 2. Расчет q_e (таблица сохранится в папку "tables")
    print("\n" + "=" * 80)
    print("ЗАПУСК РАСЧЕТА КРИТИЧЕСКОЙ МОЩНОСТИ")
    print("=" * 80)
    qe_results = sim.print_qe_results()
    df_qe = sim.get_qe_dataframe(save_csv=True, save_dir="tables")
    print("\n" + "=" * 80)
    print("ТАБЛИЦА КРИТИЧЕСКИХ МОЩНОСТЕЙ")
    print("=" * 80)
    print(df_qe.to_string(index=False))

    # 3. Построение графиков (сохраняются в папку "graphics")
    plotter = Plotter()
    plotter.results = sim.results
    plotter.q_v = sim.q_v
    plotter.t_m = sim.t_m
    plotter.t_02_r = sim.t_02_r
    plotter.lambda_f = sim.lambda_f
    plotter.lambda_s = sim.lambda_s
    plotter.lambda_a = sim.lambda_a
    plotter.lambda_he = sim.lambda_he
    plotter.delta = sim.delta
    plotter.d = sim.d
    plotter.c = sim.c
    plotter.x = sim.x
    plotter.r = sim.r

    # Построение всех графиков с сохранением в папку "graphics"
    plotter.plot_all(save_dir="./graphics", show=True)