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

    def run_all_tables(self, num_points=50, save_csv=True):
        print("\n" + "=" * 80)
        print("ЗАПУСК РАСЧЕТОВ ТЕМПЕРАТУРНЫХ ПОЛЕЙ")
        print("=" * 80)

        df1 = self.get_plane_ideal_contact_table(num_points)
        self.print_table(df1, "ПЛАСТИНА - ИДЕАЛЬНЫЙ КОНТАКТ (случай а)")
        if save_csv:
            self.save_table_to_csv(df1, "plane_ideal_contact.csv")

        df2 = self.get_cylinder_ideal_contact_table(num_points)
        self.print_table(df2, "ЦИЛИНДР - ИДЕАЛЬНЫЙ КОНТАКТ (случай а)")
        if save_csv:
            self.save_table_to_csv(df2, "cylinder_ideal_contact.csv")

        df3 = self.get_plane_air_gap_table(num_points)
        self.print_table(df3, "ПЛАСТИНА - ВОЗДУШНЫЙ ЗАЗОР (случай б)")
        if save_csv:
            self.save_table_to_csv(df3, "plane_air_gap.csv")

        df4 = self.get_cylinder_air_gap_table(num_points)
        self.print_table(df4, "ЦИЛИНДР - ВОЗДУШНЫЙ ЗАЗОР (случай б)")
        if save_csv:
            self.save_table_to_csv(df4, "cylinder_air_gap.csv")

        df5 = self.get_plane_helium_gap_table(num_points)
        self.print_table(df5, "ПЛАСТИНА - ГЕЛИЕВЫЙ ЗАЗОР (случай в)")
        if save_csv:
            self.save_table_to_csv(df5, "plane_helium_gap.csv")

        df6 = self.get_cylinder_helium_gap_table(num_points)
        self.print_table(df6, "ЦИЛИНДР - ГЕЛИЕВЫЙ ЗАЗОР (случай в)")
        if save_csv:
            self.save_table_to_csv(df6, "cylinder_helium_gap.csv")

        print("\n" + "=" * 80)
        print("✅ ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ")
        print("=" * 80)
        return self.results

        # ==================== РАСЧЕТ КРИТИЧЕСКОЙ МОЩНОСТИ q_e ====================

    def calculate_qe_from_models(self):
        """
        Расчет критической мощности q_e на основе аналитических формул из Math_Model
        """
        # Сохраняем текущее значение q_v
        original_q_v = self.q_v

        # Временно устанавливаем символьное q_v для расчетов
        q_v_sym = sp.Symbol('q_v', positive=True)
        self.q_v = q_v_sym

        results = {}

        try:
            # ========== ПЛАСТИНА ==========

            # 1. Пластина, идеальный контакт
            fuel_sol = self.ODE_fuel_rod_plane()
            t_center = fuel_sol.rhs.subs(self.x, 0)
            # Приравниваем к t_m и решаем относительно q_v
            eq_ideal = sp.Eq(t_center, self.t_m)
            qe_ideal = sp.solve(eq_ideal, q_v_sym)
            results['plane_ideal'] = float(qe_ideal[0]) if qe_ideal else None

            # 2. Пластина, воздушный зазор
            # Получаем выражение для температуры после зазора
            _, t_01 = self.ODE_clearance_plane_air()
            # Перепад в оболочке
            q_flow = q_v_sym * (self.d / 2)
            delta_shell = q_flow * self.delta / self.lambda_s
            t_surface = t_01 - delta_shell
            # Температура в центре
            t_center_air = fuel_sol.rhs.subs(self.x, 0) + (self.t_02_r - t_surface)
            eq_air = sp.Eq(t_center_air, self.t_m)
            qe_air = sp.solve(eq_air, q_v_sym)
            results['plane_air'] = float(qe_air[0]) if qe_air else None

            # 3. Пластина, гелиевый зазор
            _, t_01_he = self.ODE_clearance_plane_helium()
            delta_shell_he = q_flow * self.delta / self.lambda_s
            t_surface_he = t_01_he - delta_shell_he
            t_center_he = fuel_sol.rhs.subs(self.x, 0) + (self.t_02_r - t_surface_he)
            eq_he = sp.Eq(t_center_he, self.t_m)
            qe_he = sp.solve(eq_he, q_v_sym)
            results['plane_helium'] = float(qe_he[0]) if qe_he else None

            # ========== ЦИЛИНДР ==========

            # 4. Цилиндр, идеальный контакт
            fuel_sol_cyl = self.ODE_fuel_rod_cylinder()
            t_center_cyl = fuel_sol_cyl.rhs.subs(self.r, 0)
            eq_cyl_ideal = sp.Eq(t_center_cyl, self.t_m)
            qe_cyl_ideal = sp.solve(eq_cyl_ideal, q_v_sym)
            results['cylinder_ideal'] = float(qe_cyl_ideal[0]) if qe_cyl_ideal else None

            # 5. Цилиндр, воздушный зазор
            _, t_01_cyl_air = self.ODE_clearance_cylinder_air()
            t_center_cyl_air = fuel_sol_cyl.rhs.subs(self.r, 0) + (self.t_02_r - t_01_cyl_air)
            eq_cyl_air = sp.Eq(t_center_cyl_air, self.t_m)
            qe_cyl_air = sp.solve(eq_cyl_air, q_v_sym)
            results['cylinder_air'] = float(qe_cyl_air[0]) if qe_cyl_air else None

            # 6. Цилиндр, гелиевый зазор
            _, t_01_cyl_he = self.ODE_clearance_cylinder_helium()
            t_center_cyl_he = fuel_sol_cyl.rhs.subs(self.r, 0) + (self.t_02_r - t_01_cyl_he)
            eq_cyl_he = sp.Eq(t_center_cyl_he, self.t_m)
            qe_cyl_he = sp.solve(eq_cyl_he, q_v_sym)
            results['cylinder_helium'] = float(qe_cyl_he[0]) if qe_cyl_he else None

        finally:
            # Восстанавливаем исходное значение q_v
            self.q_v = original_q_v

        return results

    def print_qe_results(self):
        """Вывод результатов расчета критической мощности"""
        print("\n" + "=" * 80)
        print("РАСЧЕТ КРИТИЧЕСКОЙ МОЩНОСТИ ТЕПЛОВЫДЕЛЕНИЯ q_e")
        print("=" * 80)
        print(f"Исходные условия:")
        print(f"  Максимальная допустимая температура сердечника t_m = {self.t_m} °C")
        print(f"  Температура наружной поверхности оболочки t_02_r = {self.t_02_r} °C")
        print(f"  Разница температур: Δt_max = {self.t_m - self.t_02_r} °C")

        qe_results = self.calculate_qe_from_models()

        print("\n" + "-" * 50)
        print("РЕЗУЛЬТАТЫ РАСЧЕТА q_e:")
        print("-" * 50)

        names = {
            'plane_ideal': 'Пластина, идеальный контакт',
            'plane_air': 'Пластина, воздушный зазор',
            'plane_helium': 'Пластина, гелиевый зазор',
            'cylinder_ideal': 'Цилиндр, идеальный контакт',
            'cylinder_air': 'Цилиндр, воздушный зазор',
            'cylinder_helium': 'Цилиндр, гелиевый зазор'
        }

        for key, name in names.items():
            qe = qe_results.get(key)
            if qe:
                print(f"\n{name}:")
                print(f"  q_e = {qe:.2e} Вт/м³ = {qe / 1e6:.2f} МВт/м³")

        # Сравнение с текущим q_v
        print("\n" + "-" * 50)
        print("СРАВНЕНИЕ С ТЕКУЩИМ РЕЖИМОМ:")
        print("-" * 50)
        print(f"Текущая мощность: q_v = {self.q_v / 1e6:.2f} МВт/м³")

        for key, name in names.items():
            qe = qe_results.get(key)
            if qe:
                margin = qe / self.q_v
                if margin > 1.2:
                    status = "✅ БЕЗОПАСНО"
                elif margin > 1.0:
                    status = "⚠️ ПРЕДЕЛЬНО"
                else:
                    status = "❌ ОПАСНО"
                print(f"  {name:35}: запас = {margin:.3f} → {status}")

        return qe_results

    def get_qe_dataframe(self):
        """Возвращает DataFrame с результатами расчета q_e"""
        qe_results = self.calculate_qe_from_models()

        data = []
        for key, qe in qe_results.items():
            if qe:
                if 'plane' in key:
                    geometry = 'Пластина'
                else:
                    geometry = 'Цилиндр'

                if 'ideal' in key:
                    contact = 'Идеальный контакт'
                elif 'air' in key:
                    contact = 'Воздушный зазор'
                else:
                    contact = 'Гелиевый зазор'

                data.append({
                    'Геометрия': geometry,
                    'Тип контакта': contact,
                    'q_e, Вт/м³': f"{qe:.2e}",
                    'q_e, МВт/м³': round(qe / 1e6, 2),
                    'Запас прочности': round(qe / self.q_v, 3)
                })

        df = pd.DataFrame(data)
        return df


sim = Simulation()

# 1. Запуск расчетов температурных полей
results = sim.run_all_tables(num_points=20, save_csv=True)

# 2. Расчет критической мощности q_e
print("\n" + "=" * 80)
print("ЗАПУСК РАСЧЕТА КРИТИЧЕСКОЙ МОЩНОСТИ")
print("=" * 80)

# Вывод результатов
qe_results = sim.print_qe_results()

# Получение DataFrame
df_qe = sim.get_qe_dataframe()
print("\n" + "=" * 80)
print("ТАБЛИЦА КРИТИЧЕСКИХ МОЩНОСТЕЙ")
print("=" * 80)
print(df_qe.to_string(index=False))

# Сохранение в CSV
df_qe.to_csv("critical_power.csv", index=False, encoding='utf-8-sig')
print("\n✓ Таблица критических мощностей сохранена в 'critical_power.csv'")