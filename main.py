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
        """
        Преобразует sympy решение в numpy массивы
        """
        if solution is None:
            raise ValueError("Решение не найдено")

        # Проверяем, содержит ли решение переменную координаты
        rhs = solution.rhs
        free_symbols = rhs.free_symbols

        # Если переменная координаты отсутствует в решении (константа)
        if coord_var not in free_symbols:
            # Создаем массив координат
            coord_array_m = np.linspace(coord_range[0], coord_range[1], num_points)
            # Температура постоянна
            temp_array = np.full(num_points, float(rhs))
            return coord_array_m, temp_array

        # Если переменная присутствует, преобразуем как обычно
        temp_func = sp.lambdify(coord_var, rhs, 'numpy')
        coord_array_m = np.linspace(coord_range[0], coord_range[1], num_points)
        temp_array = temp_func(coord_array_m)

        return coord_array_m, temp_array

    def get_plane_ideal_contact_table(self, num_points=20):
        """Таблица для пластины с идеальным контактом (случай а)"""

        fuel_solution = self.ODE_fuel_rod_plane()
        shell_solution = self.ODE_shell_plane()

        x_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.x, (0, self.d / 2), num_points
        )

        x_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.x, (self.d / 2, self.d / 2 + self.delta), num_points
        )

        # Проверка, что t_shell - массив
        print(f"t_fuel type: {type(t_fuel)}, shape: {t_fuel.shape if hasattr(t_fuel, 'shape') else 'scalar'}")
        print(f"t_shell type: {type(t_shell)}, shape: {t_shell.shape if hasattr(t_shell, 'shape') else 'scalar'}")

        # Объединяем данные (исключаем дублирование граничной точки)
        x_total_m = np.concatenate([x_fuel_m, x_shell_m[1:]])
        t_total = np.concatenate([t_fuel, t_shell[1:]])

        x_total_mm = x_total_m * 1000

        df = pd.DataFrame({
            'x, мм': np.round(x_total_mm, 4),
            'Температура, °C': np.round(t_total, 2)
        })

        self.results['plane_ideal'] = df
        return df

    def get_cylinder_ideal_contact_table(self, num_points=100):
        """Таблица для цилиндра с идеальным контактом (случай а)"""

        fuel_solution = self.ODE_fuel_rod_cylinder()
        shell_solution = self.ODE_shell_cylinder()

        r_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.r, (0, self.d / 2), num_points
        )

        r_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.r, (self.d / 2, self.d / 2 + self.delta), num_points
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

    def get_plane_air_gap_table(self, num_points=100):
        """Таблица для пластины с воздушным зазором (случай б)"""

        fuel_solution = self.ODE_fuel_rod_plane()
        clearance_solution = self.ODE_clearance_plane_air()
        shell_solution = self.ODE_shell_plane_gas()

        x_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.x, (0, self.d / 2), num_points
        )

        x_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.x, (self.d / 2, self.d / 2 + self.c), num_points
        )

        x_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.x, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points
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

    def get_plane_helium_gap_table(self, num_points=100):
        """Таблица для пластины с гелиевым зазором (случай в)"""

        fuel_solution = self.ODE_fuel_rod_plane()
        clearance_solution = self.ODE_clearance_plane_helium()
        shell_solution = self.ODE_shell_plane_gas()

        x_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.x, (0, self.d / 2), num_points
        )

        x_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.x, (self.d / 2, self.d / 2 + self.c), num_points
        )

        x_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.x, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points
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

    def get_cylinder_air_gap_table(self, num_points=100):
        """Таблица для цилиндра с воздушным зазором (случай б)"""

        fuel_solution = self.ODE_fuel_rod_cylinder()
        clearance_solution = self.ODE_clearance_cylinder_air()
        shell_solution = self.ODE_shell_cylinder_gas()

        r_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.r, (0, self.d / 2), num_points
        )

        r_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.r, (self.d / 2, self.d / 2 + self.c), num_points
        )

        r_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.r, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points
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

    def get_cylinder_helium_gap_table(self, num_points=100):
        """Таблица для цилиндра с гелиевым зазором (случай в)"""

        fuel_solution = self.ODE_fuel_rod_cylinder()
        clearance_solution = self.ODE_clearance_cylinder_helium()
        shell_solution = self.ODE_shell_cylinder_gas()

        r_fuel_m, t_fuel = self.solution_to_numpy(
            fuel_solution, self.r, (0, self.d / 2), num_points
        )

        r_clearance_m, t_clearance = self.solution_to_numpy(
            clearance_solution, self.r, (self.d / 2, self.d / 2 + self.c), num_points
        )

        r_shell_m, t_shell = self.solution_to_numpy(
            shell_solution, self.r, (self.d / 2 + self.c, self.d / 2 + self.c + self.delta), num_points
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

    def print_table(self, df, title="Распределение температуры", num_rows=None):
        """Вывод таблицы в консоль"""

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
        """Сохранение таблицы в CSV файл"""

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Таблица сохранена в файл: {filename}")

    def run_all_tables(self, num_points=50, save_csv=True):
        """Запуск всех расчетов и вывод таблиц"""

        print("\n" + "=" * 80)
        print("ЗАПУСК РАСЧЕТОВ ТЕМПЕРАТУРНЫХ ПОЛЕЙ")
        print("=" * 80)

        # Пластина, идеальный контакт
        df1 = self.get_plane_ideal_contact_table(num_points)
        self.print_table(df1, "ПЛАСТИНА - ИДЕАЛЬНЫЙ КОНТАКТ (случай а)")
        if save_csv:
            self.save_table_to_csv(df1, "plane_ideal_contact.csv")

        # Цилиндр, идеальный контакт
        df2 = self.get_cylinder_ideal_contact_table(num_points)
        self.print_table(df2, "ЦИЛИНДР - ИДЕАЛЬНЫЙ КОНТАКТ (случай а)")
        if save_csv:
            self.save_table_to_csv(df2, "cylinder_ideal_contact.csv")

        # Пластина, воздушный зазор
        df3 = self.get_plane_air_gap_table(num_points)
        self.print_table(df3, "ПЛАСТИНА - ВОЗДУШНЫЙ ЗАЗОР (случай б)")
        if save_csv:
            self.save_table_to_csv(df3, "plane_air_gap.csv")

        # Цилиндр, воздушный зазор
        df4 = self.get_cylinder_air_gap_table(num_points)
        self.print_table(df4, "ЦИЛИНДР - ВОЗДУШНЫЙ ЗАЗОР (случай б)")
        if save_csv:
            self.save_table_to_csv(df4, "cylinder_air_gap.csv")

        # Пластина, гелиевый зазор
        df5 = self.get_plane_helium_gap_table(num_points)
        self.print_table(df5, "ПЛАСТИНА - ГЕЛИЕВЫЙ ЗАЗОР (случай в)")
        if save_csv:
            self.save_table_to_csv(df5, "plane_helium_gap.csv")

        # Цилиндр, гелиевый зазор
        df6 = self.get_cylinder_helium_gap_table(num_points)
        self.print_table(df6, "ЦИЛИНДР - ГЕЛИЕВЫЙ ЗАЗОР (случай в)")
        if save_csv:
            self.save_table_to_csv(df6, "cylinder_helium_gap.csv")

        print("\n" + "=" * 80)
        print("✅ ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ")
        print("=" * 80)

        return self.results


# Пример использования
if __name__ == "__main__":
    sim = Simulation()

    # Запуск всех расчетов с сохранением CSV
    results = sim.run_all_tables(num_points=20, save_csv=True)

    # Или отдельный расчет
    # df = sim.get_plane_ideal_contact_table(num_points=10)
    # sim.print_table(df, "Моя таблица", num_rows=5)
    # sim.save_table_to_csv(df, "my_table.csv")