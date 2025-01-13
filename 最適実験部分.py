import numpy as np
import gurobipy as gp
import random
import math
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import matplotlib.font_manager as fm
#農機台数の平均が.233台数のときは
# 設定クラス
@dataclass
class SimulationConfig:
    """シミュレーションの設定を保持するクラス"""
    FARMERS: int = 5                                                     # 参加農家の数
    DAYS: int = 7                                                        # 対象期間の日数
    TESTS: int = 1                                                       # シミュレーション実験回数
    COST_MOVE: float = 8.9355/3                                          # 農機1台を移動させるコスト（20分あたり）
    COST_CULTIVATION: float = (6.5+7.1)/2                                # 農地開墾を金額換算した値
    WORK_EFFICIENCY: float = 12000/14                                    # 農機1台が1日に耕せる面積
    AREA_AVERAGE: float = 34000.0                                        # 農地面積の平均値
    NEW_FARMER_AREA_AVERAGE: float = 27000.0                             # 新規就農者の農地面積の平均値
    FARMER_UTILITY: float = 4/7                                          # 農家が事前に申告する効用確率（労働意欲の確率）
    AREA_RANGE: float = 10000.0                                          # 農地面積のばらつき幅
    NEW_FARMER_RATE_STEP: float = 0.2                                    # 新規就農者率のステップ
    UTILITY_START: int = 10                                              # 効用値の初期値
    UTILITY_STEP: int = 20                                               # 効用値のステップ
    UTILITY_MAX: int = 100                                               # 効用値の最大値
    NEW_FARMER_RATE_BEGIN: float = 0.00                                  # 新規就農者分析の初期割合
    NEW_FARMER_RATE_END: float = 0.21                                    # 新規就農者分析の最終割合
    NEW_FARMER_RATE_MIN: float = 0.0                                     # 新規就農者率の最小値
    NEW_FARMER_RATE_RANGE: float = 1/30                                  # 新規就農者の増加率
    NEW_FARMER_RATE_MAX: float = 0.21                                    # 新規就農者率の最大値
    WEATHER_RATE: float = 18/30                                          # 天気予報の確率
    THERE_IS_A_LIER: bool  = True                                        # 農家0が農機台数を虚偽申告するか否か

# カスタム例外
class OptimizationError(Exception):
    """最適化計算に関連するエラー"""
    def __init__(self, message):
        super().__init__(message)
        print("最適化計算部分でエラー発生")

def generate_random_with_mean(target_mean):
    # 乱数を生成（0から1の範囲）
    random_value = random.random()  # 0から1の乱数を生成
    # スケーリングして平均がターゲットになるように調整
    adjusted_value = random_value * target_mean / 0.5  # 0.5は0から1の平均
    return adjusted_value

# 天候管理クラス
class Weather:
    def __init__(self, days: int):
        self.days = days
        self.config = SimulationConfig()

    def generate_forecast(self) -> List[float]:
        """天気予報の生成"""
        return [generate_random_with_mean(self.config.WEATHER_RATE) for _ in range(self.days)]

    def generate_actual_weather(self, forecast: List[float]) -> List[int]:
        """実際の天気の生成"""
        return [0 if random.random() < prob else 1 for prob in forecast]

    def generate_weather_patterns(self) -> Dict[Tuple[int, int], int]:
        """天候パターンの生成"""
        W = {}
        for d in range(2 ** self.days):
            w_bin = bin(d)[2:].zfill(self.days)
            for w_range in range(self.days):
                W[d, w_range] = int(w_bin[w_range])
        return W


class Farm:
    """農家クラス"""
    def __init__(self, config, farm_id, is_new=False):
        self.config = config
        self.id = farm_id
        self.land_area = 0.0  # 農地面積の初期化
        self.machine_count = 0  # 農機台数の初期化
        self.is_new = is_new  # 新規就農者フラグ
        self.utility = []
        self.cultivated_area = 0.0  # 開墾面積の初期化

    def initialize_random(self, new_farmer_rate: float):
        """農家データのランダム初期化"""
        if self.is_new:
            self.land_area = self._generate_new_farmer_land_area()  # 新規就農者の農地面積を生成
        else:
            self.land_area = self._generate_land_area()  # 既存農家の農地面積を生成

        self.machine_count = self._generate_machine_count(new_farmer_rate)
        self.utility = self._generate_utility()

    def _generate_new_farmer_land_area(self) -> float:
        """新規就農者の農家の所持面積を割り当て"""
        min_area = self.config.NEW_FARMER_AREA_AVERAGE - self.config.AREA_RANGE
        if min_area < 0:
            raise ValueError("Invalid area parameters. Please check NEW_FARMER_AREA_AVERAGE, AREA_RANGE")
        return min_area + random.random() * 2 * self.config.AREA_RANGE

    def _generate_land_area(self) -> float:
        """農家の所持面積を割り当て"""
        min_area = self.config.AREA_AVERAGE - self.config.AREA_RANGE
        if min_area < 0:
            raise ValueError("Invalid area parameters. Please check AREA_AVERAGE, AREA_RANGE")
        return min_area + random.random() * 2 * self.config.AREA_RANGE

    def _generate_machine_count(self, new_farmer_rate: float) -> int:
        """農家の所持農機の割当（真値）"""
        if self.is_new:
            return 0
        a = random.random()         
        if a < 0.17:             
            return 1         
        elif a < 0.33:             
            return 2        
        else:             
            return 3

    def _generate_utility(self) -> List[int]:
        """各農家の効用を導出"""
        return [1 if random.random() < self.config.FARMER_UTILITY else 0 
                for _ in range(self.config.DAYS)]


class lier_Farm(Farm):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.config = config
        self.land_area: float = 0
        self.machine_count: int = 0
        self.utility: List[int] = []


class FarmingOptimizer:
    """最適化クラス"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.weather = Weather(config.DAYS)
        self.reset_model()

    def reset_model(self):
        """最適化モデルのリセット"""
        self.model = gp.Model("FarmingOptimization")
        self.model.setParam('OutputFlag', 0)  # ログ出力を無効にする
        self.model.setParam('LogToConsole', 0)
        self.model.setParam('TimeLimit', 30000)
        self.model.setParam('MIPGap', 0.1)


    def optimize(self, farms: List[Farm], weather_forecast: List[float], 
                actual_weather: List[int], W, Calculate_FARMER0_profit: bool) -> Dict:
        """最適化の実行とVCGオークションによる支払金額の計算"""
        Farmer_payments_truth = []
        Farmer_payments_lie = []
        Profit_truth = []
        Profit_FARMER0_lie = []
        payment = []
        payment_lie = []
        Social_surplus_truth = 0
        Social_surplus_lie = 0
        Farmer_payments = []
        Farmer_profit = []
        weather_pattern = weather_forecast
        not_i = []
        remove_i = []
        # リストの初期化を修正
        how_many_farmer_trasnsport_truth = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]
        how_many_farmer_trasnsport_lie = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]
        how_many_farmer_land_area_truth = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]
        how_many_farmer_land_area_lie = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]
        how_many_farmer_schedule_truth = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]
        how_many_farmer_schedule_lie = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]
        farmers_available_schedule = [[0] * self.config.DAYS for _ in range(self.config.FARMERS)]

        for d in range(self.config.DAYS):
            for farm in farms:
                farmers_available_schedule[farm.id][d] = farm.utility[d]
            # メインの最適化モデルをリセットし、設定を行う
            self.reset_model()
            self.model.setParam('OutputFlag', 0)  # ログ出力を無効にする
            vars = self.create_variables(boolean_number=False,farm_i=0,machine_count=0)  # 変数の作成
            self.set_constraints(vars, farms, weather_forecast,farm_i=0,boolean_number=False)  # 制約の設定
            self.set_objective(vars, farms)  # 目的関数の設定
            self.model.optimize()  # 最適化の実行

            # 最適化が成功したか確認
            if self.model.status != gp.GRB.OPTIMAL:
                raise OptimizationError(f"最適化に失敗しました。ステータス: {self.model.status}")
            Profit=[]#個人個人の利益
            t =1.0
            for w in range(2 ** self.config.DAYS):
                t *= vars['z'][w].X
            for i in range(self.config.FARMERS):
                Profit.append(self.config.COST_CULTIVATION * 
                            (sum(vars['c'][i, w].X * vars['z'][w].X for w in range(2**self.config.DAYS))) 
                                 
                            - self.config.COST_MOVE * sum(vars['t'][i, d].X for d in range(self.config.DAYS))
                            )
            print(f"最適化成功。現在の目的関数値: { self.model.ObjVal}")
            
            
            Obj_present = self.model.ObjVal  # 正直に申告したときの目的関数の値
            results = self.get_results(vars, farms, actual_weather, weather_forecast, W)
            Profit_present = []
            for i in range(self.config.FARMERS):
                
                Profit_present.append(Obj_present - Profit[i])
                a = b = c= 0
                for d in range(self.config.DAYS):
                    how_many_farmer_trasnsport_truth[i][d] = vars['t'][i, d].X
                    how_many_farmer_land_area_truth[i][d] = sum(vars['c'][i, w].X*vars['z'][w].X for w in range(2 ** self.config.DAYS))
                    how_many_farmer_schedule_truth[i][d] = vars['s'][i, d].X
            # 各農家の支払金額を計算
            for i in range(self.config.FARMERS):
                Profit_absent = []
                print(f"\n農家{i}の支払金額を計算中...")
                
                # 農家iを除外した新しい農家リストを作成
                farmi_machine_count = farms[i].machine_count
                farmi_land_area = farms[i].land_area
                farms[i].machine_count = 0
                farms[i].land_area = 0
                #farms_without_i = [farm for farm in farms if farm.id != i]
                
                # 農家iを除外した最適化モデルを設定
                self.reset_model()
                self.model.setParam('OutputFlag', 0)  # ログ出力を無効にする
                vars_removed = self.create_variables(boolean_number=True,farm_i=i,machine_count=farms[i].machine_count)  # 変数の作成
                self.set_constraints(vars_removed, farms, weather_forecast,farm_i=i,boolean_number=False)  # 制約の設定
                self.set_objective(vars_removed, farms)  # 目的関数の設定
                self.model.optimize()  # 最適化の実行
                farms[i].machine_count = farmi_machine_count
                farms[i].land_area = farmi_land_area
                
                # 最適化が成功したか確認
                if self.model.status != gp.GRB.OPTIMAL:
                    raise OptimizationError(f"農家{i}を除外した最適化に失敗しました。ステータス: {self.model.status}")
                
                Obj_absent = self.model.ObjVal  # 農家iを除外した場合の目的関数の値
                #print(f"農家{i}を除外した場合の目的関数値: {Obj_absent}")
                not_i.append(Profit_present[i])
                remove_i.append(Obj_absent)
                # VCG支払金額の計算
                payment_i = Profit_present[i] -Obj_absent
                Farmer_payments.append(payment_i)
                print(f"農家{i}の利益: {Profit[i]}")
                print(f"農家{i}の支払額: {payment_i}")
                Farmer_profit.append(Profit[i]+payment_i)
                print(f"農家{i}の利得: {Profit[i]+payment_i}")
                farms[i].machine_count = farmi_machine_count
                farms[i].land_area = farmi_land_area
                Farmer_payments_truth.append(payment_i)
                Profit_truth.append(Profit[i])
                Social_surplus_truth += Profit[i]+payment_i

            # 支払金額の総和を計算
            """全農家の支払額を計算"""
            total_payment = sum(Farmer_payments)
            print(f"\n全農家の総支払額: {total_payment}")
            print(f"全農家の利得(社会的余剰): {sum(Farmer_profit)}")




            """農家0が嘘をついたときの獲得利益"""
            farms0_machine_count = farms[0].machine_count
            farms[0].machine_count = 0
            self.reset_model()
            self.model.setParam('TimeLimit', 30000)  # 時間制限を10分に設定
            self.model.setParam('OutputFlag', 0)  # ログ出力を無効にする
            self.model.setParam('NodefileStart', 0.5)
            vars = self.create_variables(boolean_number=False,farm_i=0,machine_count=farms0_machine_count)#変数の作成
            self.set_constraints(vars, farms, weather_forecast,farm_i=0,boolean_number=False)#制約の設定
            self.set_objective(vars, farms)#目的関数の設定
            self.model.optimize()#最適化の実行

            if self.model.status != gp.GRB.OPTIMAL:
                print("モデルは実行不可能です。診断情報を生成します。")
                self.model.computeIIS()
                self.model.write("model.ilp")
                print("診断情報が 'model.ilp' に保存されました。")
                raise OptimizationError(f"農家0が嘘をついたときの最適化に失敗しました。ステータス: {self.model.status}")
            Profit_lie=[]
            Profitaaa=[]
            for i in range(self.config.FARMERS):
                Profitaaa.append(self.config.COST_CULTIVATION * 
                            (sum(vars['c'][i, w].X * vars['z'][w].X for w in range(2**self.config.DAYS))) 
                                 
                            - self.config.COST_MOVE * sum(vars['t'][i, d].X for d in range(self.config.DAYS))
                            )
            Profit_lie1 = []#農家i以外の獲得利益
            for i in range(self.config.FARMERS):
                Profit_lie.append(self.model.ObjVal-Profitaaa[i])
                if i==0:
                    farmer0_profit_difference = sum(self.config.COST_CULTIVATION *self.config.WORK_EFFICIENCY*farms0_machine_count * actual_weather[d]*farms[0].utility[d] for d in range(self.config.DAYS))
                    Profit_lie[i] = Profit_lie[i] + farmer0_profit_difference
                    if Profit_lie[i] > self.config.COST_CULTIVATION * farms[0].land_area:
                        Profit_lie[i] = self.config.COST_CULTIVATION * farms[0].land_area
            for i in range(self.config.FARMERS):
                Profit_lie1.append(self.model.ObjVal-Profit_lie[i])
                for d in range(self.config.DAYS):
                    how_many_farmer_trasnsport_lie[i][d] = vars['t'][i, d].X
                    how_many_farmer_land_area_lie[i][d] = sum(vars['c'][i, w].X*vars['z'][w].X for w in range(2 ** self.config.DAYS))
                    how_many_farmer_schedule_lie[i][d] = vars['s'][i, d].X
            print("-----------農家oがうそをついたとき------------------------")
            Obj_lie = self.model.ObjVal  # 農家0が嘘をついたときの目的関数の値
            print(f"最適化成功。現在の目的関数値: {Obj_present}")

            Farmer_payment_inlier =[]
            Farmer_profit1 = []
            """農家0が嘘をついたときの支払額の計算"""
            # 各農家の支払金額を計算
            for i in range(self.config.FARMERS):
                
                Profit_absent = []
                print(f"\n農家0が嘘をついたときの農家{i}の支払金額を計算中...")
                
                # 農家iを除外した新しい農家リストを作成
                farmi_machine_count = farms[i].machine_count
                farmi_land_area = farms[i].land_area
                farms[i].machine_count = 0
                farms[i].land_area = 0
                #farms_without_i = [farm for farm in farms if farm.id != i]
                
                # 農家iを除外した最適化モデルを設定
                self.reset_model()
                self.model.setParam('OutputFlag', 0)  # ログ出力を無効にする
                vars_removed = self.create_variables(boolean_number=True,farm_i=i,machine_count=farms[i].machine_count)  # 変数の作成
                self.set_constraints(vars_removed, farms, weather_forecast,farm_i=i,boolean_number=True)  # 制約の設定
                self.set_objective(vars_removed, farms)  # 目的関数の設定
                self.model.optimize()  # 最適化の実行
                farms[i].machine_count = farmi_machine_count
                farms[i].land_area = farmi_land_area

                # 最適化が成功したか確認
                if self.model.status != gp.GRB.OPTIMAL:
                    raise OptimizationError(f"農家0が嘘をついたときの農家{i}を除外した最適化に失敗しました。ステータス: {self.model.status}")
                
                Obj_absent = self.model.ObjVal  # 農家iを除外した場合の目的関数の値
                #print(f"農家0が嘘をついたときの農家{i}を除外した場合の目的関数値: {Obj_absent}")
                
                # VCG支払金額の計算
                payment_i = Profit_lie1[i] - Obj_absent
                Farmer_payment_inlier.append(payment_i)
                print(f"農家{i}の利益: {Profit_lie[i]}")
                print(f"農家{i}の支払額: {payment_i}")
                Farmer_profit1.append(Profit_lie[i]+payment_i)
                print(f"農家{i}の利得: {Profit_lie[i]+payment_i}")
                farms[i].machine_count = farmi_machine_count
                farms[i].land_area = farmi_land_area
                Farmer_payments_lie.append(payment_i)
                Profit_FARMER0_lie.append(Profit_lie[i])
                Social_surplus_lie += Profit_lie1[i]+payment_i
            
            # 支払金額の総和を計算
            """全農家の支払額を計算"""
            total_payment = sum(Farmer_payments)
            print(f"\n全農家の総支払額: {total_payment}")
            print(f"全農家の利得(社会的余剰): {sum(Farmer_profit)}")


            # 支払金額の総和を計算
            """農家0が嘘をついたとき全農家の支払額を計算"""
            total_payment = sum(Farmer_payment_inlier)
            print(f"\n全農家の総支払額: {total_payment}")
            print(f"全農家の利得(社会的余剰): {sum(Farmer_profit1)}")


            #raise OptimizationError(f"農家0が嘘をついたときの最適化に失敗しました。ステータス: {self.model.status}")
            #farms[0].land_area = farms0_land_area
            farms[0].machine_count = farms0_machine_count



            # 結果を辞書としてまとめる
            
            print("-----------------------------------")
            print("-----------------------------------")
            # results['objective_present'] = Obj_present
            # results['Farmer_payments'] = Farmer_payments
            # results['total_payment'] = total_payment
            # #results['total_payment_inlier'] = total_payment_inlier
            # results['Farmer_profit'] = Farmer_profit
            # results['Farmer_profit1'] = Farmer_profit1
            profit_difference_farm0 = Profit_truth[0] - Profit_lie[0]
            social_surplus_difference = Social_surplus_truth - Social_surplus_lie
            total_payment_lie = sum(Farmer_payments_lie)
            total_payment_truth = sum(Farmer_payments_truth)


            results = {
                'objective_present': Obj_present,
                'objective_lie': Obj_lie,
                'Farmer_payments_truth': Farmer_payments_truth,
                'Farmer_payments_lie': Farmer_payments_lie,
                'Profit_truth': Profit_truth,
                'Profit_lie': Profit_FARMER0_lie,
                'profit_difference_farm0': profit_difference_farm0,
                'social_surplus_difference': social_surplus_difference,
                'total_payment_truth': total_payment_truth,
                'total_payment_lie': total_payment_lie,
                'farmer0_profit_difference': farmer0_profit_difference,
                'how_many_farmer_trasnsport_truth': how_many_farmer_trasnsport_truth,
                'how_many_farmer_trasnsport_lie': how_many_farmer_trasnsport_lie,
                'how_many_farmer_land_area_truth': how_many_farmer_land_area_truth,
                'how_many_farmer_land_area_lie': how_many_farmer_land_area_lie,
                'payment':payment,
                'payment_lie':payment_lie,
                'how_many_farmer_schedule_truth': how_many_farmer_schedule_truth,
                'how_many_farmer_schedule_lie': how_many_farmer_schedule_lie,
                'weather_pattern': weather_pattern,
                'farmers_available_schedule' : farmers_available_schedule,
                'not_i': not_i,
                'remove_i': remove_i
            }
            
            return results
        
        

    def create_variables(self,boolean_number:bool,farm_i :int,machine_count:int) -> Dict:
        """最適化変数の作成"""
        vars = {
            's': {},  # 作業量変数
            'c': {},  # 容量変数
            't': {},  # 移動変数
            'z': {}   # 天候変数
        }
        if boolean_number == True:
            for i in range(self.config.FARMERS):
                for d in range(self.config.DAYS):
                    vars['s'][i, d] = self.model.addVar(vtype=gp.GRB.INTEGER)
                    vars['t'][i, d] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
                    if i == farm_i:
                        self.model.addConstr(vars['s'][i, d] == machine_count)
                        self.model.addConstr(vars['t'][i, d] == 0)
                for w in range(2 ** self.config.DAYS):
                    vars['c'][i, w] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
            for w in range(2 ** self.config.DAYS):
                vars['z'][w] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
            return vars
        elif boolean_number == False:
            for i in range(self.config.FARMERS):
                for d in range(self.config.DAYS):
                    vars['s'][i, d] = self.model.addVar(vtype=gp.GRB.INTEGER)
                    vars['t'][i, d] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
                for w in range(2 ** self.config.DAYS):
                    vars['c'][i, w] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
            for w in range(2 ** self.config.DAYS):
                vars['z'][w] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
            return vars
        else:
            raise ValueError("boolean_numberはTrueまたはFalseでなければなりません。")

    def set_constraints(self, vars: Dict, farms: List[Farm], weather_forecast: List[float],farm_i:int,boolean_number:bool):
        """制約条件の設定"""
        W = self.weather.generate_weather_patterns()
        
        # 天候制約
        for w in range(2 ** self.config.DAYS):
            prod = self._calculate_weather_probability(w, weather_forecast, W)
            self.model.addConstr(vars['z'][w] == prod)

        # 農家ごとの制約
        for i in range(self.config.FARMERS):
            
            # 容量制約
            for w in range(2 ** self.config.DAYS):
                self.model.addConstr(vars['c'][i, w] <= farms[i].land_area)  # c_{i,w}<=a_i
                self.model.addConstr(
                    vars['c'][i, w] <= gp.quicksum(
                        W[w, d] * vars['s'][i, d] * farms[i].utility[d] * 
                        self.config.WORK_EFFICIENCY for d in range(self.config.DAYS)  # c_{i,w}<=Σw_d*s_{i,d}*n*e_{i,d}
                    )
                )
            
            # 作業量と移動の制約
            for d in range(self.config.DAYS):
                self.model.addConstr(vars['s'][i, d] >= 0)  # s_{i,d}>=0
                self.model.addConstr(vars['t'][i, d] >= 0)  # t_{i,d}>=0
                self.model.addConstr(
                    vars['t'][i, d] >= vars['s'][i, d] - farms[i].machine_count  # t_{i,d}>= s_{i,d}-m_i
                )

        # 総農機具数制約
        for d in range(self.config.DAYS):
            self.model.addConstr(
                gp.quicksum(vars['s'][i, d] for i in range(self.config.FARMERS)) 
                == sum(farm.machine_count for farm in farms)  # Σs_{i,d}=Σm_i
            )

    def set_objective(self, vars: Dict, farms: List[Farm],):
        """目的関数の設定"""
        objective = (
            self.config.COST_CULTIVATION * 
            gp.quicksum(vars['c'][i, w] * vars['z'][w] 
                       for i in range(self.config.FARMERS) 
                       for w in range(2 ** self.config.DAYS))
            - self.config.COST_MOVE * 
            gp.quicksum(vars['t'][i, d] 
                       for i in range(self.config.FARMERS)
                       for d in range(self.config.DAYS))
        )
        self.model.setObjective(objective, gp.GRB.MAXIMIZE)



    def get_results(self, vars: Dict, farms: List[Farm], actual_weather: List[int], weather_forecast, W) -> Dict:
        """最適化結果の取得"""

        results = {
            'objective_value': self.model.ObjVal,
            'farm_results': [],
            'total_payment_present': 0.0,
            'total_payment_inlier': 0.0,
            'payment_FARMER0': 0.0,
            'payment_FARMER0_inlier': 0.0,
            'profit_difference': 0.0,  # 初期値
            'social_surplus_difference': 0.0,  # 社会的余剰の差分初期値
            'profit_FARMER0': 0.0 ,#農家0の利益初期値
            'objective_present': 0.0,
            'objective_lie': 0.0,
            'Farmer_payments_truth': [],
            'Farmer_payments_lie': [],
            'Profit_truth': [],
            'Profit_lie': [],
            'profit_difference_farm0': 0.0,
            'social_surplus_difference': 0.0,
            'total_payment_truth': 0.0,
            'total_payment_lie': 0.0,
            'farmer0_profit_difference': 0.0,
            'how_many_farmer_trasnsport_truth': [],
            'how_many_farmer_trasnsport_lie': [],
            'how_many_farmer_land_area_truth': [],
            'how_many_farmer_land_area_lie': [],
            'payment':[],
            'payment_lie':[],
            'how_many_farmer_schedule_truth': [],
            'how_many_farmer_schedule_lie': [],
            'weather_pattern': [],
            'farmers_available_schedule' : [],
            'not_i': [],
            'remove_i': []





        }
        
        
        for i in range(self.config.FARMERS):
            farm_result = {
                'farm_id': i,
                'machine_count': farms[i].machine_count,
                'land_area': farms[i].land_area,
                'work_schedule': {d: vars['s'][i, d].X for d in range(self.config.DAYS)},
                'transfers_truth': {d: vars['t'][i, d].X for d in range(self.config.DAYS)},
                'transfers_lie': {d: vars['t'][i, d].X for d in range(self.config.DAYS)},
                'capacity': {w: vars['c'][i, w].X for w in range(2 ** self.config.DAYS)},
                'cultivated_area': farms[i].cultivated_area
            }
            results['farm_results'].append(farm_result)
        
        return results

    def _calculate_weather_probability(self, w: int, forecast: List[float], W: Dict) -> float:
        """制約条件式 Π_{d∈D}|w_d-P_d|の設定"""
        prod = 1.0
        for d in range(self.config.DAYS):
            diff = abs(W[w, d] - forecast[d])
            if diff < 1e-10:
                continue
            prod *= diff
        return prod

    def _calculate_payment(self, vars: Dict, farms: List[Farm], actual_weather: List[int]) -> float:
            """支払額の計算"""
            total_payment = 0
            for i in range(self.config.FARMERS):
                farm_payment = 0
                for d in range(self.config.DAYS):
                    if actual_weather[d] == 1:
                        work_done = vars['s'][i, d].X * farms[i].utility[d]
                        farm_payment += self.config.COST_CULTIVATION * min(
                            work_done,
                            farms[i].land_area
                        )
                        farm_payment -= self.config.COST_MOVE * vars['t'][i, d].X
                total_payment += farm_payment
            return total_payment


def calculate_social_surplus(farms):
    """社会的余剰を計算する関数"""
    total_surplus = 0
    for farm in farms:
        total_surplus += sum(farm.utility) * farm.machine_count * farm.land_area    # 各農家の利益を合計
    return total_surplus

def calculate_average_profit_for_new_farmers(farms, new_farmers):
    """新規就農者ごとの農家0の実際の利益の平均を計算する関数"""
    total_profit = 0
    count = len(new_farmers)  # 新規就農者の数
    if count == 0:
        return 0  # 新規就農者がいない場合は0を返す

    # 農家0の利益を計算
    profit = sum(farms[0].utility) * farms[0].machine_count  # 農家0の利益
    total_profit += profit * count  # 新規就農者の数で利益を掛ける

    return total_profit / count  # 平均を計算して返す

def generate_random_with_mean_and_surplus(farms, new_farmers, target_mean=0.3):
    """乱数を生成し、社会的余剰と平均利益を出力する関数"""
    random_value = random.random()  # 0から1の乱数を生成
    adjusted_value = random_value * target_mean / 0.5  # スケーリング

    # 農家0の所持農機台数が0台のときの社会的余剰を計算
    if farms[0].machine_count == 0:
        social_surplus = calculate_social_surplus(farms)
        print(f"Social surplus when farm 0 has 0 machines: {social_surplus}")

    # 新規就農者ごとの農家0の実際の利益の平均を計算
    #average_profit = calculate_average_profit_for_new_farmers(farms, new_farmers)
    #print(f"Average profit for new farmers for farm 0: {average_profit}")

    return adjusted_value

class FarmingSimulation:
    """農業シミュレーションを実行するクラス"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.optimizer = FarmingOptimizer(config)
        self.results = {}  # 新規就農率ごとのテスト結果を格納する辞書

    def run(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        
        rates = np.arange(self.config.NEW_FARMER_RATE_BEGIN, 
                          self.config.NEW_FARMER_RATE_END + self.config.NEW_FARMER_RATE_STEP, 
                          self.config.NEW_FARMER_RATE_STEP)
        
        for rate in rates:
            self.results[rate] = []
            for test_num in range(1, self.config.TESTS + 1):
                print(f"\n新規就農者率: {rate*100:.1f}%, テスト番号: {test_num}")
                
                # 農家のリストを作成
                farms = self._create_farms(self.config, rate)
                
                # 天気予報と実際の天気を生成
                forecast = self.optimizer.weather.generate_forecast()
                actual_weather = self.optimizer.weather.generate_actual_weather(forecast)
                W = self.optimizer.weather.generate_weather_patterns()
                
                # 最適化を実行
                test_result = self.optimizer.optimize(farms, forecast, actual_weather, W, Calculate_FARMER0_profit=True)
                
                # 各農家の詳細な結果を収集
                farm_results = []
                for farm in farms :
                        farm_result = {
                            'farm_id': farm.id,
                            'machine_count': farm.machine_count,
                            'land_area': farm.land_area,
                            '売上': test_result.get('Profit_truth', [])[farm.id] if 'Profit_truth' in test_result and farm.id < len(test_result['Profit_truth']) else '',
                            '支払額': test_result.get('Farmer_payments_truth', [])[farm.id] if 'Farmer_payments_truth' in test_result and farm.id < len(test_result['Farmer_payments_truth']) else '',
                            '利得': (test_result.get('Profit_truth', [])[farm.id] + 
                                    test_result.get('Farmer_payments_truth', [])[farm.id]
                                    ) if 'Profit_truth' in test_result and 'Farmer_payments_truth' in test_result and farm.id < len(test_result['Profit_truth']) and farm.id < len(test_result['Farmer_payments_truth']) else '',
                            '移動回数': test_result.get('how_many_farmer_trasnsport_truth', [])[farm.id] if 'how_many_farmer_trasnsport_truth' in test_result and farm.id < len(test_result['how_many_farmer_trasnsport_truth']) else '',
                            '開墾面積': test_result.get('how_many_farmer_land_area_truth', [])[farm.id] if 'how_many_farmer_land_area_truth' in test_result and farm.id < len(test_result['how_many_farmer_land_area_truth']) else '',
                            '作業スケジュール': test_result.get('how_many_farmer_schedule_truth', [])[farm.id] if 'how_many_farmer_schedule_truth' in test_result and farm.id < len(test_result['how_many_farmer_schedule_truth']) else '',
                            '農家0が嘘をついたときの利益': test_result.get('Profit_lie', [])[farm.id] if 'Profit_lie' in test_result and farm.id < len(test_result['Profit_lie']) else '',
                            '農家0が嘘をついたときの支払額': test_result.get('Farmer_payments_lie', [])[farm.id] if 'Farmer_payments_lie' in test_result and farm.id < len(test_result['Farmer_payments_lie']) else '',
                            '農家0が嘘をついたときの利得': (test_result.get('Profit_lie', [])[farm.id] + 
                                    test_result.get('Farmer_payments_lie', [])[farm.id]
                                    ) if 'Profit_lie' in test_result and 'Farmer_payments_lie' in test_result and farm.id < len(test_result['Profit_lie']) and farm.id < len(test_result['Farmer_payments_lie']) else '',
                            '農家0が嘘をついたときの移動回数': test_result.get('how_many_farmer_trasnsport_lie', [])[farm.id] if 'how_many_farmer_trasnsport_lie' in test_result and farm.id < len(test_result['how_many_farmer_trasnsport_lie']) else '',
                            '農家0が嘘をついたときの開墾面積': test_result.get('how_many_farmer_land_area_lie',  [])[farm.id] if 'how_many_farmer_land_area_lie' in test_result and farm.id < len(test_result['how_many_farmer_land_area_lie']) else '',
                            '農家0が嘘をついたときの作業スケジュール': test_result.get('how_many_farmer_schedule_lie', [])[farm.id] if 'how_many_farmer_schedule_lie' in test_result and farm.id < len(test_result['how_many_farmer_schedule_lie']) else '',
                            '天気予報':  test_result.get('weather_pattern', [])[farm.id] if 'weather_pattern' in test_result and farm.id < len(test_result['weather_pattern']) else '',
                            '農家0が嘘をついたときの農家0の利用可能なスケジュール': test_result.get('farmers_available_schedule', [])[farm.id] if 'farmers_available_schedule' in test_result and farm.id < len(test_result['farmers_available_schedule']) else '',
                            '全体の社会-i': test_result.get('not_i', [])[farm.id] if 'not_i' in test_result and farm.id < len(test_result['not_i']) else '',
                            '-iの社会': test_result.get('remove_i', [])[farm.id] if 'remove_i' in test_result and farm.id < len(test_result['remove_i']) else '',
                        }
                        farm_results.append(farm_result)

                if len(farms) < self.config.DAYS:
                        for d in range(self.config.DAYS-len(farms)):
                            farm_result = {
                                'farm_id': 0,
                                'machine_count': 0,
                                'land_area': 0,
                                '売上': 0,
                                '支払額': 0,
                                '利得': 0,
                                '移動回数': 0,
                                '開墾面積': 0,
                                '作業スケジュール': 0,
                                '農家0が嘘をついたときの利益': 0,
                                '農家0が嘘をついたときの支払額': 0,
                                '農家0が嘘をついたときの利得': 0,
                                '農家0が嘘をついたときの移動回数':0,
                                '農家0が嘘をついたときの開墾面積':0,
                                '農家0が嘘をついたときの作業スケジュール': 0,
                                '天気予報':  test_result.get('weather_pattern', [])[d+len(farms)] if 'weather_pattern' in test_result and farm.id < len(test_result['weather_pattern']) else '',
                                '農家0が嘘をついたときの農家0の利用可能なスケジュール': 0,
                                '全体の社会-i': 0 ,
                                '-iの社会': 0 }
                            farm_results.append(farm_result)
                # テスト結果に 'farm_results' を追加
                test_result['farm_results'] = farm_results
                
                # 結果を保存
                self.results[rate].append(test_result)

    def visualize_tractor_transfers(self):
        """新規就農者ごとのトラクター移動回数を可視化"""
        if not self.results:
            print("可視化するデータがありません")
            return

        # データを整形
        transfer_data = []
        for rate, tests in self.results.items():
            for test in tests:
                for farm_result in test['farm_results']:
                    if farm_result['farm_id'] == 0:  # 新規就農者のIDが0であると仮定
                        transfers = sum(farm_result['transfers_truth'].values())
                        transfer_data.append({
                            'rate': rate,
                            'transfers': transfers
                        })
        # DataFrameに変換
        df = pd.DataFrame(transfer_data)

        # プロット
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='rate', y='transfers', data=df)
        plt.title('Tractor Transfers by New Farmer Rate')
        plt.xlabel('New Farmer Rate')
        plt.ylabel('Total Transfers')
        plt.show()

    def _run_all_rates(self):
        """全ての新規就農者率でシミュレーションを実行"""
        for new_farmer_rate in self._generate_rate_range():
            self._run_single_rate(new_farmer_rate)

    def _generate_rate_range(self) -> np.ndarray:
        """新規就農者率の範囲を生成（ステップごと）"""
        return np.arange(self.config.NEW_FARMER_RATE_BEGIN, self.config.NEW_FARMER_RATE_END + self.config.NEW_FARMER_RATE_STEP, self.config.NEW_FARMER_RATE_STEP)
    
    def _run_single_rate(self, new_farmer_rate: float):
        """特定の新規就農者率でのシミュレーション実行"""
        print(f"\n新規就農者率 {new_farmer_rate*100:.1f}% でのシミュレーション")
        self.results[new_farmer_rate] = []
        
        for test in range(self.config.TESTS):
            farms = self._initialize_farms(new_farmer_rate)
            actual_new_rate = self._calculate_actual_rate(farms)

            weather_data = self._generate_weather()
            print(f"テスト {test + 1}/{self.config.TESTS} 実行中...")
            
            # 農家データの生成
            farms = self._create_farms(self.config, new_farmer_rate)
            
            # 農家データの出力
            for farm in farms:
                print(f"農家ID: {farm.id}, 農地面積: {farm.land_area}, 農機台数: {farm.machine_count}, 新規就農者: {farm.is_new}")
            
            # 最適化の実行
            try:
                result = self._optimize_and_record(farms, weather_data, actual_new_rate)
                #result = self._optimize()
                self.results[new_farmer_rate].append(result)
            except Exception as e:
                print(f"最適化エラー: {e}")

    def _initialize_farms(self, new_farmer_rate: float) -> List[Farm]:
        """農家リストを初期化"""
        farms = [Farm(self.config, farm_id=i) for i in range(self.config.FARMERS)]
        
        # 新規就農者の割合に基づいて農家を初期化
        for farm in farms:
            farm.initialize_random(new_farmer_rate)
        
        return farms

    def _calculate_actual_rate(self, farms: List[Farm]) -> float:
        """実際の新規就農者率を計算（出力用）"""
        new_farmers = 0
        for farm in farms:
                if farm.is_new:
                    new_farmers+=1
        
        return new_farmers / len(farms)

    def _generate_weather(self) -> Dict:
        """天候データを生成"""
        weather = Weather(self.config.DAYS)
        return {
            'forecast': weather.generate_forecast(),
            'actual': weather.generate_actual_weather(weather.generate_forecast())
        }

    def _optimize_and_record(self, farms: List[Farm], weather_data: Dict, 
                           actual_rate: float) -> Dict:
        """最適化を実行し結果を記録"""
        Calculate_FARMER0_profit: bool = False
        W = self.weather_pattern.generate_weather_patterns()
        result = self.optimizer.optimize(
            farms, 
            weather_data['forecast'], 
            weather_data['actual'],
            W,
            Calculate_FARMER0_profit
        )
        Calculate_FARMER0_profit = True
        result['actual_new_rate'] = actual_rate
        return result

    def _analyze_results(self):
        """結果の統計分析を実行"""
        for rate, tests in self.results.items():
            if not tests:
                continue
            
            rate_stats = self._calculate_statistics(tests)
            self.summary[rate] = rate_stats

    def _calculate_statistics(self, tests: List[Dict]) -> Dict:
        """テスト結果の統計を計算"""
        values = {
            'objective_present': [test['objective_present'] for test in tests],
            'objective_lie': [test['objective_lie'] for test in tests],
            'Farmer_payments_truth': [test['Farmer_payments_truth'] for test in tests],
            'Farmer_payments_lie': [test['Farmer_payments_lie'] for test in tests],
            'Profit_truth': [test['Profit_truth'] for test in tests],
            'Profit_lie': [test['Profit_lie'] for test in tests],
            'profit_difference_farm0': [test['profit_difference_farm0'] for test in tests],
            'social_surplus_difference': [test['social_surplus_difference'] for test in tests],
            'total_payment_truth': [test['total_payment_truth'] for test in tests],
            'total_payment_lie': [test['total_payment_lie'] for test in tests],
            'how_many_farmer_trasnsport_truth': [test['how_many_farmer_trasnsport_truth'] for test in tests],
            'how_many_farmer_trasnsport_lie': [test['how_many_farmer_trasnsport_lie'] for test in tests],
            'how_many_farmer_land_area_truth': [test['how_many_farmer_land_area_truth'] for test in tests],
            'how_many_farmer_land_area_lie': [test['how_many_farmer_land_area_lie'] for test in tests],
            'weather_pattern': [test['weather_pattern'] for test in tests],
            'farmers_available_schedule': [test['farmers_available_schedule'] for test in tests],
            'not_i': [test['not_i'] for test in tests],
            'remove_i': [test['remove_i'] for test in tests],
        }

        # 農家0の平均利得を計算
        avg_profit_FARMER0 = np.mean([profit[0] for profit in values['Profit_truth']]) if values['Profit_truth'] else 0
        avg_profit_FARMER0_inlier = np.mean([profit[0] for profit in values['Profit_lie']]) if values['Profit_lie'] else 0
        avg_profit_difference_farm0 = np.mean(values['profit_difference_farm0']) if values['profit_difference_farm0'] else 0
        std_profit_difference_farm0 = np.std(values['profit_difference_farm0']) if values['profit_difference_farm0'] else 0

        return {
            'avg_objective_present': np.mean(values['objective_present']) if values['objective_present'] else 0,
            'std_objective_present': np.std(values['objective_present']) if values['objective_present'] else 0,
            'avg_objective_lie': np.mean(values['objective_lie']) if values['objective_lie'] else 0,
            'std_objective_lie': np.std(values['objective_lie']) if values['objective_lie'] else 0,
            'avg_payment_truth': np.mean(values['Farmer_payments_truth']) if values['Farmer_payments_truth'] else 0,
            'std_payment_truth': np.std(values['Farmer_payments_truth']) if values['Farmer_payments_truth'] else 0,
            'avg_payment_lie': np.mean(values['Farmer_payments_lie']) if values['Farmer_payments_lie'] else 0,
            'std_payment_lie': np.std(values['Farmer_payments_lie']) if values['Farmer_payments_lie'] else 0,
            'avg_profit_truth': np.mean(values['Profit_truth']) if values['Profit_truth'] else 0,
            'std_profit_truth': np.std(values['Profit_truth']) if values['Profit_truth'] else 0,
            'avg_profit_lie': np.mean(values['Profit_lie']) if values['Profit_lie'] else 0,
            'std_profit_lie': np.std(values['Profit_lie']) if values['Profit_lie'] else 0,
            'avg_profit_difference_farm0': avg_profit_difference_farm0,
            'std_profit_difference_farm0': std_profit_difference_farm0,
            'avg_social_surplus_difference': np.mean(values['social_surplus_difference']) if values['social_surplus_difference'] else 0,
            'std_social_surplus_difference': np.std(values['social_surplus_difference']) if values['social_surplus_difference'] else 0,
            'avg_total_payment_truth': np.mean(values['total_payment_truth']) if values['total_payment_truth'] else 0,
            'std_total_payment_truth': np.std(values['total_payment_truth']) if values['total_payment_truth'] else 0,
            'avg_total_payment_lie': np.mean(values['total_payment_lie']) if values['total_payment_lie'] else 0,
            'std_total_payment_lie': np.std(values['total_payment_lie']) if values['total_payment_lie'] else 0,

            'avg_profit_FARMER0': avg_profit_FARMER0,  # ここで追加
            'avg_profit_FARMER0_inlier': avg_profit_FARMER0_inlier,
            'avg_how_many_farmer_land_area_truth': np.mean(values['how_many_farmer_land_area_truth']) if values['how_many_farmer_land_area_truth'] else 0,
            'avg_how_many_farmer_land_area_lie': np.mean(values['how_many_farmer_land_area_lie']) if values['how_many_farmer_land_area_lie'] else 0,
            'avg_how_many_farmer_trasnsport_truth': np.mean(values['how_many_farmer_trasnsport_truth']) if values['how_many_farmer_trasnsport_truth'] else 0,
            'avg_how_many_farmer_trasnsport_lie': np.mean(values['how_many_farmer_trasnsport_lie']) if values['how_many_farmer_trasnsport_lie'] else 0,
            'std_how_many_farmer_land_area_truth': np.std(values['how_many_farmer_land_area_truth']) if values['how_many_farmer_land_area_truth'] else 0,
            'std_how_many_farmer_land_area_lie': np.std(values['how_many_farmer_land_area_lie']) if values['how_many_farmer_land_area_lie'] else 0,
            'std_how_many_farmer_trasnsport_truth': np.std(values['how_many_farmer_trasnsport_truth']) if values['how_many_farmer_trasnsport_truth'] else 0,
            'std_how_many_farmer_trasnsport_lie': np.std(values['how_many_farmer_trasnsport_lie']) if values['how_many_farmer_trasnsport_lie'] else 0
        }

    def visualize_results(self):
        """結果の可視化"""
        if not self.summary:
            print("可視化するデータがありません")
            return

        #self._setup_plot_style()
        #self._create_main_plots()
        #self._create_correlation_plot_present()
        #self._create_distribution_plot_present()
        #self._create_difference_plots_present()
        #self._create_comparison_plot()
        #self._create_objective_comparison_plot()
        #self._create_payment_comparison_plot()
        #self._create_farmer0_profit_difference_plot()  # 新しいグラフ
        #self._create_gain_farm0_merit_average_plot()  # 新しいグラフ
        #plt.show()

    def _create_farmer0_profit_difference_plot(self):
        """農家0の利得差異を新規就農者ごとに表示"""
        rates = sorted(self.results.keys())
        profit_diffs = []

        for rate in rates:
            rounded_rate = round(rate, 2)  # 小数点以下2桁に丸める
            if rounded_rate in self.summary:
                profit_diff = self.summary[rounded_rate].get('avg_profit_difference_farm0', 0)
                profit_diffs.append(profit_diff)
            else:
                print(f"Warning: Rate {rounded_rate} が summary に存在しません。デフォルト値 0 を使用します。")
                profit_diffs.append(0)  # デフォルト値を設定

        # プロット
        plt.figure(figsize=(12, 6))
        plt.plot([r * 100 for r in rates], profit_diffs, 'o-', color='blue', label='利得差異')

        # 各ポイントの値を表示
        for i, rate in enumerate(rates):
            plt.text(rate * 100, profit_diffs[i], f'{profit_diffs[i]:.2f}', fontsize=9, ha='center', va='bottom', color='blue')

        plt.title('Relationship between the rate of new farmers and the profit difference', fontsize=14, pad=20)
        plt.xlabel('Rate of Newly Regulated Farmers (%)')
        plt.ylabel('Profit Difference (Truth - False Declaration)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def _create_comparison_plot(self):
        """農家0の真値と虚偽申告時の目的関数の値を比較"""
        rates = np.array([rate * 100 for rate in sorted(self.summary.keys())])
        objective_truth = [self.summary[rate/100]['objective_present'] for rate in sorted(self.summary.keys())]
        objective_lie = [self.summary[rate/100]['objective_lie'] for rate in sorted(self.summary.keys())]

        plt.figure(figsize=(12, 6))
        plt.plot(rates, objective_truth, 'o-', label='Truth Declaration', color='blue')
        plt.plot(rates, objective_lie, 'o-', label='False Declaration', color='red')
        plt.fill_between(rates, 
                         [self.summary[r/100]['avg_objective_present'] - self.summary[r/100]['std_objective_present'] for r in sorted(self.summary.keys())],
                         [self.summary[r/100]['avg_objective_present'] + self.summary[r/100]['std_objective_present'] for r in sorted(self.summary.keys())],
                         alpha=0.2, color='blue')
        plt.fill_between(rates, 
                         [self.summary[r/100]['avg_objective_lie'] - self.summary[r/100]['std_objective_lie'] for r in sorted(self.summary.keys())],
                         [self.summary[r/100]['avg_objective_lie'] + self.summary[r/100]['std_objective_lie'] for r in sorted(self.summary.keys())],
                         alpha=0.2, color='red')
        
        # 各ポイントの値を表示
        for i, rate in enumerate(rates):
            plt.text(rate, objective_truth[i], f'{objective_truth[i]:.2f}', fontsize=9, ha='center', va='bottom', color='blue')
            plt.text(rate, objective_lie[i], f'{objective_lie[i]:.2f}', fontsize=9, ha='center', va='bottom', color='red')

        plt.title('Comparison of Objective Function Values for New Farming Rates', fontsize=14, pad=20)
        plt.xlabel('new farmer rate (%)')
        plt.ylabel('objective function value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    def _create_objective_comparison_plot(self, ax: plt.Axes, rates: np.ndarray, 
                               objectives: List[float]):
        """目的関数の推移をプロット"""
        ax.plot(rates, objectives, 'o-', linewidth=2, markersize=8)
        ax.fill_between(rates, 
                       [self.summary[r/100]['avg_objective_present'] - 
                        self.summary[r/100]['std_objective_present'] for r in sorted(self.summary.keys())],
                       [self.summary[r/100]['avg_objective_present'] + 
                        self.summary[r/100]['std_objective_present'] for r in sorted(self.summary.keys())],
                       alpha=0.2)
        
        ax.set_title('relationship between new farmer rate and objective function', fontsize=14, pad=20)
        ax.set_xlabel('new farmer rate (%)')
        ax.set_ylabel('objective function value')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        best_rate = max(self.summary.keys(), 
                       key=lambda r: self.summary[r]['avg_objective_present'])
        best_value = self.summary[best_rate]['avg_objective_present']
        ax.plot(best_rate * 100, best_value, 'r*', markersize=15, 
                label=f'optimal rate ({best_rate*100:.1f}%)')
        ax.legend()

    def _create_payment_comparison_plot(self):
        """農家0の真値と虚偽申告時の支払額の合計を比較"""
        rates = np.array([rate * 100 for rate in sorted(self.summary.keys())])
        payment_truth = [self.summary[rate/100]['avg_total_payment_truth'] for rate in sorted(self.summary.keys())]
        payment_lie = [self.summary[rate/100]['avg_total_payment_lie'] for rate in sorted(self.summary.keys())]

        plt.figure(figsize=(12, 6))
        plt.plot(rates, payment_truth, 'o-', label='真値申告時', color='green')
        plt.plot(rates, payment_lie, 'o-', label='虚偽申告時', color='orange')
        plt.fill_between(rates, 
                         [self.summary[r/100]['avg_total_payment_truth'] - self.summary[r/100]['std_total_payment_truth'] for r in sorted(self.summary.keys())],
                         [self.summary[r/100]['avg_total_payment_truth'] + self.summary[r/100]['std_total_payment_truth'] for r in sorted(self.summary.keys())],
                         alpha=0.2, color='green')
        plt.fill_between(rates, 
                         [self.summary[r/100]['avg_total_payment_lie'] - self.summary[r/100]['std_total_payment_lie'] for r in sorted(self.summary.keys())],
                         [self.summary[r/100]['avg_total_payment_lie'] + self.summary[r/100]['std_total_payment_lie'] for r in sorted(self.summary.keys())],
                         alpha=0.2, color='orange')
        # 各ポイントの値を表示
        for i, rate in enumerate(rates):
            plt.text(rate, payment_truth[i], f'{payment_truth[i]:.2f}', fontsize=9, ha='center', va='bottom', color='green')
            plt.text(rate, payment_lie[i], f'{payment_lie[i]:.2f}', fontsize=9, ha='center', va='bottom', color='orange')

        plt.title('Comparison of Total Payments to New Farmer Percentage', fontsize=14, pad=20)
        plt.xlabel('Percentage of new farmers (%)')
        plt.ylabel('Total Amount Paid')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    def _setup_plot_style(self):
        """プロットスタイルの設定"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12

    def _create_main_plots(self):
        """主要な指標のプロット作成"""
        rates = np.array([rate * 100 for rate in sorted(self.summary.keys())])
        objectives = [self.summary[rate/100]['objective_present'] for rate in sorted(self.summary.keys())]

        # プロット
        plt.figure(figsize=(12, 6))
        plt.plot(rates, objectives, 'o-', linewidth=2, markersize=8)
        plt.fill_between(rates, 
                         [self.summary[r/100]['avg_objective_present'] - 
                          self.summary[r/100]['std_objective_present'] for r in sorted(self.summary.keys())],
                         [self.summary[r/100]['avg_objective_present'] + 
                          self.summary[r/100]['std_objective_present'] for r in sorted(self.summary.keys())],
                         alpha=0.2)
        
        plt.title('主要指標のプロット', fontsize=14, pad=20)
        plt.xlabel('新規就農者率 (%)')
        plt.ylabel('目的関数値')
        plt.grid(True, linestyle='--', alpha=0.7)

    def _create_correlation_plot_present(self):
        """目的関数と支払額の相関プロット"""
        plt.figure(figsize=(8, 8))
        
        objectives = [stats['avg_objective_present'] for stats in self.summary.values()]
        payments = [stats['avg_payment_truth'] for stats in self.summary.values()]
        rates = [rate * 100 for rate in self.summary.keys()]
        
        plt.scatter(objectives, payments, c=rates, cmap='viridis', s=100)
        plt.colorbar(label='Rate of Newly Regulated Farmers (%)')
        
        plt.title('Correlation between objective function and payment amount', fontsize=14, pad=20)
        plt.xlabel('objective function value')
        plt.ylabel('amount paid')
        plt.grid(True, linestyle='--', alpha=0.7)

    def _create_distribution_plot_present(self):
        """分布プロットを作成"""
        # ここでresultsをDataFrameに変換
        data = []
        for rate, results in self.results.items():
            for result in results:
                data.append({
                    'rate': rate * 100,
                    'objective_present': result['objective_present'],
                    'objective_lie': result['objective_lie']
                })
                #うえにあったやつ'profit_FARMER0': result['profit_FARMER0']
        
        # DataFrameに変換
        df = pd.DataFrame(data)

        # Seabornを使ってボックスプロットを作成
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='rate', y='objective_present', data=df, ax=ax, color='skyblue', label='objective function (truth)')
        sns.boxplot(x='rate', y='objective_lie', data=df, ax=ax, color='salmon', label='objective function (lie)')
        ax.set_title('distribution of objective function value', fontsize=14, pad=20)
        ax.set_xlabel('new farmer rate (%)')
        ax.set_ylabel('objective function value')
        ax.legend()
        plt.show()

    def _create_difference_plots_present(self):
        """利得差異と社会的余剰差異の分布プロットを作成"""
        # 利得差異のプロット
        rates = np.array([rate * 100 for rate in sorted(self.summary.keys())])
        profit_diffs = [self.summary[rate/100]['avg_profit_difference_farm0'] for rate in sorted(self.summary.keys())]
        social_surplus_diffs = [self.summary[rate/100]['avg_social_surplus_difference'] for rate in sorted(self.summary.keys())]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 利得差異のプロット
        ax1.plot(rates, profit_diffs, 'o-', linewidth=2, markersize=8, color='purple')
        ax1.fill_between(rates, 
                        [self.summary[rate/100]['avg_profit_difference_farm0'] - 
                            self.summary[rate/100]['std_profit_difference_farm0'] for rate in sorted(self.summary.keys())],
                        [self.summary[rate/100]['avg_profit_difference_farm0'] + 
                            self.summary[rate/100]['std_profit_difference_farm0'] for rate in sorted(self.summary.keys())],
                        alpha=0.2, color='purple')
        
        ax1.set_title('Relationship between the rate of new farmers and the profit difference', fontsize=14, pad=20)
        ax1.set_xlabel('Rate of Newly Regulated Farmers (%)')
        ax1.set_ylabel('Profit Difference (Truth - False Declaration)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 社会的余剰差異のプロット
        ax2.plot(rates, social_surplus_diffs, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.fill_between(rates, 
                        [self.summary[rate/100]['avg_social_surplus_difference'] - 
                            self.summary[rate/100]['std_social_surplus_difference'] for rate in sorted(self.summary.keys())],
                        [self.summary[rate/100]['avg_social_surplus_difference'] + 
                            self.summary[rate/100]['std_social_surplus_difference'] for rate in sorted(self.summary.keys())],
                        alpha=0.2, color='orange')
        
        ax2.set_title('Relationship between the rate of new farmers and the social surplus difference', fontsize=14, pad=20)
        ax2.set_xlabel('Rate of Newly Regulated Farmers (%)')
        ax2.set_ylabel('Social Surplus Difference (Truth - False Declaration)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()

    def _create_gain_farm0_merit_average_plot(self):
        """gain_farm0_meritの新規就農者あたりの平均値と頻度をプロット"""
        rates = np.array([rate * 100 for rate in sorted(self.results.keys())])
        average_gains = []

        # 各新規就農率に対するgain_farm0_meritの平均値を計算
        for rate, tests in self.results.items():
            gain_farm0_merit = []
            for test in tests:
                gain_farm0_merit.extend(test['Profit_lie'] + test['Profit_lie1'])
            average_gain = np.mean(gain_farm0_merit) if gain_farm0_merit else 0
            average_gains.append(average_gain)

        # プロット
        plt.figure(figsize=(12, 6))
        plt.plot(rates, average_gains, 'o-', color='green', label='Average Gain')
        
        # 各ポイントの値を表示
        for i, rate in enumerate(rates):
            plt.text(rate, average_gains[i], f'{average_gains[i]:.2f}', fontsize=9, ha='center', va='bottom', color='green')

        plt.title('Average Gain of farmer0 for New Farmer Rate', fontsize=14, pad=20)
        plt.xlabel('new farmer rate (%)')
        plt.ylabel('average gain')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    def _print_summary(self):
        """結果サマリーを表示"""
        if not self.summary:
            print("有効な結果がありません")
            return

        self._print_table_header_truth()
        self._print_results_table_truth()
        self._print_optimal_rate_truth()
        self._print_table_header_lie()
        self._print_results_table_lie()
        self._print_optimal_rate_lie()

    #######################################################ここから             
    def _print_table_header_lie(self):
        """結果テーブルのヘッダーを表示"""
        print("\n=== シミュレーション結果 (嘘申告時) ===")
        print("率(%) | 目的関数平均(嘘) | 標準偏差(嘘) | 支払額平均(嘘) | 標準偏差(嘘) | 農家0の利得差異平均 | 農家0の利得差異標準偏差 | 社会的余剰差異平均 | 社会的余剰差異標準偏差 | 農家0の利得 | 農家0の輸送回数平均(嘘) | 農家0の輸送回数標準偏差(嘘)")
        print("-" * 160)
    
    def _print_results_table_lie(self):
        """結果テーブルの内容を表示"""
        for rate, stats in sorted(self.summary.items()):
            print(
                f"{rate*100:5.1f} | {stats['avg_objective_lie']:17.2f} | "
                f"{stats['std_objective_lie']:12.2f} | {stats['avg_payment_lie']:14.2f} | "
                f"{stats['std_payment_lie']:12.2f} | {stats['avg_profit_difference_farm0']:20.2f} | "
                f"{stats['std_profit_difference_farm0']:20.2f} | "
                f"{stats['avg_social_surplus_difference']:25.2f} | "
                f"{stats['std_social_surplus_difference']:25.2f} | "
                f"{stats['avg_profit_FARMER0']:8.1f} | "
                f"{stats['avg_how_many_farmer_trasnsport_lie']:8.1f} | "
                f"{stats['std_how_many_farmer_trasnsport_lie']:8.1f} | "
                f"{stats['avg_how_many_farmer_land_area_truth']:8.1f} | "
                f"{stats['std_how_many_farmer_land_area_truth']:8.1f}"
            )
    
    def _print_table_header_truth(self):
        """結果テーブルのヘッダーを表示"""
        print("\n=== シミュレーション結果 (真値申告時) ===")
        print("率(%) | 目的関数平均(真) | 標準偏差(真) | 支払額平均(真) | 標準偏差(真) | 農家0の利得差異平均 | 農家0の利得差異標準偏差 | 社会的余剰差異平均(真) | 社会的余剰差異標準偏差(真) | 農家0の利得 | 農家0の輸送回数平均(真) | 農家0の輸送回数標準偏差(真)")
        print("-" * 160)

    def _print_results_table_truth(self):
        """結果テーブルの内容を表示"""
        stats = {'avg_profit_FARMER0': 0.0}

        for rate, stats in sorted(self.summary.items()):
            print(
                f"{rate*100:5.1f} | {stats['avg_objective_present']:16.2f} | "
                f"{stats['std_objective_present']:11.2f} | {stats['avg_payment_truth']:13.2f} | "
                f"{stats['std_payment_truth']:11.2f} | {stats['avg_profit_difference_farm0']:20.2f} | "
                f"{stats['std_profit_difference_farm0']:20.2f} | "
                f"{stats['avg_social_surplus_difference']:20.2f} | "
                f"{stats['std_social_surplus_difference']:20.2f} | "
                f"{stats['avg_profit_FARMER0']:8.1f} | "
                f"{stats['avg_how_many_farmer_trasnsport_truth']:8.1f} | "
                f"{stats['std_how_many_farmer_trasnsport_truth']:8.1f} | "
                f"{stats['avg_how_many_farmer_land_area_truth']:8.1f} | "
                f"{stats['std_how_many_farmer_land_area_truth']:8.1f}"
            )

    def _print_optimal_rate_truth(self):
        """すべて真値のとき最適な新規就農者率の情報を表示"""
        best_rate = max(self.summary.keys(), 
                       key=lambda r: self.summary[r]['avg_objective_present'])
        
        print("\n=== 分析結果 (真値申告時) ===")
        print(f"目的関数が最大となる新規就農者率: {best_rate*100:.1f}%")
        print(f"このときの目的関数平均: {self.summary[best_rate]['avg_objective_present']:.2f}")
        print(f"このときの支払額平均: {self.summary[best_rate]['avg_payment_truth']:.2f}")
        print(f"このときの農家0の輸送回数平均: {self.summary[best_rate]['avg_how_many_farmer_trasnsport_truth']:.2f}")
        print(f"このときの農家0の土地面積平均: {self.summary[best_rate]['avg_how_many_farmer_land_area_truth']:.2f}")
    def _print_optimal_rate_lie(self):
        """すべて嘘申告のとき最適な新規就農者率の情報を表示"""
        best_rate = max(self.summary.keys(), 
                       key=lambda r: self.summary[r]['avg_objective_lie'])
        
        print("\n=== 分析結果 (嘘申告時) ===")
        print(f"目的関数が最大となる新規就農者率: {best_rate*100:.1f}%")
        print(f"このときの目的関数平均: {self.summary[best_rate]['avg_objective_lie']:.2f}")
        print(f"このときの支払額平均: {self.summary[best_rate]['avg_payment_lie']:.2f}")
        print(f"このときの農家0の輸送回数平均: {self.summary[best_rate]['avg_how_many_farmer_trasnsport_lie']:.2f}")

    def _create_farms(self, config: SimulationConfig, new_farmer_rate: float) -> List[Farm]:
        """農家リストを作成"""
        total_farmers = config.FARMERS
        new_farmers_count = int(total_farmers * new_farmer_rate)
        
        farms = []
        
        # 既存農家を作成
        for i in range(total_farmers):
            farm = Farm(config, farm_id=i, is_new=False)
            farm.initialize_random(new_farmer_rate)
            farms.append(farm)
        
        # 新規就農者を作成
        for i in range(total_farmers):
            if i == 0:
                continue
            if random.random() < new_farmer_rate:
                farm = Farm(self.config, farm_id= i, is_new=True)
                farm.initialize_random(new_farmer_rate)
                farms[i] = farm

        return farms

    def save_results_to_excel(self):
        """
        シミュレーション結果をExcelファイルに保存します。
        
        条件:
        1. 新規就農率ごとに別ファイルに分ける。
        2. 各ファイルの最上行には指定されたヘッダーを持つシートを作成する。
        """
        output_dir = setup_output_directory()
        
        for rate, tests in self.results.items():
            filename = f"simulation_results_rate_{rate*100:.1f}%.xlsx"
            filepath = output_dir / filename
    
            # テスト詳細のデータ収集
            test_details = []
            for test_num, test in enumerate(tests, start=1):
                for farm in test['farm_results']:
                    test_details.append({
                        'TEST': test_num,
                        '農家id': farm['farm_id'],
                        '所持農機台数': farm.get('machine_count', ''),
                        '農地面積': farm.get('land_area', ''),
                        '売上': farm.get('売上', ''),
                        '支払額': farm.get('支払額', ''),
                        '利得': farm.get('利得', ''),
                        '移動回数': farm.get('移動回数', ''),
                        '開墾面積': farm.get('開墾面積', ''),
                        '作業スケジュール': farm.get('作業スケジュール', ''),
                        '農家0が嘘をついたときの利益': farm.get('農家0が嘘をついたときの利益', ''),
                        '農家0が嘘をついたときの支払額': farm.get('農家0が嘘をついたときの支払額', ''),
                        '農家0が嘘をついたときの利得': farm.get('農家0が嘘をついたときの利得', ''),
                        '農家0が嘘をついたときの移動回数': farm.get('農家0が嘘をついたときの移動回数', ''),
                        '農家0が嘘をついたときの開墾面積': farm.get('農家0が嘘をついたときの開墾面積', ''),
                        '農家0が嘘をついたときの作業スケジュール': farm.get('農家0が嘘をついたときの作業スケジュール', ''),
                        '天気予報': farm.get('天気予報', ''),
                        '農家0が嘘をついたときの農家0の利用可能なスケジュール': farm.get('農家0が嘘をついたときの農家0の利用可能なスケジュール', ''),
                        '全体の社会-i': farm.get('全体の社会-i', ''),
                        '-iの社会': farm.get('-iの社会', '')
                    })
                    #'売上': test['Profit_truth'][farm['farm_id']] if 'Profit_truth' in test and farm['farm_id'] < len(test['Profit_truth']) else '',
                    #    '支払額': test['Farmer_payments_truth'][farm['farm_id']] if 'Farmer_payments_truth' in test and farm['farm_id'] < len(test['Farmer_payments_truth']) else '',
                    #   '利得': test['Profit_truth'][farm['farm_id']] + test['Farmer_payments_truth'][farm['farm_id']] if 'Profit_truth' in test and 'Farmer_payments_truth' in test and farm['farm_id'] < len(test['Profit_truth']) and farm['farm_id'] < len(test['Farmer_payments_truth']) else '',
                    #   '移動回数': test['how_many_farmer_trasnsport_truth'][farm['farm_id']] if 'how_many_farmer_trasnsport_truth' in test and farm['farm_id'] < len(test['how_many_farmer_trasnsport_truth']) else '',
                    #     '開墾面積': sum(farm['capacity'].values()) if 'capacity' in farm else '',
                    #'農家0が嘘をついたときの利益': test['Profit_lie'][farm['farm_id']] if 'Profit_lie' in test and farm['farm_id'] < len(test['Profit_lie']) else '',
                    #'農家0が嘘をついたときの支払額': test['Farmer_payments_lie'][farm['farm_id']] if 'Farmer_payments_lie' in test and farm['farm_id'] < len(test['Farmer_payments_lie']) else '',
                    #'農家0が嘘をついたときの利得': test['Profit_lie'][farm['farm_id']] + test['Farmer_payments_lie'][farm['farm_id']] if 'Profit_lie' in test and 'Farmer_payments_lie' in test and farm['farm_id'] < len(test['Profit_lie']) and farm['farm_id'] < len(test['Farmer_payments_lie']) else '',
                    #'農家0が嘘をついたときの移動回数': test['how_many_farmer_trasnsport_lie'][farm['farm_id']] if 'how_many_farmer_trasnsport_lie' in test and farm['farm_id'] < len(test['how_many_farmer_trasnsport_lie']) else '',
                    #'農家0が嘘をついたときの開墾面積': sum(farm['capacity'].values()) if 'capacity' in farm else ''

            df_details = pd.DataFrame(test_details)
    
            # 目的関数のデータ収集
            objective_data = []
            for test_num, test in enumerate(tests, start=1):
                objective_data.append({
                    'TEST': test_num,
                    '目的関数平均（真）': test.get('objective_present', ''),
                    '目的関数平均（嘘）': test.get('objective_lie', ''),
                    '目的関数（真）-目的関数（嘘）': test.get('objective_present', '') - test.get('objective_lie', '')

                })
            df_objectives = pd.DataFrame(objective_data)
    
            # Excelファイルに書き込む
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # テスト詳細シート
                df_details.to_excel(writer, sheet_name='テスト詳細', index=False)
                # 目的関数シート
                df_objectives.to_excel(writer, sheet_name='目的関数', index=False)
            
            print(f"結果がExcelファイルに保存されました: {filepath}")

    def save_results_to_excel_alternative(self, file_path: str):
        """
        シミュレーション結果をExcelファイルに保存します。
        
        Parameters:
            file_path (str): 保存するExcelファイルのパス。
        """
        # データフレームを格納する辞書
        excel_data = {
            'Parameters': self._get_parameters_df(),
            'Farmers': self._get_farmers_df(),
            'Results': self._get_results_df(),
            'Farmer0_Lie_Results': self._get_farmer0_lie_results_df()
        }

        # Excelライターを使用して複数のシートに書き込む
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in excel_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"シミュレーション結果がExcelファイルに保存されました: {file_path}")

    def _get_parameters_df(self) -> pd.DataFrame:
        """
        シミュレーションパラメーターをデータフレームとして取得します。
        
        Returns:
            pd.DataFrame: シミュレーションパラメーターのデータフレーム。
        """
        params = vars(self.config)
        params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        return params_df

    def _get_farmers_df(self) -> pd.DataFrame:
        """
        各農家のデータをシミュレーションごとにまとめたデータフレームを取得します。
        
        Returns:
            pd.DataFrame: 農家データのデータフレーム。
        """
        records = []
        for rate, tests in self.results.items():
            for test_num, test in enumerate(tests, start=1):
                for farm in test['farm_results']:
                    records.append({
                        'New_Farmer_Rate (%)': rate * 100,
                        'Test_Number': test_num,
                        'Farmer_ID': farm.id,
                        'Land_Area': farm.land_area,
                        'Machine_Count': farm.machine_count,
                        'Is_New_Farmer': farm.is_new,
                        'How_Many_Farmer_Trasnspor_Truth': farm.how_many_farmer_trasnsport_truth,
                        'How_Many_Farmer_Trasnspor_Lie': farm.how_many_farmer_trasnsport_lie,
                        'How_Many_Farmer_Land_Area_Truth': farm.how_many_farmer_land_area_truth,
                        'How_Many_Farmer_Land_Area_Lie': farm.how_many_farmer_land_area_lie,
                        'How_Many_Farmer_Schedule_Truth': farm.how_many_farmer_schedule_truth,
                        'How_Many_Farmer_Schedule_Lie': farm.how_many_farmer_schedule_lie,
                        'Weather_Pattern': farm.weather_pattern,
                        'Farmers_Available_Schedule': farm.farmers_available_schedule,
                        'not_i': farm.not_i,
                        'remove_i': farm.remove_i
                    })
        farmers_df = pd.DataFrame(records)
        return farmers_df

    def _get_results_df(self) -> pd.DataFrame:
        """
        シミュレーションごとの結果をまとめたデータフレームを取得します。
        
        Returns:
            pd.DataFrame: シミュレーション結果のデータフレーム。
        """
        records = []
        for rate, tests in self.results.items():
            for test_num, test in enumerate(tests, start=1):
                records.append({
                    'New_Farmer_Rate (%)': rate * 100,
                    'Test_Number': test_num,
                    'Objective_Function': test.get('objective_present', None),
                    'Total_Payment_Truth': test.get('total_payment_truth', None),
                    'Total_Payment_Lie': test.get('total_payment_lie', None),
                    'Farmer0_Profit_Truth': test.get('Profit_truth', [])[0] if test.get('Profit_truth') else None,
                    'Farmer0_Profit_Lie': test.get('Profit_lie', [])[0] if test.get('Profit_lie') else None,
                    'Farmer0_Profit_Difference': test.get('profit_difference_farm0', None),
                    'Social_Surplus_Difference': test.get('social_surplus_difference', None),
                    'How_Many_Farmer_Trasnsport_Truth': test.get('how_many_farmer_trasnsport_truth', None),
                    'How_Many_Farmer_Trasnsport_Lie': test.get('how_many_farmer_trasnsport_lie', None),
                    'How_Many_Farmer_Land_Area_Truth': test.get('how_many_farmer_land_area_truth', None),
                    'How_Many_Farmer_Land_Area_Lie': test.get('how_many_farmer_land_area_lie', None),
                    'How_Many_Farmer_Schedule_Truth': test.get('how_many_farmer_schedule_truth', None),
                    'How_Many_Farmer_Schedule_Lie': test.get('how_many_farmer_schedule_lie', None),
                    'Weather_Pattern': test.get('weather_pattern', None),
                    'Farmers_Available_Schedule': test.get('farmers_available_schedule', None),
                    '全体の社会-i': test.get('not_i', None),
                    '-iの社会': test.get('remove_i', None)
                })
        results_df = pd.DataFrame(records)
        return results_df

    def _get_farmer0_lie_results_df(self) -> pd.DataFrame:
        """
        農家0が嘘をついたときの各農家の利得と支払い金額をまとめたデータフレームを取得します。
        
        Returns:
            pd.DataFrame: 農家0が嘘をついたときの各農家の利得と支払い金額のデータフレーム。
        """
        records = []
        for rate, tests in self.results.items():
            for test_num, test in enumerate(tests, start=1):
                if 'Profit_lie' in test and 'Profit_lie1' in test:
                    for i, farm_id in enumerate(range(self.config.FARMERS)):
                        profit_lie = test['Profit_lie'][i] if i < len(test['Profit_lie']) else None
                        profit_lie1 = test['Profit_lie1'][i] if i < len(test['Profit_lie1']) else None
                        gain_merit = profit_lie + profit_lie1 if profit_lie is not None and profit_lie1 is not None else None
                        payment_lie = test['Farmer_payments_lie'][i] if 'Farmer_payments_lie' in test and i < len(test['Farmer_payments_lie']) else None
                        records.append({
                            'New_Farmer_Rate (%)': rate * 100,
                            'Test_Number': test_num,
                            'Farmer_ID': farm_id,
                            'Profit_Lie': profit_lie,
                            'Profit_Lie1': profit_lie1,
                            'Gain_Farm0_Merit': gain_merit,
                            'Payment_Lie': payment_lie,
                            'How_Many_Farmer_Trasnsport_Truth': test.get('how_many_farmer_trasnsport_truth', None),
                            'How_Many_Farmer_Trasnsport_Lie': test.get('how_many_farmer_trasnsport_lie', None),
                            'How_Many_Farmer_Land_Area_Truth': test.get('how_many_farmer_land_area_truth', None),
                            'How_Many_Farmer_Land_Area_Lie': test.get('how_many_farmer_land_area_lie', None),
                            'How_Many_Farmer_Schedule_Truth': test.get('how_many_farmer_schedule_truth', None),
                            'How_Many_Farmer_Schedule_Lie': test.get('how_many_farmer_schedule_lie', None),
                            'Weather_Pattern': test.get('weather_pattern', None),
                            'Farmers_Available_Schedule': test.get('farmers_available_schedule', None),
                            '全体の社会-i': test.get('not_i', None),
                            '-iの社会': test.get('remove_i', None)
                        })
        farmer0_lie_df = pd.DataFrame(records)
        return farmer0_lie_df


def setup_output_directory():
    """出力ディレクトリの設定"""
    output_dir = Path('simulation_results')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def main():
    """メイン実行関数"""
    # シード値の読み込み
    TEST_NUMBER = 0
    Seeds_value = pd.read_csv('seeds.csv', header = None).values.tolist()
    try:
        print("=== 農業シミュレーション開始 ===")
        
        # 出力ディレクトリの設定
        output_dir = setup_output_directory()
        print(f"結果の出力先: {output_dir.absolute()}")
        
        # 設定の初期化
        config = SimulationConfig()
        print("\n=== シミュレーション設定 ===")
        print(f"総農家数: {config.FARMERS}")
        print(f"シミュレーション期間: {config.DAYS}日")
        print(f"実験回数: {config.TESTS}")
        print(f"新規就農者率範囲: {config.NEW_FARMER_RATE_MIN*100:.1f}% - {config.NEW_FARMER_RATE_MAX*100:.1f}%")
        print(f"新規就農者率刻み: {config.NEW_FARMER_RATE_STEP*100:.1f}%")
        
        # シミュレーションの実行
        simulation = FarmingSimulation(config)
        seedValue = Seeds_value[0][TEST_NUMBER]
        simulation.run(seedValue)
        TEST_NUMBER += 1
        print("シミュレーション完了")
        
        # 結果をExcelファイルに保存
        try:
            print("Excelファイルの書き込みを開始します")
            # 新規就農率ごとに別ファイルに保存
            simulation.save_results_to_excel()
            print("Excelファイルの書き込みが完了しました")
        except Exception as e:
            print(f"Excelファイルの保存中にエラーが発生しました: {e}")
            import traceback
            print(traceback.format_exc())
        
        print("\n=== シミュレーション完了 ===")
        print(f"結果はExcelファイルとグラフで保存されました")
        
    except Exception as e:
        print("\n=== エラーが発生しました ===")
        print(f"エラー内容: {str(e)}")
        import traceback
        print("\nスタックトレース:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()