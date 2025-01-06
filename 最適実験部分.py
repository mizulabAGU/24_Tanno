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

# 日本語フォントの設定
#font_path = '/Library/Fonts/NotoSansCJKjp-Regular.otf'  # 適切なフォントのパスを指定
#font_prop = fm.FontProperties(fname=font_path)

# フォントを設定
#plt.rcParams['font.family'] = font_prop.get_name()
#plt.rcParams['font.sans-serif'] = font_prop.get_name()
#plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示するための設定

# 設定クラス
@dataclass
class SimulationConfig:
    """シミュレーションの設定を保持するクラス"""
    FARMERS: int = 5                                                    # 参加農家の数
    DAYS: int = 7                                                        # 対象期間の日数
    TESTS: int = 1                                                       # シミュレーション実験回数
    COST_MOVE: float = 8.9355/3                                          # 農機1台を移動させるコスト（20分あたり）
    COST_CULTIVATION: float = (6.5+7.1)/2                                # 農地開墾を金額換算した値
    WORK_EFFICIENCY: float = 12000/14                                    # 農機1台が1日に耕せる面積
    AREA_AVERAGE: float = 34000.0                                        # 農地面積の平均値
    NEW_FARMER_AREA_AVERAGE: float = 30000.0                             # 新規就農者の農地面積の平均値
    FARMER_UTILITY: float = 5/7                                          # 農家が事前に申告する効用確率（労働意欲の確率）
    AREA_RANGE: float = 10000.0                                          # 農地面積のばらつき幅
    NEW_FARMER_RATE_STEP: float = 0.2                                    # 新規就農者率のステップ
    UTILITY_START: int = 10                                              # 効用値の初期値
    UTILITY_STEP: int = 20                                               # 効用値のステップ
    UTILITY_MAX: int = 100                                               # 効用値の最大値
    NEW_FARMER_RATE_BEGIN: float = 0.00                                  # 新規就農者分析の初期割合
    NEW_FARMER_RATE_END: float = 0.21                                    # 新規就農者分析の最終割合
    NEW_FARMER_RATE_MIN: float = 0.0                                     # 新規就農者率の最小値
    NEW_FARMER_RATE_RANGE: float = 0.1                                   # 新規就農者の増加率
    NEW_FARMER_RATE_MAX: float = 0.21                                    # 新規就農者率の最大値

# カスタム例外
class OptimizationError(Exception):
    """最適化計算に関連するエラー"""
    print("最適化計算部分でエラー発生")
    pass

# 天候管理クラス
class Weather:
    def __init__(self, days: int):
        self.days = days

    def generate_forecast(self) -> List[float]:
        """天気予報の生成"""
        return [random.random() for _ in range(self.days)]

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
    def __init__(self, config: SimulationConfig, farm_id: int, is_new: bool = False):
        self.config = config
        self.id = farm_id  # 農家のIDを設定
        self.land_area: float = 0
        self.machine_count: int = 0
        self.utility: List[int] = []
        self.is_new: bool = is_new

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
            raise ValueError("Invalid area parameters. Please check NEW_FARMER_AREA_AVERAGE,AREA_RANGE")
        return min_area + random.random() * 2 * self.config.AREA_RANGE
    
    def _generate_newfarmers_machine_count(self) -> int:
        """新規就農者の所持農機の割当（真値）"""
        return 0

    def _generate_land_area(self) -> float:
        """農家の所持面積を割り当て"""######################################################
        min_area = self.config.AREA_AVERAGE - self.config.AREA_RANGE
        if min_area < 0:
            raise ValueError("Invalid area parameters. Please check AREA_AVERAGE,AREA_RANGE")
        return min_area + random.random() * 2 * self.config.AREA_RANGE

    def _generate_machine_count(self, new_farmer_rate: float) -> int:
        """"農家の所持農機の割当（真値）"""
        if self.is_new:
            return 0
        if random.random() < 0.7:
            return 1
        else:
            return 2

        
    def _generate_newfarmers_machine_count(self) -> float:
        """新規就農者の農家の所持面積を割り当て"""
        min_area = self.config.NEW_FARMER_AREA_AVERAGE - self.config.AREA_RANGE
        if min_area < 0:
            raise ValueError("Invalid area parameters. Please check NEW_FARMER_AREA_AVERAGE,AREA_RANGE")
        return min_area + random.random() * 2 * self.config.AREA_RANGE
    
    def _generate_newfarmers_utility(self) -> int:
        """新規就農者の所持農機の割当（真値）"""
        return 0


    def _generate_utility(self) -> List[int]:
        """"各農家の効用を導出"""
        return [1 if random.random() < self.config.FARMER_UTILITY else 0 
                for _ in range(self.config.DAYS)]

class lier_Farm(Farm):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.config = config
        self.land_area: float = 0
        self.machine_count: int = 0
        self.utility: List[int] = []


def create_farms(config: SimulationConfig, new_farmer_rate: float) -> List[Farm]:
    """農家リストを作成"""
    existing_farmers_count = config.FARMERS
    new_farmers_count = int(existing_farmers_count *(1+ new_farmer_rate))

    farms = []

    # 既存農家を作成
    for _ in range(existing_farmers_count):
        farm = Farm(config)
        farm.initialize_random(is_new_farmer=False)
        farms.append(farm)

    # 新規就農者を作成
    for _ in range(new_farmers_count):
        farm = Farm(config)
        farm.initialize_random(is_new_farmer=True)
        farms.append(farm)

    return farms



class FarmingOptimizer:
    """最適化クラス"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.weather = Weather(config.DAYS)
        self.reset_model()

    def reset_model(self):
        """最適化モデルのリセット"""
        self.model = gp.Model("FarmingOptimization")
        self.model.setParam('LogToConsole', 0)
        self.model.setParam('TimeLimit', 30000)
        self.model.setParam('MIPGap', 0.1)

    def optimize(self, farms: List[Farm], weather_forecast: List[float], 
                actual_weather: List[int], W) -> Dict:
        """最適化の実行"""
        try:
            self.reset_model()
            vars = self.create_variables()#変数の作成
            self.set_constraints(vars, farms, weather_forecast)#制約の設定
            self.set_objective(vars, farms)#目的関数の設定
            
            self.model.optimize()#最適化の実行
            
            if self.model.Status == gp.GRB.OPTIMAL:#最適解が得られた場合、get_resultsメソッドを呼び出して解を生成し、返す
                return self.get_results(vars, farms, actual_weather,weather_forecast, W)
            elif self.model.Status == gp.GRB.TIME_LIMIT:
                print(f"警告: 時間制限到達 (Gap: {self.model.MIPGap})")#時間制限に到達した場合、現時点での最良解をget_resultsメソッドを通じて返す
                return self.get_results(vars, farms, actual_weather,weather_forecast, W)
            else:#モデルが異常終了した場合は、OptimizationErrorをスローする
                raise OptimizationError(f"最適化失敗 (Status: {self.model.Status})")
        except Exception as e:#何らかの例外が発生した場合、エラーメッセージを表示し、OptimizationErrorをスローする
            print(f"最適化エラー: {str(e)}")
            raise OptimizationError(str(e))
        
    def create_variables(self) -> Dict:
        """最適化変数の作成"""
        vars = {
            's': {},  # 作業量変数
            'c': {},  # 容量変数
            't': {},  # 移動変数
            'z': {}   # 天候変数
        }
        
        for i in range(self.config.FARMERS):
            for d in range(self.config.DAYS):
                vars['s'][i, d] = self.model.addVar(vtype=gp.GRB.INTEGER)
                vars['t'][i, d] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)
                
            for w in range(2 ** self.config.DAYS):
                vars['c'][i, w] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)

        for w in range(2 ** self.config.DAYS):
            vars['z'][w] = self.model.addVar(vtype=gp.GRB.CONTINUOUS)

        #print("vars:", vars)  # varsの内容を出力

        return vars

    def set_constraints(self, vars: Dict, farms: List[Farm], weather_forecast: List[float]):
        """制約条件の設定"""
        W = self.weather.generate_weather_patterns()
        
        # 天候制約＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
        for w in range(2 ** self.config.DAYS):
            prod = self._calculate_weather_probability(w, weather_forecast, W)
            self.model.addConstr(vars['z'][w] == prod)

        # 農家ごとの制約
        for i in range(self.config.FARMERS):
            # 容量制約
            for w in range(2 ** self.config.DAYS):
                self.model.addConstr(vars['c'][i, w] <= farms[i].land_area)#c_{i,w}<=a_i
                self.model.addConstr(
                    vars['c'][i, w] <= gp.quicksum(
                        W[w, d] * vars['s'][i, d] * farms[i].utility[d] * 
                        self.config.WORK_EFFICIENCY for d in range(self.config.DAYS)#c_{i,w}<=Σw_d*s_{i,d}*n*e_{i,d}
                    )
                )
            
            # 作業量と移動の制約
            for d in range(self.config.DAYS):
                self.model.addConstr(vars['s'][i, d] >= 0)#s_{i,d}>=0
                self.model.addConstr(vars['t'][i, d] >= 0)#t_{i,d}>=0
                self.model.addConstr(
                    vars['t'][i, d] >= vars['s'][i, d] - farms[i].machine_count#t_{i,d}>= s_{i,d}-m_i
                )

        # 総農機具数制約
        for d in range(self.config.DAYS):
            self.model.addConstr(
                gp.quicksum(vars['s'][i, d] for i in range(self.config.FARMERS)) 
                == sum(farm.machine_count for farm in farms)#Σs_{i,d}=Σm_i
            )

    def set_objective(self, vars: Dict, farms: List[Farm]):
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
            'total_payment': self._calculate_payment(vars, farms, actual_weather,weather_forecast, W)
        }

        for i in range(self.config.FARMERS):
            farm_result = {
                'farm_id': i,
                'work_schedule': {d: vars['s'][i, d].X for d in range(self.config.DAYS)},
                'transfers': {d: vars['t'][i, d].X for d in range(self.config.DAYS)},
                'capacity': {w: vars['c'][i, w].X for w in range(2 ** self.config.DAYS)}
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

    #def _calculate_payment(self, vars: Dict, farms: List[Farm], actual_weather: List[int]) -> float:
    #    """支払額の計算"""
    #    total_payment = 0
    #    for i in range(self.config.FARMERS):
    #        farm_payment = 0
    #        for d in range(self.config.DAYS):
    #            if actual_weather[d] == 1:
    #                work_done = vars['s'][i, d].X * farms[i].utility[d]
    #                farm_payment += self.config.COST_CULTIVATION * min(
    #                    work_done,
    #                    farms[i].land_area
    #                )
    #                farm_payment -= self.config.COST_MOVE * vars['t'][i, d].X
    #        total_payment += farm_payment
    #    return total_payment

    def _optimize(self, farms: List[Farm]) -> Dict:
        """最適化の実行"""
        model = gp.Model("FarmOptimization")
        
        # 変数の定義
        machine_vars = model.addVars(len(farms), vtype=gp.GRB.INTEGER, name="machines")
        
        # 目的関数の設定
        model.setObjective(gp.quicksum(machine_vars[i] for i in range(len(farms))), gp.GRB.MAXIMIZE)
        
        # 制約条件の設定
        for i, farm in enumerate(farms):
            model.addConstr(machine_vars[i] <= farm.land_area / self.config.WORK_EFFICIENCY, 
                            name=f"machine_limit_{i}")
        
        # 最適化の実行
        model.optimize()
        if model.status == gp.GRB.OPTIMAL:
            print("モデルは最適化されました。")
        elif model.status == gp.GRB.INFEASIBLE:
            print("モデルは実行不可能です。")
        elif model.status == gp.GRB.UNBOUNDED:
            print("モデルは非有界です。")
        elif model.status == gp.GRB.INTERRUPTED:
            print("最適化が中断されました。")
        else:
            print(f"モデルのステータス: {model.status}")

        
        # 結果の取得
        if model.status == gp.GRB.OPTIMAL:
            total_machines = sum(machine_vars[i].X for i in range(len(farms)))
            total_payment = total_machines * self.config.COST_MOVE
            return {
                'objective_value': total_machines,
                'total_payment': total_payment
            }
        else:
            raise OptimizationError("最適化に失敗しました。")

    def _calculate_payment(self, vars: Dict, farms: List[Farm], actual_weather: List[int], weather_forecast, W) -> float:
        """支払額の計算"""
        vcg_payment = 0  # ここで初期化

        total_payment = 0
        total_utility = [0] * self.config.FARMERS  # 各農家の効用を格納するリスト

        # 各農家の効用を計算
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
            total_utility[i] = farm_payment  # 各農家の効用を記録

        # 各農家のVCG支払額を計算
        for h in range(self.config.FARMERS):
            # 他の農家がいない場合の効用
            utility_without_i = sum(total_utility[j] for j in range(self.config.FARMERS) if j != i)
            
            # 現在の農家の効用
            current_utility = total_utility[i]
            

            # VCG支払額の計算
            vcg_payment +=self._calculration_removei(h,vars,farms) 
        #vcg_payment -= self.Removei(weather_forecast,farms, W)
        return vcg_payment
    
    def _calculration_removei(self,h,vars:Dict,farms: List[Farm])->float:
        """農家hを除いた目的関数値の計算"""
        total = 0
        # 農家iがいないときの各農家の効用を計算
        
        total += self.config.COST_CULTIVATION * gp.quicksum(vars['c'][i, w] if i != h else 0 * vars['z'][w] 
                    for i in range(self.config.FARMERS) 
                    for w in range(2 ** self.config.DAYS))
        - self.config.COST_MOVE * gp.quicksum(vars['t'][i, d] if i != h else 0 
                    for i in range(self.config.FARMERS) 
                    for d in range(self.config.DAYS))
            
        return total
    
    #def Removei(self,z, list_w_real, list_kouyou, list_nouti, list_daisuu, list_tenkiyohou, h, day):
    #    model_2 = gp.Model(name="Gurobi")
    #    if h == -1:
    #        h = mt.inf()
        
        t_i = {}
        c_i = {}
        s_i = {}
        z_vars = {}
        FARMERS = self.config.FARMERS
        D = self.config.DAYS
        
        for idx in range(FARMERS):
            for d in range(D):
                t_i[idx, d] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)
                s_i[idx, d] = model_2.addVar(vtype=gp.GRB.INTEGER)
                for w in range(2 ** D):
                    c_i[idx, w] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)

        for w in range(2 ** D):
            z_vars[w] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)

        # Wの設定
        W = {}
        for d in range(2 ** D):
            w_bin = bin(d)[2:].zfill(D)
            for w_range in range(D):
                W[d, w_range] = int(w_bin[w_range])

        # 目的関数設定
        objective_expr = k * gp.quicksum(
            c_i[idx, w] * z_vars[w] if idx != h else 0 for idx in range(FARMERS) for w in range(2 ** D)
        ) - gp.quicksum(
            l * t_i[idx, d] if idx != h else 0 for idx in range(FARMERS) for d in range(D)  
        )
        model_2.setObjective(objective_expr, gp.GRB.MAXIMIZE)

        # 制約条件の設定
        for w in range(2 ** D):
            prod = 1 
            diff = 1
            for d in range(D):
                if d == 0: continue
                weather_val = list_tenkiyohou[d]
                diff *= abs(W[w, d] - weather_val)
                if diff == 0:
                    prod = 0
                    break  
            model_2.addConstr(z_vars[w] == prod)

        for idx in range(FARMERS):
            if idx == h:
                continue
            for w in range(2 ** D):
                model_2.addConstr(c_i[idx, w] <= list_nouti[idx])
                model_2.addConstr(
                    c_i[idx, w] <= gp.quicksum(W[w, d] * s_i[idx, d] * list_kouyou[idx][d] * n for d in range(D))
                )
            for d in range(D):
                model_2.addConstr(s_i[idx, d] >= 0)
                model_2.addConstr(t_i[idx, d] >= 0)
                model_2.addConstr(
                    t_i[idx, d] >= s_i[idx, d] - list_daisuu[idx]
                )

        for d in range(D):
            model_2.addConstr(
                gp.quicksum(s_i[idx, d] for idx in range(FARMERS)) == gp.quicksum(list_daisuu[idx] for idx in range(FARMERS))
            )

        # 最適化の実行
        model_2.optimize()

        # 結果の取得
        number_1 = {}
        for idx in range(FARMERS):
            number_1[idx] = {}
            for d in range(D):
                if list_w_real[day] * s_i[idx, day].X * list_kouyou[idx][day] > list_nouti[idx]:
                    number_1[idx][d] = list_nouti[idx]
                else:
                    number_1[idx][d] = list_w_real[day] * s_i[idx, day].X * list_kouyou[idx][day]

        number_tekitou = 0
        for w in range(2 ** D):
            for idx in range(FARMERS):
                if idx == h:
                    continue
                number_tekitou += number_1[idx][day]

        optimal_value = k * number_tekitou - gp.quicksum(
            l * t_i[idx, day].X if idx != h else 0 for idx in range(FARMERS)  
        )
        return optimal_value

#    def Removei(self, farms: List[Farm], h: int, weather_data) -> float:
#        """農家hが存在しない場合の目的関数の値を計算"""
#        h_machine_count = farms[h].machine_count
#        h_land_area = farms[h].land_area
#        #農家hを存在しないものとする(所持農機台数と所持農地面積を０にする)
#        farms[h].machine_count = 0 
#        farms[h].land_area = 0
#        # 農家hの状態を確認
#        actual_new_rate = 0
#        try:
#                result = self._optimize_and_record(farms, weather_data, actual_new_rate)
#                #result = self._optimize()
#                farms[h].machine_count = h_machine_count
#                farms[h].land_area = h_land_area
#                return result['objective_value']
#        except Exception as e:
#                print(f"最適化エラー: {e}")
        


    

# シミュレーション実行クラス
class FarmingSimulation:
    """農業シミュレーションを実行するクラス"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.optimizer = FarmingOptimizer(config)
        self.weather_pattern = Weather(self.config.DAYS)
        self.results = {}  # 新規就農者率ごとの結果を保存
        self.summary = {}  # 結果の統計情報

    def run(self):
        """シミュレーション全体を実行"""
        self._run_all_rates()
        self._analyze_results()
        self._print_summary()
        self.visualize_results()  # 可視化を追加
    def _run_all_rates(self):
        """全ての新規就農者率でシミュレーションを実行"""
        for new_farmer_rate in self._generate_rate_range():
            self._run_single_rate(new_farmer_rate)

    def _generate_rate_range(self) -> np.ndarray:
        """新規就農者率の範囲を生成（5%刻み）"""
        return np.arange(self.config.NEW_FARMER_RATE_BEGIN, self.config.NEW_FARMER_RATE_END, self.config.NEW_FARMER_RATE_RANGE)
    
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
            farms = self._create_farms(new_farmer_rate)
            
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

    #def _run_test(self, new_farmer_rate: float):
    #    """単一テストを実行"""
    #    farms = self._initialize_farms(new_farmer_rate)
    #    actual_new_rate = self._calculate_actual_rate(farms)
        
    #    weather_data = self._generate_weather()
        
    #    try:
    #        result = self._optimize_and_record(farms, weather_data, actual_new_rate)
    #        self.results[new_farmer_rate].append(result)
    #    except OptimizationError as e:
    #        print(f"最適化エラー: {e}")

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
            
        #new_farmers = sum(1 for farm in farms if farm.machine_count == 0)
        return new_farmers / len(farms)

    def _generate_weather(self) -> Dict:
        """天候データを生成"""
        weather = Weather(self.config.DAYS)
        return {
            'forecast': weather.generate_forecast(),
            'actual': weather.generate_actual_weather(weather.generate_forecast())
        }

    def _optimize_and_record(self, farms: List[Farm], weather_data: Dict, 
                           actual_rate: float) -> Dict:########################################################
        """最適化を実行し結果を記録"""
        W = self.weather_pattern.generate_weather_patterns()
        result = self.optimizer.optimize(
            farms, 
            weather_data['forecast'], 
            weather_data['actual'],
            W
        )
        result['actual_new_rate'] = actual_rate
        return result

    def _analyze_results(self):
        """結果の統計分析を実行"""
        for rate, tests in self.results.items():
            if not tests:
                continue
            
            rate_stats = self._calculate_statistics(tests)
            self.summary[rate] = rate_stats

    #def _calculate_statistics(self, tests: List[Dict]) -> Dict:
    #    """テスト結果の統計を計算"""
    #    values = {
    #        'objective_values': [test['objective_value'] for test in tests],
    #        'payments': [test['total_payment'] for test in tests],
    #        'actual_rates': [test['actual_new_rate'] for test in tests]
    #    }
    #    return {
    #        'avg_objective': np.mean(values['objective_values']),
    #        'std_objective': np.std(values['objective_values']),
    #        'avg_payment': np.mean(values['payments']),
    #        'std_payment': np.std(values['payments']),
    ##        'actual_rate': np.mean(values['actual_rates'])
     #   }
    def _calculate_statistics(self, tests: List[Dict]) -> Dict:
        """テスト結果の統計を計算"""
        values = {
            'objective_values': [
                test['objective_value'].getValue() if isinstance(test['objective_value'], gp.LinExpr) else test['objective_value']
                for test in tests
            ],
            'payments': [
                test['total_payment'].x if isinstance(test['total_payment'], gp.Var) and hasattr(test['total_payment'], 'x') else
                (test['total_payment'].getValue() if isinstance(test['total_payment'], gp.LinExpr) else test['total_payment'])
                for test in tests
            ],
            'actual_rates': [test['actual_new_rate'] for test in tests]
        }
        return {
            'avg_objective': np.mean(values['objective_values']),
            'std_objective': np.std(values['objective_values']),
            'avg_payment': np.mean(values['payments']),
            'std_payment': np.std(values['payments']),
            'actual_rate': np.mean(values['actual_rates'])
        }


#    def _calculate_statistics(self, tests: List[Dict]) -> Dict:
#        """テスト結果の統計を計算"""
#        
#
        values = {
            'objective_values': [test['objective_value'].getValue() if isinstance(test['objective_value'], gp.LinExpr) else test['objective_value'] for test in tests],
            'payments': [test['total_payment'].getValue() if isinstance(test['total_payment'], gp.LinExpr) else test['total_payment'] for test in tests],
            'actual_rates': [test['actual_new_rate'] for test in tests]
        }
        return {
            'avg_objective': np.mean(values['objective_values']),
            'std_objective': np.std(values['objective_values']),
            'avg_payment': np.mean(values['payments']),
            'std_payment': np.std(values['payments']),
            'actual_rate': np.mean(values['actual_rates'])
        }


    def visualize_results(self):
        """結果の可視化"""
        if not self.summary:
            print("可視化するデータがありません")
            return

        self._setup_plot_style()
        self._create_main_plots()
        self._create_correlation_plot()
        self._create_distribution_plot()
        plt.show()

    def _setup_plot_style(self):
        """プロットスタイルの設定"""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12

    def _create_main_plots(self):
        """主要な指標のプロット作成"""
        rates = np.array([rate * 100 for rate in sorted(self.summary.keys())])
        objectives = [self.summary[rate/100]['avg_objective'] for rate in rates]
        payments = [self.summary[rate/100]['avg_payment'] for rate in rates]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        self._plot_objective_function(ax1, rates, objectives)
        self._plot_payments(ax2, rates, payments)
        
        plt.tight_layout()

    def _plot_objective_function(self, ax: plt.Axes, rates: np.ndarray, 
                               objectives: List[float]):
        """目的関数の推移をプロット"""
        ax.plot(rates, objectives, 'o-', linewidth=2, markersize=8)
        ax.fill_between(rates, 
                       [self.summary[r/100]['avg_objective'] - 
                        self.summary[r/100]['std_objective'] for r in rates],
                       [self.summary[r/100]['avg_objective'] + 
                        self.summary[r/100]['std_objective'] for r in rates],
                       alpha=0.2)
        
        ax.set_title('Relationship between the rate of new farmers and the objective function', fontsize=14, pad=20)
        ax.set_xlabel('Rate of Newly Regulated Farmers (%)')
        ax.set_ylabel('objective function value')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        best_rate = max(self.summary.keys(), 
                       key=lambda r: self.summary[r]['avg_objective'])
        best_value = self.summary[best_rate]['avg_objective']
        ax.plot(best_rate * 100, best_value, 'r*', markersize=15, 
                label=f'最適値 ({best_rate*100:.1f}%)')
        ax.legend()
    def _plot_payments(self, ax: plt.Axes, rates: np.ndarray, 
                      payments: List[float]):
        """支払額の推移をプロット"""
        ax.plot(rates, payments, 'o-', linewidth=2, markersize=8, color='green')
        ax.fill_between(rates,
                       [self.summary[r/100]['avg_payment'] - 
                        self.summary[r/100]['std_payment'] for r in rates],
                       [self.summary[r/100]['avg_payment'] + 
                        self.summary[r/100]['std_payment'] for r in rates],
                       alpha=0.2, color='green')
        
        ax.set_title('Relationship between the percentage of new farmers and the amount of payments', fontsize=14, pad=20)
        ax.set_xlabel('Percentage of new farmers (%)')
        ax.set_ylabel('amount paid')
        ax.grid(True, linestyle='--', alpha=0.7)

    def _create_correlation_plot(self):
        """目的関数と支払額の相関プロット"""
        plt.figure(figsize=(8, 8))
        
        objectives = [stats['avg_objective'] for stats in self.summary.values()]
        payments = [stats['avg_payment'] for stats in self.summary.values()]
        rates = [rate * 100 for rate in self.summary.keys()]
        
        plt.scatter(objectives, payments, c=rates, cmap='viridis', s=100)
        plt.colorbar(label='Rate of Newly Regulated Farmers (%)')
        
        plt.title('Correlation between objective function and payment amount', fontsize=14, pad=20)
        plt.xlabel('objective function value')
        plt.ylabel('amount paid')
        plt.grid(True, linestyle='--', alpha=0.7)

    def _create_distribution_plot(self):
        """分布プロットを作成"""
        # ここでresultsをDataFrameに変換
        data = []
        for rate, results in self.results.items():
            for result in results:
                data.append({
                    'rate': rate,
                    'objective': result['objective_value'],
                    'payment': result['total_payment']
                })
        
        # DataFrameに変換
        df = pd.DataFrame(data)

        # Seabornを使ってボックスプロットを作成
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='rate', y='objective', data=df, ax=ax1)
        ax1.set_title('Objective Value Distribution by New Farmer Rate')
        plt.show()

    def _print_summary(self):
        """結果サマリーを表示"""
        if not self.summary:
            print("有効な結果がありません")
            return

        self._print_table_header()
        self._print_results_table()
        self._print_optimal_rate()

    def _print_table_header(self):
        """結果テーブルのヘッダーを表示"""
        print("\n=== シミュレーション結果 ===")
        print("率(%) | 目的関数平均 | 標準偏差 | 支払額平均 | 標準偏差 | 実際の新規率(%)")
        print("-" * 75)

    def _print_results_table(self):
        """結果テーブルの内容を表示"""
        for rate, stats in sorted(self.summary.items()):
            print(
                f"{rate*100:5.1f} | {stats['avg_objective']:11.2f} | "
                f"{stats['std_objective']:8.2f} | {stats['avg_payment']:9.2f} | "
                f"{stats['std_payment']:8.2f} | {stats['actual_rate']*100:8.1f}"
            )

    def _print_optimal_rate(self):
        """最適な新規就農者率の情報を表示"""
        best_rate = max(self.summary.keys(), 
                       key=lambda r: self.summary[r]['avg_objective'])
        
        print("\n=== 分析結果 ===")
        print(f"目的関数が最大となる新規就農者率: {best_rate*100:.1f}%")
        print(f"このときの目的関数平均: {self.summary[best_rate]['avg_objective']:.2f}")
        print(f"このときの支払額平均: {self.summary[best_rate]['avg_payment']:.2f}")

    def _create_farms(self, new_farmer_rate: float) -> List[Farm]:
        """農家リストを作成"""
        total_farmers = self.config.FARMERS
        new_farmers_count = int(total_farmers * new_farmer_rate)
        
        farms = []
        
        # 既存農家を作成
        for i in range(total_farmers):
            farm = Farm(self.config, farm_id=i, is_new=False)
            farm.initialize_random(new_farmer_rate)
            farms.append(farm)

        # 新規就農者を作成
        for i in range(new_farmers_count):
            farm = Farm(self.config, farm_id=total_farmers + i, is_new=True)
            farm.initialize_random(new_farmer_rate)
            farms.append(farm)

        return farms

    
def setup_output_directory():
    """出力ディレクトリの設定"""
    output_dir = Path('simulation_results')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def main():
    """メイン実行関数"""
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
        simulation.run()
        
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