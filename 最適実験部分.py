import numpy as np
import gurobipy as gp
import pandas as pd
import random
import math as mt
import openpyxl as op
import sys 



TEST = 10                                             #実験の回数

FARMS = 100                                             #参加農家数
D= 7                                                    #予報日数
KOUYOUKAKURITU = 0.6                                    #農家が働く確率
# パラメータ設定（例: l, k, vなど）
l = 1 #1台を移動させるのに必要なコスト
k = 100 #開墾した農地を金額に変換する
v = 1
a = 1000


result_value = []
result_shiharai = []

#農地面積の設定
Area_average = 25000                                        #農地面積の平均値
Area_haba = 300                                             #数値の根拠


#農家の所有台数の設定
Posess_average = 2                                               #台数の平均値
Posess_haba = 2


#農機を所有していない人の割合
not_posession_rate = 0.1








def sifromremove(s,t,list_w_real,list_kouyou,list_nouti,h,day):
    # 最初に二重辞書を正しく初期化
    number_2 = {}
    for idx in range(FARMS):
        number_2[idx] = {}
        for d in range(D):
            number_2[idx][d] = 0

    # 値を計算
    for idx in range(FARMS):
        for d in range(D):
            if list_w_real[day] * list_kouyou[idx][day] * s[idx, day].X > list_nouti[idx]:
                number_2[idx][d] = list_nouti[idx]
            else:
                number_2[idx][d] = list_w_real[day] * list_kouyou[idx][day] * s[idx, day].X


    # number_tekitou2の計算
    number_tekitou2 = 0
    for idx in range(FARMS):
        if idx != h:
            continue
        number_tekitou2 += number_2[idx][day]
    optimal_value = k * number_tekitou2 - gp.quicksum(
        l * t[idx, day].X if idx != h else 0 for idx in range(FARMS)
    )
    return optimal_value

def Removei(z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou,h,day):#農家 h のいない小社会で最大化された余剰
    model_2 = gp.Model(name="Gurobi")
    if h ==-1:
        h = mt.inf()
    t_i = {} #t_{i,w}
    c_i = {} #c_{i,w}
    s_i = {} #s_{i,d}
    z = {} #z_{w}
    for idx in  range(FARMS):
        for d in range(D):
            #print(f"農家{idx}: {i.name}")  # デバッグ用
            t_i[idx, d] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)  # tの名前を修正
            s_i[idx, d] = model_2.addVar(vtype=gp.GRB.INTEGER)
            for w in range(2 ** D):
                
                    c_i[idx, w] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)
    

    print("process1")

        # 制約条件式 Π_{d∈D}|
    for w in range(2 ** D):
            z[w] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)
    print("process2")

        # Wの設定
    W = {}
    for d in range(2 ** D):
        # d を2進数に変換し、先頭にゼロ埋めして D 桁に調整
        w_bin = bin(d)[2:].zfill(D)
        for w_range in range(D):
            # w_range の位置の文字を整数に変換して代入
            W[d, w_range] = int(w_bin[w_range])
            #print (W) #Wの確認
    print("process5")

    Z = {}
    for i in range(2 ** D):
        for d in range(D):
            Z[i,d] = model_2.addVar(vtype=gp.GRB.INTEGER)
    print("process3")

    # 目的関数設定
    objective_expr = k * gp.quicksum(
        c_i[idx, w] * z[w] if idx != h else 0 for idx in range(FARMS) for w in range(2 ** D)
    ) - gp.quicksum(
        l * t_i[idx, d] if idx != h else 0 for idx in range(FARMS) for d in range(D)  
    )
    model_2.setObjective(objective_expr, gp.GRB.MAXIMIZE)
    print("process4")
    




    for w in range(2 ** D):
        prod = 1 
        diff = 1
        for d in range(D):
            if d == 0:continue
            weather_val = list_tenkiyohou[d]# 最初の天気予報データを使用
            diff *= abs(W[w, d] - weather_val)
            if diff == 0:
                prod = 0
                break  
            prod *= diff
        model_2.addConstr(z[w] == prod)
    print("process6")
    op = 0
    for idx in range(FARMS):
        if idx == h:
            continue
        #print("idxの数"+str(idx))
        #print(i.b[idx])
        for w in range(2 ** D):
            model_2.addConstr(c_i[idx, w] <=list_nouti[idx])
            model_2.addConstr(
                c_i[idx, w] <= gp.quicksum(W[w, d] * s_i[idx, d] * list_kouyou[idx][d] * a for d in range(D))
            )
        for d in range(D):
            model_2.addConstr(s_i[idx, d] >= 0)
            model_2.addConstr(t_i[idx, d] >= 0)
            model_2.addConstr(
                t_i[idx, d]>= s_i[idx, d] -list_daisuu[idx]
            )
        op += 1
            
    Z_i = {}
    for i in range(2 ** D):
        for d in range(D):
            Z_i[i,d] = model_2.addVar(vtype=gp.GRB.INTEGER)
    print("process7")


        #print("model_2")
    #print(dict_M.values())
    for d in range(D):
        model_2.addConstr(
            gp.quicksum(s_i[idx, d] for idx in range(FARMS)) == gp.quicksum(list_daisuu[idx] for idx in range(FARMS))
        )
    print("process8")
        # 最適化の実行
        
    print("[Gurobi Optimize2ログ]")
    model_2.optimize()

    number_1 = {}
    for idx  in range(FARMS):
        number_1[idx] = {}
        for d in range(D):
            if list_w_real[day] * s_i[idx, day].X * list_kouyou[idx][day] > list_nouti[idx]:
                number_1[idx][d]  = list_nouti[idx]
            else:
                number_1[idx][d] = list_w_real[day] * s_i[idx, day].X * list_kouyou[idx][day]
    print("process9")
    number_tekitou = 0
    for w in range(2 ** D):
        for idx in range(FARMS):
            if idx == h:
                number_tekitou += 0
            else:
                number_tekitou += number_1[idx][day]  
    print("process10")


    optimal_value =k * number_tekitou - gp.quicksum(

        l * t_i[idx, day].X if idx != h else 0 for idx in range(FARMS)  
    )
    print("process11")
    return optimal_value





    





def shiharai(s,c,t,z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou):#支払い決定関数
    a = 0
    for day in range(D):
        for h in range(FARMS):
            a += list_w_real[day]*(Removei( z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou,h,day)
                                   -sifromremove(s,t,list_w_real,list_kouyou,list_nouti,h,day)) 
    print("process12")
    return a

def random_number(average,haba):                               #農地面積、農機台数のランダム割当関数
    gosa = average-haba
    if gosa<0:
        print("幅を見直してください。平均："+str(average)+"幅："+str(haba))
        sys.exit()
    return gosa + int(random.random()*2*haba)                   #平均からランダムで+-haba以内の値を割り当て


def simulation(COUNT):
    model_1 = gp.Model(name="Gurobi")



    
    #農地面積のランダム割当
    list_nouti = []                                              #リスト農地の設定
    for farm in range(FARMS):                                             
        list_nouti.append(random_number(Area_average,Area_haba))   #要修正  #農地平均からランダムで平均から+-300以内の値を割り当て

    #農機台数のランダム導出
    list_daisuu = []                                                  #リスト農機台数の設定
    for farm in range(FARMS):
        list_daisuu.append(random_number(Posess_average,Posess_haba)) #平均農機台数からランダムで平均から+-2以内の値を割り当て


    #農家効用のランダム導出
    #とりあえず確率６０％(=KOUYOUKAKURITU)で働くものとする(根拠なし)
    list_kouyou = []                                                  # リスト効用の初期化
    for farm in range(FARMS):
        farm_schedule = []                                            # 各農家のスケジュールを格納するリスト
        for d in range(D):
            if random.random() < KOUYOUKAKURITU:
                farm_schedule.append(1)
            else:
                farm_schedule.append(0)
        list_kouyou.append(farm_schedule)                              # 各農家のスケジュールをリストに追加

#    #ランダムで晴か雨か決定する(割合は日本の年間平均降水日数から算出34.19%)
#    list_hareame = []
#    nenkankousuiritsu = 0.01*34.19
#    for d in range(D):
#        if random.random() <nenkankousuiritsu:
#            list_hareame.append( 0)
#        else:
#            list_hareame.append(1)



    #天候の設定
    list_tenkiyohou = [] #0日目から|D|日目まで精度が0.8から0.2までを軸に幅0.17で変わっていく

    for d in range(D):
        list_tenkiyohou.append(random.random())

    
    list_w_real = [] #0日目から|D|日目まで実際の天気
    for d in range(D):
        if random.random() < list_tenkiyohou[d]:
            list_w_real.append(0)
        else:
            list_w_real.append(1)





    # 決定変数の定義
    s = {}  # s_{i,d}
    c = {}  # c_{i,w}
    t = {} #t_{i,w}
    z = {} #z_{w}

    #s_{i,d}とt_{i,d},c_{i,w}の設定
    for idx in range(FARMS):
        for d in range(D):
            t[idx, d] = model_1.addVar(vtype=gp.GRB.CONTINUOUS )  
            s[idx, d] = model_1.addVar(vtype=gp.GRB.INTEGER)
            for w in range(2 ** D):
                c[idx, w] = model_1.addVar(vtype=gp.GRB.CONTINUOUS)

    # ∏_(𝑑∈𝐷)〖|𝑤_𝑑−𝑃_𝑑 |〗の値を格納するz[w]の設定
    for w in range(2 ** D):
        z[w] = model_1.addVar(vtype=gp.GRB.CONTINUOUS)






    # Wの設定
    W = {}
    for d in range(2 ** D):
        # d を2進数に変換し、先頭にゼロ埋めして D 桁に調整
        w_bin = bin(d)[2:].zfill(D)
        for w_range in range(D):
            # w_range の位置の文字を整数に変換して代入
            W[d, w_range] = int(w_bin[w_range])
            #print (W) #Wの確認

    for w in range(2 ** D):
        prod = 1 
        diff = 1
        for d in range(D):
            #if d == 0:continue
            weather_val = list_tenkiyohou[d]# 最初の天気予報データを使用
            diff *= abs(W[w, d] - weather_val)
            if diff == 0:
                continue
            prod *= diff
        model_1.addConstr(z[w] == prod)


    # 目的関数設定
    objective_expr = k * gp.quicksum(
        c[idx, w] * z[w] for idx in range(FARMS) for w in range(2 ** D)
    ) - gp.quicksum(
        l * t[idx, d] for idx in range(FARMS) for d in range(D) 
    )
    model_1.setObjective(objective_expr, gp.GRB.MAXIMIZE)

    # 制約条件式 Π_{d∈D}|w_d-P_d|の設定






    #制約条件の設定

    op = 0
    for idx in range(FARMS):
        
        #print("idxの数"+str(idx))
        #print(i.b[idx])
        for w in range(2 ** D):
            model_1.addConstr(c[idx, w] <=list_nouti[idx])
            model_1.addConstr(
                c[idx, w] <= gp.quicksum(W[w, d] * s[idx, d] * list_kouyou[idx][d] * a for d in range(D))
            )
        for d in range(D):
            model_1.addConstr(s[idx, d] >= 0)
            model_1.addConstr(t[idx, d] >= 0)
            model_1.addConstr(
                t[idx, d]>= s[idx, d] -list_daisuu[idx])
        op += 1
            
    Z = {}
    for i in range(2 ** D):
        for d in range(D):
            Z[i,d] = model_1.addVar(vtype=gp.GRB.INTEGER)

    for d in range(D):
        model_1.addConstr(
            gp.quicksum(s[idx, d] for idx in range(FARMS)) == gp.quicksum(list_daisuu[idx] for idx in range(FARMS))
        )

    # 最適化の実行
    print(str(COUNT)+"回目"+ "[Gurobi Optimizeログ]")
    model_1.optimize()

    print()
    print("[解]")
    if model_1.Status == gp.GRB.OPTIMAL:
        print("    最適解: ")
        for idx in range(FARMS):
            for d in range(D):
                    print(f"        農家 {idx} が日 {d} に作業量 {s[idx, d].X}")
        val_opt = model_1.ObjVal
        print(f"    最適値: {val_opt}")
    else:
        print("最適解が見つかりませんでした")

    for idx in range(FARMS):
        for d in range(D):
            print("ｓ"+str(s[idx,d]))
            print("最適値"+str(s[idx,d].X))
    

    
    

    print("農地面積:", list_nouti)
    print("農機台数:", list_daisuu)
    shiharai_value = (shiharai(s,c,t,z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou))
    print("process17")
    print("全農家の支払額:"+str(shiharai_value))

    
    result_value.append(val_opt)
    result_shiharai.append(shiharai_value)




def main():
    for count in range(TEST):
        simulation(count)
    
    for i in range(TEST):
        print(result_value[i])
        print(result_shiharai[i])

if __name__ == "__main__":
   main()