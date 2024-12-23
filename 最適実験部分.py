import numpy as np
import gurobipy as gp
import pandas as pd
import random
import math as mt
import openpyxl as op
import sys 
import math
import csv



TEST = 100                                             #実験の回数

FARMERS = 100                                             #参加農家数
D= 7                                                    #期間日数

#要修正
# パラメータ設定（例: l, k, vなど）
l = 8.9355/3 #1台を移動させるのに必要なコスト20分
k = (6.5+7.1)/2 #開墾した農地を金額に変換する
n = 12000/14#農機1台が1日に耕せる面積

header = ['回数','効用確率','新機就農者割合','虚偽申告農家','虚偽申告台数','真値申告台数','合計支払い金額','期待売上','期待総利益','利益']#書き込みシートの行名
body=[]#書き込み内容


result_value = []
result_shiharai = []

#農地面積の設定
Area_average = 34000                                        #農地面積の平均値
Area_haba = 10000                                             #数値の根拠なし


with open('kekka.csv', 'a') as f:

    writer = csv.writer(f)
    writer.writerow(header)


f.close()




def sifromremove(s,t,list_w_real,list_kouyou,list_nouti,h,day):
    # 最初に二重辞書を正しく初期化
    number_2 = {}
    for idx in range(FARMERS):
        number_2[idx] = {}
        for d in range(D):
            number_2[idx][d] = 0

    # 値を計算
    for idx in range(FARMERS):
        for d in range(D):
            if list_w_real[day] * list_kouyou[idx][day] * s[idx, day].X > list_nouti[idx]:
                number_2[idx][d] = list_nouti[idx]
            else:
                number_2[idx][d] = list_w_real[day] * list_kouyou[idx][day] * s[idx, day].X


    # number_tekitou2の計算
    number_tekitou2 = 0
    for idx in range(FARMERS):
        if idx != h:
            continue
        number_tekitou2 += number_2[idx][day]
    optimal_value = k * number_tekitou2 - gp.quicksum(
        l * t[idx, day].X if idx != h else 0 for idx in range(FARMERS)
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
    for idx in  range(FARMERS):
        for d in range(D):
            t_i[idx, d] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)  # tの名前を修正
            s_i[idx, d] = model_2.addVar(vtype=gp.GRB.INTEGER)
            for w in range(2 ** D):
                    c_i[idx, w] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)

        # 制約条件式 Π_{d∈D}|
    for w in range(2 ** D):
            z[w] = model_2.addVar(vtype=gp.GRB.CONTINUOUS)

        # Wの設定
    W = {}
    for d in range(2 ** D):
        # d を2進数に変換し、先頭にゼロ埋めして D 桁に調整
        w_bin = bin(d)[2:].zfill(D)
        for w_range in range(D):
            # w_range の位置の文字を整数に変換して代入
            W[d, w_range] = int(w_bin[w_range])

    Z = {}
    for i in range(2 ** D):
        for d in range(D):
            Z[i,d] = model_2.addVar(vtype=gp.GRB.INTEGER)

    # 目的関数設定
    objective_expr = k * gp.quicksum(
        c_i[idx, w] * z[w] if idx != h else 0 for idx in range(FARMERS) for w in range(2 ** D)
    ) - gp.quicksum(
        l * t_i[idx, d] if idx != h else 0 for idx in range(FARMERS) for d in range(D)  
    )
    model_2.setObjective(objective_expr, gp.GRB.MAXIMIZE)
    




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
    op = 0
    for idx in range(FARMERS):
        if idx == h:
            continue
        #print("idxの数"+str(idx))
        #print(i.b[idx])
        for w in range(2 ** D):
            model_2.addConstr(c_i[idx, w] <=list_nouti[idx])
            model_2.addConstr(
                c_i[idx, w] <= gp.quicksum(W[w, d] * s_i[idx, d] * list_kouyou[idx][d] * n for d in range(D))
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


        #print("model_2")
    #print(dict_M.values())
    for d in range(D):
        model_2.addConstr(
            gp.quicksum(s_i[idx, d] for idx in range(FARMERS)) == gp.quicksum(list_daisuu[idx] for idx in range(FARMERS))
        )
        # 最適化の実行
        
    print("[Gurobi Optimize2ログ]")
    model_2.optimize()

    number_1 = {}
    for idx  in range(FARMERS):
        number_1[idx] = {}
        for d in range(D):
            if list_w_real[day] * s_i[idx, day].X * list_kouyou[idx][day] > list_nouti[idx]:
                number_1[idx][d]  = list_nouti[idx]
            else:
                number_1[idx][d] = list_w_real[day] * s_i[idx, day].X * list_kouyou[idx][day]
    number_tekitou = 0
    for w in range(2 ** D):
        for idx in range(FARMERS):
            if idx == h:
                number_tekitou += 0
            else:
                number_tekitou += number_1[idx][day]


    optimal_value =k * number_tekitou - gp.quicksum(

        l * t_i[idx, day].X if idx != h else 0 for idx in range(FARMERS)  
    )
    return optimal_value









def lier_simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,liers_truth,lier_FARMER,liers_lie,COUNT,roop_number,new_farmers_rate):
    while True:
                if list_daisuu[lier_FARMER] == liers_truth:#もしliers_truth 台の農機台数の農家がいたらその人を嘘つきにする
                    simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,COUNT,liers_lie,liers_truth,lier_FARMER,roop_number,new_farmers_rate)

                    list_daisuu[lier_FARMER] = liers_truth
                    break
                else:  lier_FARMER = random.randint(0,FARMERS-1)



def shiharai(s,c,t,z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou):#支払い決定関数
    q = 0
    for day in range(D):
        for h in range(FARMERS):
            q += (Removei( z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou,h,day)
                                   -sifromremove(s,t,list_w_real,list_kouyou,list_nouti,h,day)) 
    return q

def random_number(average,haba):                               #農地面積、農機台数のランダム割当関数
    gosa = average-haba
    if gosa<0:
        print("幅を見直してください。平均："+str(average)+"幅："+str(haba))
        sys.exit()
    return gosa + int(random.random()*2*haba)                   #平均からランダムで+-haba以内の値を割り当て


def Kouyou_Random(roop_number,schedule):#各農家のroop_number%で農作業を行う効用のリストを提出
    for d in range(D):
        if random.random() < roop_number*0.01:
            schedule.append(1)
        else:
            schedule.append(0)
        
    return schedule

def Hensuukettei(COUNT):#各配列値の詳細な設定
    
    
    list_kouyou = []
    list_daisuu = []
    list_tenkiyohou = []
    list_w_real = []
    list_nouti = []

    new_farmers_increaserate = 0.3#新規就農者の上昇率
    new_farmers_rate = 0.0-new_farmers_increaserate#新規就農者の割合
    
    while(new_farmers_rate <= 1.0):
        new_farmers_rate += new_farmers_increaserate
        #農地面積のランダム割当
        list_nouti.clear()                                         #リスト農地の設定
        for farm in range(FARMERS):                                             
            list_nouti.append(random_number(Area_average,Area_haba))   #要修正  #農地平均からランダムで平均から+-300以内の値を割り当て
            i = 0
        roop_number =10
        while(roop_number <= 100):
            #農家効用のランダム導出
            
            list_kouyou.clear()                                               # リスト効用の初期化
            for farm in range(FARMERS):
                farm_schedule = []                                            # 各農家のスケジュールを格納するリスト
                list_kouyou.append(Kouyou_Random(roop_number, farm_schedule))                              # 各農家のスケジュールをリストに追加
            roop_number += 20
            i+=1


            #天候の設定
            list_tenkiyohou.clear()
            for d in range(D):
                list_tenkiyohou.append(random.random())


            list_w_real.clear() #0日目から|D|日目まで実際の天気
            for d in range(D):
                if random.random() < list_tenkiyohou[d]:
                    list_w_real.append(0)
                else:
                    list_w_real.append(1)


            #農機台数のランダム導出
            list_daisuu.clear()                                                  #リスト農機台数の設定

            for farm in range(FARMERS):
                if farm <=FARMERS*(1-new_farmers_rate):    # まず全員に1台ずつ配分
                    list_daisuu.append(1)
                else:# 残りの農機をランダムに配分
                    list_daisuu.append(0)

            gap = math.inf
            for farm in range(FARMERS):#すべての農家が同じような農機台数を所持しているときの農機割当
                
                last_used_random = []
                while ((sum(list_daisuu)+1)/len(list_daisuu) -1.33990801659)< gap:
                    t = random.randint(0,(FARMERS-1))
                    gap = (sum(list_daisuu)+1)/len(list_daisuu) -1.3399080165
                    if t in last_used_random:
                        continue
                    else:    
                        last_used_random.append(t)
                        print(f"エラー: list_daisuuのサイズ: {len(list_daisuu)}")
                        print(f"アクセスしようとしたインデックス: {t}")
                        print(f"アクセスしようとした値{list_daisuu[t]}")
                        list_daisuu[t] += 1


            lier_FARMER = random.randint(0,FARMERS-1)

            liers_truth = 1
            liers_lie = 0
            lier_simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,liers_truth,lier_FARMER,liers_lie,COUNT,roop_number,new_farmers_rate)
            

            liers_truth = 2
            liers_lie = 1
            lier_simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,liers_truth,lier_FARMER,liers_lie,COUNT,roop_number,new_farmers_rate)

            liers_truth = 2
            liers_lie = 0
            lier_simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,liers_truth,lier_FARMER,liers_lie,COUNT,roop_number,new_farmers_rate)





            list_daisuu.clear()         
            for farm in range(FARMERS):
                if farm <=FARMERS*(1-new_farmers_rate):
                    list_daisuu.append(1)
                else:
                    list_daisuu.append(0)
            gap = math.inf

            rich_FARMER = 0
            list_rich_FARMER = []
            list_rich_FARMER.append(rich_FARMER)
            while ((sum(list_daisuu)+1)/len(list_daisuu) -1.33990801659)< gap:
                if list_daisuu[rich_FARMER] > 6:
                    rich_FARMER = random.randint(0,FARMERS-1)
                    if rich_FARMER in list_rich_FARMER:
                        continue
                    list_rich_FARMER.append(rich_FARMER)

                list_daisuu[rich_FARMER] += 1
                
                gap = (sum(list_daisuu)+1)/len(list_daisuu) -1.33990801659
            liers_truth = list_daisuu[0]
            while list_daisuu[0] > 0:
                liers_lie = list_daisuu[0]
                simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,COUNT,liers_lie,liers_truth,0,roop_number,new_farmers_rate)
                list_daisuu[0] -= 1


def simulation(list_nouti,list_daisuu,list_kouyou,list_tenkiyohou,list_w_real,COUNT,liers_lie,liers_truth,lier_Farmer,roop_number,new_farmers_rate):
    data = []
    data.append(COUNT)
    data.append(roop_number)
    data.append(new_farmers_rate)
    data.append(lier_Farmer)
    data.append(liers_lie)
    data.append(liers_truth)


    print("Checking input parameters:")
    print(f"list_nouti length: {len(list_nouti)}")
    print(f"list_daisuu length: {len(list_daisuu)}")
    print(f"list_kouyou length: {len(list_kouyou)}")
    print(f"list_tenkiyohou length: {len(list_tenkiyohou)}")
    print(f"list_w_real length: {len(list_w_real)}")

    model_1 = gp.Model(name="Gurobi")
    print("成功")
    # 決定変数の定義
    s = {}  # s_{i,d}
    c = {}  # c_{i,w}
    t = {} #t_{i,w}
    z = {} #z_{w}

    #s_{i,d}とt_{i,d},c_{i,w}の設定
    for idx in range(FARMERS):
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
        c[idx, w] * z[w] for idx in range(FARMERS) for w in range(2 ** D)
    ) - gp.quicksum(
        l * t[idx, d] for idx in range(FARMERS) for d in range(D) 
    )
    model_1.setObjective(objective_expr, gp.GRB.MAXIMIZE)

    # 制約条件式 Π_{d∈D}|w_d-P_d|の設定






    #制約条件の設定

    op = 0
    for idx in range(FARMERS):
        for w in range(2 ** D):
            model_1.addConstr(c[idx, w] <=list_nouti[idx])
            model_1.addConstr(
                c[idx, w] <= gp.quicksum(W[w, d] * s[idx, d] * list_kouyou[idx][d] * n for d in range(D))
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
            gp.quicksum(s[idx, d] for idx in range(FARMERS)) == gp.quicksum(list_daisuu[idx] for idx in range(FARMERS))
        )

    # 最適化の実行
    print(str(COUNT)+"回目"+ "[Gurobi Optimizeログ]")
    model_1.optimize()

    print()
    print("[解]")
    if model_1.Status == gp.GRB.OPTIMAL:
        print("    最適解: ")
        for idx in range(FARMERS):
            for d in range(D):
                    print(f"        農家 {idx} が日 {d} に作業量 {s[idx, d].X}")
        val_opt = model_1.ObjVal
        print(f"    最適値: {val_opt}")
    else:
        print("最適解が見つかりませんでした")

    for idx in range(FARMERS):
        for d in range(D):
            print("ｓ"+str(s[idx,d]))
            print("最適値"+str(s[idx,d].X))
    
    rieki = 0
    for i in range(FARMERS):
        for d in range(D):#獲得利益の計算
            if random.random() < list_w_real[d]:
                if list_daisuu[lier_Farmer] == liers_lie and idx == list_kouyou:
                    rieki += (s[idx,d].X+list_daisuu[lier_Farmer]-liers_lie )*k*n*list_kouyou[i][d]-l*max(0,s[idx,d].X-liers_lie)
                    continue
            rieki += s[idx,d].X*k*n*list_kouyou[i][d]-l*max(0,s[idx,d].X-list_daisuu[idx])
        

    print("農地面積:", list_nouti)
    print("農機台数:", list_daisuu)
    shiharai_value = (shiharai(s,c,t,z,list_w_real,list_kouyou,list_nouti,list_daisuu,list_tenkiyohou))
    #print("process17")
    print("全農家の支払額:"+str(shiharai_value))




    result_value.append(val_opt)
    result_shiharai.append(shiharai_value)
    data.append(val_opt)
    data.append(shiharai_value)
    data.append(val_opt-shiharai_value)
    data.append(rieki)

    body.append(data)

    with open('kekka.csv', 'a') as f:
    
        writer = csv.writer(f)
        writer.writerows(body)
    
    
    f.close()
    for i in range(5):
        print(f"\n")
    print("complete")
    for i in body:
        print(i)
    body.clear()


def main():
    global result_value, result_shiharai
    result_value = []  # リストの初期化
    result_shiharai = []  # リストの初期化
    
    for count in range(TEST):
        Hensuukettei(count)
        print(f"Test {count + 1} completed")
    
    print("\n最終結果:")
    for i in range(len(result_value)):
        print(f"Test {i + 1}:")
        print(f"  最適値: {result_value[i]}")
        print(f"  支払額: {result_shiharai[i]}")
    
    with open('kekka.csv', 'w') as f:
    
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)

    f.close()
    print("complete")


if __name__ == "__main__":
   main()