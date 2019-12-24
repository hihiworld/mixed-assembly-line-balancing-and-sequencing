'''
@Description: 
@Author: yaologos
@Github: https://github.com/hihiworld
@Date: 2019-05-30 15:14:02
@LastEditors  : yaologos
@LastEditTime : 2019-12-19 15:06:57
'''

import dataloader
from docplex.mp.model import Model

num_tasks, num_stations, num_products, P0, AL, AR, AE, side_task, P, S,Pa,Sa, time, workspace, totalworkspace = dataloader.load_data_MIP(
    task=12,product=3,station=2)

sides = [0,1]
model = Model()
sequences = [i for i in range(1,num_products+1)]
tasks = [i for i in range(1,num_tasks+1)]
stations = [i for i in range(1,num_stations+1)]
products = [i for i in range(1,num_products+1)]

def C_opposite(i):
    if i in AR:
        return AL
    elif i in AL:
        return AR
    else:
        return set()


var_list_X = [(i, j, m, k)
              for i in tasks for j in products for m in stations for k in sides]
var_list_U = [(i, j, h, q)
              for i in tasks for j in products for h in tasks for q in products]
X = model.binary_var_dict(var_list_X, name='X')
Z = model.binary_var_matrix(tasks, tasks, name='Z')
Y = model.binary_var_matrix(sequences, products, name='Y')
C = model.integer_var_cube(tasks, products, stations, name='CompletionTime')
U = model.integer_var_dict(var_list_U, name='U')
A = model.integer_var_matrix(products, stations, name='ArrivalTime', lb= 0)
D = model.integer_var_matrix(products, stations, name='DepartTime', lb= 0)
C_max = model.integer_var(lb=0, name='C_max')

M = 99999
# workspace约束
for m in stations:
    for j in products:
        model.add_constraint(model.sum(model.sum(workspace[j-1][i-1]*X[i,j,m,k] for k in side_task[i-1]) for i in tasks)<=totalworkspace[m-1])

# ct1
for i in tasks:
    for j in products:
        model.add_constraint(model.sum(model.sum(
            X[i, j, m, k] for k in side_task[i-1]) for m in stations) == 1)
# 
for i in tasks:
    for k in sides:
        if k not in side_task[i-1]:
            for j in products:
                for m in stations:
                    model.add_constraint(X[i,j,m,k] == 0)

# ct1-1
for i in tasks:
    if i in AL:
        for j in products:
            model.add_constraint(
                model.sum(X[i, j, m, 0] for m in stations) == 1)

# ct1-2
for i in tasks:
    if i in AR:
        for j in products:
            model.add_constraint(
                model.sum(X[i, j, m, 1] for m in stations) == 1)

# ct2 所有的任务完工时间都要小于最小完工时间
for i in tasks:
    for j in products:
        for m in stations:
            model.add_constraint(C[i, j, m] <= C_max)

# ct3 每个任务的完工时间都要大于其加工时间（可删除，删除后求解时间变慢）
for i in tasks:
    for j in products:
        model.add_constraint(model.sum(C[i, j, m]
                                       for m in stations) >= time[j-1][i-1])

# ct4 没有安排在相同工作站上的完工时间不存在
for i in tasks:
    for j in products:
        for m in stations:
            model.add_constraint(
                C[i, j, m] <= M * model.sum(X[i, j, m, k] for k in side_task[i-1]))

# ct5 所有前置完成后才能进行下一个任务，也就是紧前任务所在工作站标号小于等于紧后任务所在工作站的标号
for i in set(tasks).difference(P0):
    for r in P[i]:
        for j in products:
            model.add_constraint(model.sum(model.sum(g*X[r, j, g, k] for k in side_task[r-1]) for g in stations)
                                 <= model.sum(model.sum(m*X[i, j, m, k] for k in side_task[i-1]) for m in stations))

# ct6 不同产品间task的关系，不能同时加工
for j in products:
    for q in products:
        if j != q:
            for i in tasks:
                for h in tasks:
                    for m in stations:
                        model.add_constraint(C[i, j, m]-C[h, q, m] + M * (1-model.sum(X[i, j, m, k] for k in side_task[i-1]))+M*(1-model.sum(X[h, q, m, k] for k in side_task[h-1])) + M*U[i, j, h, q]
                                                 >= time[j-1][i-1])
# ct6
for j in products:
    for q in products:
        if j != q:
            for i in tasks:
                for h in tasks:
                    for m in stations:
                            model.add_constraint(C[h, q, m]-C[i, j, m] + M * (1-model.sum(X[i, j, m, k] for k in side_task[i-1]))+M*(1-model.sum(X[h, q, m, k] for k in side_task[h-1])) + M*(1-U[i, j, h, q])
                                                 >= time[q-1][h-1])

# ct7 同一产品内的有次序关系的任务约束，要满足其次序关系，无论在哪个工作站的哪个方向
for i in list(set(tasks).difference(set(P0))):
    for r in P[i]:
        for j in products:
            for m in stations:
                model.add_constraint(C[i, j, m]-C[r, j, m] + M*(1-model.sum(X[i, j, m, k] for k in side_task[i-1]))
                                     + M * (1-model.sum(X[r, j, m, k]
                                                      for k in side_task[r-1]))
                                     >= time[j-1][i-1])
## 这部分是任意工作站
# for i in list(set(tasks).difference(set(P0))):
#     for r in P[i]:
#         for j in products:
#             for m in stations:
#                 for n in stations:
#                     model.add_constraint(C[i, j, m]-C[r, j, n] + M*(1-model.sum(X[i, j, m, k] for k in side_task[i-1]))
#                                         + M * (1-model.sum(X[r, j, n, k]
#                                                         for k in side_task[r-1]))
#                                         >= time[j-1][i-1])

# ct8-1 同一产品内的无次序关系的task之间的约束，如果i在h前面加工，满足第一个公式，否则满足下一个公式
for i in tasks:
    r = list(set(tasks).difference(
        set(set(Pa[i]).union(set(Sa[i])).union(C_opposite(i)))))
    for h in r:
        if i < h:
            for m in stations:
                for k in list(set(side_task[i-1]).intersection(set(side_task[h-1]))):
                    for j in products:
                        model.add_constraint(C[h, j, m]-C[i, j, m] + M*(1-X[i, j, m, k])
                                             + M*(1-X[h, j, m, k])
                                             + M*(1-U[i, j, h, j])
                                             >= time[j-1][h-1])
# ct8-2
for i in tasks:
    r = list(set(tasks).difference(
        set(set(Pa[i]).union(set(Sa[i])).union(C_opposite(i)))))
    for h in r:
        if i < h:
            for m in stations:
                for k in list(set(side_task[i-1]).intersection(set(side_task[h-1]))):
                    for j in products:
                        model.add_constraint(C[i, j, m]-C[h, j, m] + M*(1-X[i, j, m, k])
                                             + M*(1-X[h, j, m, k])
                                             + M*U[i, j, h, j]
                                             >= time[j-1][i-1])

# ct9 每个位置只能加工一次产品
for s in sequences:
    model.add_constraint(model.sum(Y[s, j] for j in products) == 1)

# ct10 每个产品只能在序列中出现一次
for j in products:
    model.add_constraint(model.sum(Y[s, j] for s in sequences) == 1)

# ct11 在同一个工作站上，位置在前的产品的完工时间小于位置在后的产品的开始时间
for s in sequences:
    for j in products:
        for q in products:
            if s > 1 and q != j:
                for m in stations:
                    model.add_constraint(
                        D[j, m]-A[q, m] <= M*(2-Y[s-1, j]-Y[s, q]))

# ct12 同一个产品在前一个工作站上的完工时间应小于其在下一工作站上的开始时间
for i in tasks:
    for j in products:
        for m in stations:
            for k in side_task[i-1]:
                model.add_constraint(A[j, m] <= C[i, j, m] - time[j-1][i-1]*X[i, j, m, k]
                                 + M*(1-X[i, j, m, k]))

for i in tasks:
    for j in products:
        for m in stations:
            for i in side_task[i-1]:
                model.add_constraint(A[j, m] <= D[j, m]-
                                     model.max(model.sum(time[j-1][i-1]*X[i, j, m, 0] for i in tasks),
                                               model.sum(time[j-1][i-1]*X[i, j, m, 1] for i in tasks)))
                                               
for m in stations:
    if m > 1:
        for j in products:
            model.add_constraint(A[j,m]>=D[j,m-1])

for j in products:
    for m in stations:
        for i in tasks:
            model.add_constraint(D[j, m] >= C[i, j, m])

for j in products:
    for m in stations:
        model.add_constraint(D[j, m] <= C_max)


obj = model.minimize(C_max)
sol = model.solve()
print(model.solve_details)
print(sol)
