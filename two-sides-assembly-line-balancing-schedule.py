```
参考文献：A mathematical model and a genetic algorithm for two-sided assembly line balancing
doi：10.1016/j.cor.2007.11.003
```
#任务集与加工时间
P0 = (1,2,3)
AL = (1,4,6)
AR = (2,8,12)
AE = (3,5,7,9,10,11)
time = (2,3,2,3,1,1,3,3,2,2,2,1)

#次序关系
P = {1:set(),2:set(),3:set(),4:[1],5:[2],6:[3],7:[4,5],8:[5],9:[5,6],10:[7,8],11:[9],12:[11]}
S = {1:[4],2:[5],3:[6],4:[7],5:[7,8,9],6:[9],7:[10],8:[10],9:[11],10:set(),11:[12],12:set()}
Pa = {1:set(),2:set(),3:set(),4:[1],5:[2],6:[3],7:[1,4,2,5],8:[2,5],9:[2,5,3,6],10:[1,2,4,5,7,8],11:[2,5,3,6,9],12:[2,5,3,6,9,11]}
Sa = {1:[4,7,10],2:[5,7,8,9,10,11,12],3:[6,9,11,12],4:[7,10],5:[7,8,9,10,11,12],6:[9,11,12],7:[10],8:[10],9:[11,12],10:set(),11:[12],12:set()}

def C(i):
    if i in AR:
        return AL
    elif i in AL:
        return AR
    else:
        return set()

side_task = [[0],[1],[0,1],[0],[0,1],[0],[0,1],[1],[0,1],[0,1],[0,1],[1]]
tasks = (1,2,3,4,5,6,7,8,9,10,11,12)
stations = (1,2,3)
sides = (0,1)

from docplex.mp.model import Model
model= Model()

x = model.binary_var_cube(tasks, stations, sides,name='x')
z = model.binary_var_matrix(tasks, tasks,name='z')
t_f = model.integer_var_dict(tasks,name='finish_time')
ct = model.integer_var(lb=0,name='ct')
u = 99999

# ct2
for i in tasks:
    model.add_constraint(model.sum(model.sum(x[i,j,k] for k in side_task[i-1]) for j in stations) == 1)

# ct3
for i in set(tasks).difference(P0):
    for h in P[i]:
        model.add_constraint(model.sum(model.sum(g*x[h,g,k] for k in side_task[h-1]) for g in stations)
                             <= model.sum(model.sum(j*x[i,j,k] for k in side_task[i-1]) for j in stations))
# ct4
for i in tasks:
    model.add_constraint(t_f[i] <= ct)

# ct5
for i in set(tasks).difference(P0):
    for h in P[i]:
        for j in stations:
            model.add_constraint(t_f[i]-t_f[h]+u*(1-model.sum(x[h,j,k] for k in side_task[h-1]))
                                                  +u*(1-model.sum(x[i,j,k] for k in side_task[i-1]))
                                 >= time[i-1])

# ct6
for i in tasks:
    r = list(set(tasks).difference(set(Pa[i]).union(set(Sa[i]).union(C(i)))))
    for p in r:
        if i<p:
            for j in stations:
                for k in list(set(side_task[i-1]).intersection(set(side_task[p-1]))):
                    model.add_constraint(t_f[p]-t_f[i]+u*(1-x[p,j,k])+u*(1-x[i,j,k])+u*(1-z[i,p])
                                         >= time[p-1])
# ct7
for i in tasks:
    r = list(set(tasks).difference(set(Pa[i]).union(set(Sa[i]).union(C(i)))))
    for p in r:
        if i<p:
            for j in stations:
                for k in list(set(side_task[i-1]).intersection(set(side_task[p-1]))):
                    model.add_constraint(t_f[i]-t_f[p]+u*(1-x[p,j,k])+u*(1-x[i,j,k])+u*z[i,p]
                                         >= time[i-1])
# ct8
for i in AL:
    model.add_constraint(model.sum(x[i,j,0] for j in stations) == 1)
for i in AR:
    model.add_constraint(model.sum(x[i,j,1] for j in stations) == 1)

for i in tasks:
    model.add_constraint(t_f[i] >= time[i-1])

model.minimize(ct)
sol = model.solve()
print(sol)
