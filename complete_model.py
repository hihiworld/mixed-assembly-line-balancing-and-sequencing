num_products = 3
num_tasks = 5
num_stations = 3

jobs = [
    (1, 1), (1, 2),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 3),
    (4, 1), (4, 2), (4, 3),
    (5, 2), (5, 3)]

duration = [
            [4, 2, 2, 2, 4],
            [4, 2, 2, 0, 4],
            [4, 0, 2, 2, 4]
]

workspace = [
        [1, 2, 3, 1, 0],
        [1, 2, 3, 0, 2],
        [1, 0, 3, 1, 2]
]

presences = [
    ((1, 1), (2, 1)), ((1, 1), (3, 1)), ((2, 1), (4, 1)), ((3, 1), (4, 1)),
    ((1, 2), (2, 2)), ((1, 2), (4, 2)), ((2, 2), (5, 2)), ((4, 2), (5, 2)),
    ((2, 3), (3, 3)), ((2, 3), (4, 3)), ((3, 3), (5, 3)), ((4, 3), (5, 3))
]

jobsCapableStation = {(1, 1): (1, 2, 3), (1, 2): (1, 2, 3),
                      (2, 1): (1, 2), (2, 2): (1, 2), (2, 3): (1, 2),
                      (3, 1): (1, 2, 3), (3, 3): (1, 2, 3),
                      (4, 1): (1, 3), (4, 2): (1, 3), (4, 3): (1, 3),
                      (5, 2): (2, 3), (5, 3): (2, 3)}

totalworkspace = [5, 5, 5]
tasksCapableStation = [(1, 2, 3), (1, 2), (1, 2, 3), (1, 3), (2, 3)]

setofTasks = [(1, 2, 3, 4), (1, 2, 3, 5), (1, 3, 4, 5)]

early = [
    [4, 6, 6, 8, 0],
    [4, 6, 0, 6, 10],
    [0, 2, 4, 4, 8]
]

from collections import namedtuple
from docplex.mp.model import Model
from docplex.util.environment import get_environment
import numpy as np

model = Model()

model.jobs = [job for job in jobs]
model.stations = [m for m in range(1, num_stations+1)]
model.productions = [p for p in range(1, num_products+1)]
model.tasks = [task for task in range(1, num_tasks+1)]
model.totalworkspace = totalworkspace
model.workspace = workspace
model.duration = duration
model.presences = presences


all_jobs = model.jobs
all_stations = model.stations
all_tasks = model.tasks
all_productions = model.productions


X = model.binary_var_matrix(all_stations, all_jobs, name='X')
Y = model.binary_var_matrix(all_stations, all_tasks, name='Y')
Z = model.binary_var_cube(all_stations, all_jobs, all_jobs, name='Z')
U = model.binary_var_matrix(all_productions, all_productions, name='U')

W = model.integer_var(lb=0, name='workload')
C = model.integer_var_matrix(all_stations, all_jobs, name='Compelete')
A = model.integer_var_matrix(all_stations, all_productions, name='ArrivalTime')
D = model.integer_var_matrix(all_stations, all_productions, name='DepartTime')
C_max = model.integer_var(lb=0, name='C_max')

M = 99999

# 约束2
for j in all_jobs:
    model.add_constraint(model.sum(X[m, j] for m in jobsCapableStation[j]) == 1,
                         'ct2_{}'.format(j))

# 约束3
for j in all_jobs:
    for m in jobsCapableStation[j]:
        model.add_constraint(C[m, j] >= early[j[1]-1][j[0]-1] * X[m, j],'ct3_{}_{}'.format(m,j))

# 约束4
for j in all_jobs:
    for m in jobsCapableStation[j]:
        model.add_constraint(C[m, j] <= M * X[m, j],'ct4_{}_{}'.format(m,j))

# 约束5
for j in all_jobs:
    for r in all_jobs:
        if j[1] < r[1]:
            for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):
                model.add_constraint(C[m, j] + duration[m - 1][r[0] - 1]
                                             * X[m, r] <= C[m, r] + M * (1 - Z[m, j, r]),'ct6_{}_{}_{}'.format(m,j,r))

# 约束6
for j in all_jobs:
    for r in all_jobs:
        if j[1] < r[1]:
            for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):
                model.add_constraint(X[m, j] + X[m, r] - 2 * (Z[m, j, r] + Z[m, r, j]) >= 0,'ct6_{}_{}_{}'.format(m,j,r))

# 约束7
for j in all_jobs:
    for r in all_jobs:
        if j[1] < r[1]:
            for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):
                model.add_constraint(X[m, j] + X[m, r] <= Z[m, j, r] + Z[m, r, j] + 1,'ct7_{}_{}_{}'.format(m,j,r))

# 约束8
for j in all_jobs:
    for r in all_jobs:
        if j[0] != r[0] and j[1] == r[1]:
            for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):
                model.add_constraint(C[m, j] + duration[m - 1][r[0] - 1] * X[m, r]
                                     <= C[m, r] + M * (1 - Z[m, j, r]),'ct8_{}_{}_{}'.format(m,j,r))

# 约束9
for j in all_jobs:
    for r in all_jobs:
        if j[0] != r[0] and j[1] == r[1]:
            for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):
                model.add_constraint(
                    X[m, j] + X[m, r] - 2 * (Z[m, j, r] + Z[m, r, j]) >= 0,'ct9_{}_{}_{}.for'.format(m,j,r))

# 约束10
for j in all_jobs:
    for r in all_jobs:
        if j[0] != r[0] and j[1] == r[1]:
            for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):
                model.add_constraint(X[m, j] + X[m, r] <= Z[m, j, r] + Z[m, r, j] + 1,'ct10_{}_{}_{}'.format(m,j,r))

# 约束11
for j in all_jobs:
    for m in jobsCapableStation[j]:
        model.add_constraint(C[m,j] <= C_max,'ct11_{}_{}'.format(m,j))

# 约束12
for j in all_jobs:
    for m in all_stations:
        if m not in jobsCapableStation[j]:
            model.add_constraint(C[m, j] <= 0,'ct12_C_{}_{}'.format(m,j))
            model.add_constraint(X[m, j] <= 0,'ct12_X_{}_{}'.format(m,j))
            
# 约束13
for p in presences:
    for i in jobsCapableStation[p[0]]:
        for m in jobsCapableStation[p[1]]:
            model.add_constraint(C[m, p[1]] >= C[i, p[0]] + duration[m-1][p[1][0]-1] - M*(1-X[m, p[1]]),
                                 'ct13_{}_{}_{}'.format(p, i, m))

# 约束14
for p in presences:
    model.add_constraint(model.sum(n * X[n, p[0]] for n in jobsCapableStation[p[0]])
                             <= model.sum(m * X[m, p[1]] for m in jobsCapableStation[p[1]]), 'ct14_{}'.format(p))

# 约束15
for j in all_jobs:
    for m in jobsCapableStation[j]:
        model.add_constraint(C[m, j] >= 0,'ct15_C_{}_{}'.format(m,j))
        model.add_constraint(C_max >= 0,'ct15_C_max_{}_{}'.format(m,j))

# 约束17
for t in all_tasks:
    model.add_constraint(model.sum(Y[m, t] for m in tasksCapableStation[t - 1]) >= 1,
                         'ct17_{}'.format(t))

# 约束18
for m in all_stations:
    model.add_constraint(model.sum(workspace[m - 1][t - 1] * Y[m, t] for t in setofTasks[m - 1])
                         <= totalworkspace[m - 1], 'ct18_{}'.format(m))

# 约束19
for t in all_tasks:
    for m in all_stations:
        if m not in tasksCapableStation[t - 1]:
            model.add_constraint(Y[m, t] == 0, 'ct19_{}_{}'.format(m, t))

# 约束21
for v in all_productions:
    model.add_constraint(model.sum(U[v, p] for p in all_productions) == 1)

# 约束22
for p in all_productions:
    model.add_constraint(model.sum(U[v, p] for v in all_productions) == 1)

# 约束23
for p in all_productions:
    for q in all_productions:
        for v in all_productions:
            if p != q and v ==1:
                for m in all_stations:
                    model.add_constraint(D[m, p] - A[m, q] <= M * (2 - U[v, p] - U[v + 1, q]),
                                         'ct23_{}_{}_{}_{}'.format(m,p,q,v))

# 约束24
for p in all_productions:
    for q in all_productions:
        for v in all_productions:
            if p != q and v > 1:
                for m in all_stations:
                    model.add_constraint(D[m, p] - A[m, q] <= M * (2 - U[v - 1, p] - U[v, q]),
                                         'ct24_{}_{}_{}_{}'.format(m,p,q,v))

# 约束25
for p in all_productions:
    for m in all_stations:
        if m < num_stations:
            model.add_constraint(A[m + 1, p] == D[m, p],'ct25_{}_{}'.format(m,p))

# 约束26
for p in all_productions:
    for m in all_stations:
        if m == num_stations:
            model.add_constraint(A[m, p] == D[m - 1, p],'ct26_{}_{}'.format(m,p))

# 约束28
for m in all_stations:
    for p in all_productions:
        model.add_constraint(A[m, p] >= 0,'ct28_{}_{}'.format(m,p))
        model.add_constraint(D[m, p] >= 0,'ct28_{}_{}'.format(m,p))

# 约束29
for j in all_jobs:
    for m in all_stations:
        if m in jobsCapableStation[j]:
            model.add_constraint(X[m, j] <= Y[m, j[0]], 'ct29_{}_{}'.format(m, j))

# 约束30
for t in all_tasks:
    for m in all_stations:
        model.add_constraint(Y[m, t] <= model.sum(X[m, j] for j in all_jobs if j[0] == t),
                             'ct30_{}_{}'.format(m, t))

# 约束31
for j in all_jobs:
    for m in jobsCapableStation[j]:
        model.add_constraint(
            A[m, j[1]] <= C[m, j] - duration[m - 1][j[0] - 1] * X[m, j] + M * (1 - X[m, j]),
            'ct31_{}_{}'.format(m,j))

# 约束32
for j in all_jobs:
    for m in jobsCapableStation[j]:
        model.add_constraint(D[m, j[1]] >= C[m, j],'ct32_{}_{}'.format(m,j))

# 约束33
for m in all_stations:
    for p in all_productions:
        model.add_constraint(
            D[m, p] >= A[m, p] + model.sum(duration[m - 1][j[0] - 1] * X[m, j] for j in all_jobs if j[1] == p),
            'ct33_{}_{}'.format(m, p))

# 约束34
for m in all_stations:
    for p in all_productions:
        model.add_constraint(D[m, p] <= C_max, 'ct34_{}_{}'.format(m, p))


model.minimize(C_max)


sol = model.solve()
print(sol)
