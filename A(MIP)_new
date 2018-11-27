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
    (((1, 1), (2, 1)), ((2, 1), (3, 1)), ((1, 1), (4, 1)), ((2, 1), (4, 1)), ((3, 1), (4, 1))),
    (((1, 2), (2, 2)), ((1, 2), (4, 2)), ((1, 2), (5, 2)), ((2, 2), (5, 2)), ((4, 2), (5, 2))),
    (((2, 3), (3, 3)), ((2, 3), (4, 3)), ((2, 3), (5, 3)), ((3, 3), (5, 3)), ((4, 3), (5, 3)))
]

jobsCapableStation = {(1, 1): (1, 2, 3), (1, 2): (1, 2, 3),
                      (2, 1): (1, 2), (2, 2): (1, 2), (2, 3): (1, 2),
                      (3, 1): (1, 2, 3), (3, 3): (1, 2, 3),
                      (4, 1): (1, 3), (4, 2): (1, 3), (4, 3): (1, 3),
                      (5, 2): (2, 3), (5, 3): (2, 3)}

totalworkspace = [5, 5, 5]
tasksCapableStation = [(1, 2, 3), (1, 2), (1, 2, 3), (1, 3), (2, 3)]

setofTasks = [(1, 2, 3, 4), (1, 2, 3, 5), (1, 3, 4, 5)]

from collections import namedtuple
from docplex.mp.model import Model
from docplex.util.environment import get_environment

model = Model()


model.jobs = [job for job in jobs]

model.stations = [m for m in range(1, num_stations+1)]
model.totalworkspace = totalworkspace
model.tasks = [task for task in range(1, num_tasks + 1)]
model.workspace = workspace
model.duration = duration
model.presences = presences

all_jobs, all_stations = model.jobs, model.stations
all_tasks = model.tasks

X = model.binary_var_matrix(all_stations, all_jobs, name='X')
Y = model.binary_var_matrix(all_stations, all_tasks, name='Y')

W = model.integer_var(lb=0, name='workload')

for j in all_jobs:
    model.add_constraint(model.sum(X[m, j] for m in all_stations if m in jobsCapableStation[j]) == 1)

for s in presences:
    for p in s:
        model.add_constraint(model.sum(n*X[n, p[0]] for n in jobsCapableStation[p[0]])
                             <= model.sum(m*X[m, p[1]] for m in jobsCapableStation[p[1]]))

for t in all_tasks:
    model.add_constraint(model.sum(Y[m ,t] for m in all_stations if m in tasksCapableStation[t-1]) >= 1)

for m in all_stations:
    model.add_constraint(model.sum(workspace[m-1][t-1]*Y[m, t] for t in all_tasks if t in setofTasks[m-1]) <= totalworkspace[m-1])

for t in all_tasks:
    for m in all_stations:
        if m not in tasksCapableStation[t-1]:
            model.add_constraint(Y[m, t] == 0)

for j in all_jobs:
    for m in all_stations:
        if m in jobsCapableStation[j]:
            model.add_constraint(X[m ,j] <= Y[m, j[0]])

for t in all_tasks:
    for m in all_stations:
        model.add_constraint(Y[m ,t] <= model.sum(X[m, j] for j in all_jobs if j[0] == t))

for m in all_stations:
        model.add_constraint(W >= model.sum(duration[m-1][j[0]-1]*X[m, j] for j in all_jobs))

model.minimize(W)
sol = model.solve()
print(sol)
