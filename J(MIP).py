num_products=3
num_tasks=5
num_stations=5


jobs=[
    ((1,1),1, 1, 4, 1),((1,2),1, 2, 4, 1),
    ((2,1),2, 1, 2, 2),((2,2),2, 2, 2, 2), ((2,3), 2, 3, 2, 1),
    ((3,1),2, 1, 2, 3),                       ((3,3), 3, 3, 2, 1),
    ((4,1),4, 1, 2, 1),((4,2),4, 2, 2, 1), ((4,3), 4, 3, 2, 2),
                        ((5,2),5, 2, 4, 2), ((5,3), 5, 3, 4, 2)]

duration = [
    [4,2,2,2,4],
    [4,2,2,0,4],
    [4,0,2,2,4],
    [4,2,0,2,4],
    [4,0,2,2,4]
]

workspace=[
    [1,2,3,1,0],
    [1,2,3,0,2],
    [1,0,3,1,2],
    [1,2,0,1,2],
    [1,0,3,1,2]
]

presences = [
    (((1,1), (2,1)), ((2,1), (3,1)), ((1,1), (4,1)), ((2,1), (4,1)), ((3,1), (4,1))),
    (((1,2), (2,2)), ((1,2), (4,2)), ((1,2), (5,2)), ((2,2), (5,2)), ((4,2), (5,2))),
    (((2,3), (3,3)), ((2,3), (4,3)), ((2,3), (5,3)), ((3,3), (5,3)), ((4,3), (5,3)))]

stations = [
    ('stations_1',1, 5),
    ('stations_2',2, 5),
    ('stations_3',3, 5),
    ('stations_4',4, 5),
    ('stations_5',5, 5)
]

totalworkspace = [5,5,5,5,5]


jobsCapableStation = {(1,1):(1,2,3,4,5), (1,2):(1,2,3,4,5),
                      (2,1):(1,2,4), (2,2):(1,2,4),(2,3):(1,2,4),
                      (3,1):(1,2,3,5),(3,3):(1,2,3,5),
                      (4,1):(1,3,4,5),(4,2):(1,3,4,5),(4,3):(1,3,4,5),
                      (5,2):(2,3,4,5),(5,3):(2,3,4,5)}
                      
tasksCapableStation = [(1,2,3,4,5), (1,2,4), (1,2,3,5), (1,3,4,5), (2,3,4,5)]

setofTasks = [(1,2,3,4), (1,2,3,5), (1,3,4,5), (1,2,4,5), (1,3,4,5)]

from collections import namedtuple
from docplex.mp.model import Model
from docplex.mp.environment import Environment


Tjobs = namedtuple("jobs", ["name", 'tasks', 'productions', 'duration', 'workspace'])


def setup_loaddata(model, jobs_, workspace_,duration_,presences_):
    model.jobs = [Tjobs(*job) for job in jobs_]
    model.stations = [station for station in range(1, num_stations+1)]
    model.totalworkspace = totalworkspace
    model.tasks = [task for task in range(1, num_tasks+1)]
    model.workspace = workspace_
    model.duration = duration_
    model.presences = presences_
    model.productions = [production for production in range(1, num_products+1)]
    
def setup_variables(model):
    all_jobs, all_stations = model.jobs, model.stations
    all_tasks = model.tasks
    all_productions = model.productions

    model.X = model.binary_var_matrix(all_stations, all_jobs, name='X')
    model.Y = model.binary_var_matrix(all_stations, all_tasks, name='Y')
    model.Z = model.binary_var_cube(all_stations, all_jobs, all_jobs, name='Z')
    model.U = model.binary_var_matrix(all_productions, all_productions, name='U')

    model.C = model.integer_var_matrix(all_stations, all_jobs, lb=0, name='CompletTime')
    model.A = model.integer_var_matrix(all_stations, all_productions, lb=0, name='ArrivalTime')
    model.D = model.integer_var_matrix(all_stations, all_productions, lb=0, name='DepartTime')
    model.M = model.integer_var(lb=0, name='makespan')
    model.W = model.integer_var(lb=0, name='workload')
    model.C_max = model.integer_var(lb=0, name='C_max')


def setup_constraints(model):
    all_jobs = model.jobs
    all_stations = model.stations
    all_tasks = model.tasks
    all_productions = model.productions
    C = model.C
    X = model.X
    a = model.workspace
    Y = model.Y
    b = model.totalworkspace
    d = model.duration
    W = model.W
    Z= model.Z
    M = 99999
    C_max = model.C_max
    D = model.D
    A = model.A
    U = model.U


    # 约束4
    for job in all_jobs:
        for station in all_stations:
            if station in jobsCapableStation[job.name]:
                model.add_constraint(C[station, job] <= M * X[station, job])

    # 约束5
    for j in all_jobs:
        for r in all_jobs:
            if j.productions < r.productions:
                for station in all_stations:
                    if station in jobsCapableStation[j.name] and jobsCapableStation[r.name]:
                            model.add_constraint(C[station, j] + d[station-1][r.tasks-1]
                                     * X[station, r] <= C[station, r] + M*(1-Z[station, j, r]))
    # 约束6
    for j in all_jobs:
        for r in all_jobs:
            if j.productions <= r.productions:
                for station in all_stations:
                    if station in jobsCapableStation[j.name] and jobsCapableStation[r.name]:
                        model.add_constraint(X[station, j]+X[station, r]
                                     - 2*(Z[station, j, r]+Z[station, r, j]) >= 0)

    # 约束7
    for j in all_jobs:
        for r in all_jobs:
            if j.productions <= r.productions:
                for station in all_stations:
                    if station in jobsCapableStation[j.name] and jobsCapableStation[r.name]:
                        model.add_constraint(X[station, j]+X[station, r]
                                     <= Z[station, j, r] + Z[station, r, j] + 1)
    # 约束8
    for j in all_jobs:
        for r in all_jobs:
            if j.tasks != r.tasks and j.productions == r.productions:
                for station in all_stations:
                    if station in jobsCapableStation[j.name] and jobsCapableStation[r.name]:
                        model.add_constraint(C[station, j] + d[station-1][r.tasks-1]*X[station, r]
                                     <= C[station, r] + M*(1-Z[station, j, r]))
    # 约束9
    for j in all_jobs:
        for r in all_jobs:
            if j.tasks != r.tasks and j.productions == r.productions:
                for station in all_stations:
                    if station in jobsCapableStation[j.name] and jobsCapableStation[r.name]:
                        model.add_constraint(X[station, j] + X[station, r] -2*(Z[station, j, r]+Z[station, r, j]) >= 0)

    # 约束10
    for j in all_jobs:
        for r in all_jobs:
            if j.tasks != r.tasks and j.productions == r.productions:
                for station in all_stations:
                    if station in jobsCapableStation[j.name] and jobsCapableStation[r.name]:
                        model.add_constraint(X[station, j] + X[station, r] <= Z[station, j, r]+Z[station, r, j] + 1)

    # 约束11
    for job in all_jobs:
        for station in all_stations:
            if station in jobsCapableStation[job.name]:
                model.add_constraint(C[station, job] <= C_max)

    # 约束12
    for job in all_jobs:
        for station in all_stations:
            if station not in jobsCapableStation[job.name]:
                model.add_constraint(C[station, job] <= 0)
                model.add_constraint(X[station, job] <= 0)

    # 约束13
    for s in presences:
        for p in s:
            for station_m in all_stations:
                if station_m in jobsCapableStation[p[0]]:
                    for station_i in all_stations:
                        if station_i in jobsCapableStation[p[1]]:
                            for job_1 in all_jobs:
                                if job_1.name == p[0]:
                                    for job_2 in all_jobs:
                                        if job_2.name == p[1]:
                                            model.add_constraint(C[station_m, job_2] >= C[station_i, job_1] +
                                                 d[station_m-1][job_2.tasks-1] - M * (1-X[station_m, job_2]))

    # 约束15
    # 约束16
    # 约束23
    
    for p in all_productions:
        for q in all_productions:
            for v in all_productions:
                if p != q and v == 1:
                    for station in all_stations:
                        model.add_constraint(D[station, p] - A[station, q] <= M*(2-U[v, p]) - U[v+1, q])
                        
    # 约束24
    for p in all_productions:
        for q in all_productions:
            for v in all_productions:
                if p != q and v > 1:
                    for station in all_stations:
                        model.add_constraint(D[station, p] - A[station, q] <= M*(2-U[v-1, p]) - U[v, q])
                        
    # 约束25
    for station in all_stations:
        for p in all_productions:
            if station < num_stations:
                model.add_constraint(A[station+1, p] == D[station, p])

    # 约束26
    for station in all_stations:
        for p in all_productions:
            if station == num_stations:
                model.add_constraint(A[station, p] == D[station-1, p])
                

    # 约束28
    for m in all_stations:
        for p in all_productions:
            model.add_constraint(A[m, p] >= 0)
            model.add_constraint(D[m, p] >= 0)

    # 约束31
        for j in all_jobs:
            for m in all_stations:
                if m in jobsCapableStation[j.name]:
                    model.add_constraint(A[m, j.productions] <= C[m, j] - d[m-1][j.tasks-1]*X[m, j] + M*(1-X[m, j]))

    # 约束32
    for j in all_jobs:
        for m in all_stations:
            if m in jobsCapableStation[j.name]:
                model.add_constraint(D[m, j.productions] >= C[m, j])
                
    # 约束33
    for m in all_stations:
        for p in all_productions:
            model.add_constraint(D[m, p] >= A[m, p] + model.sum(d[m-1][j.tasks-1]*X[m, j] for j in all_jobs if j.productions == p))

    # 约束33
    for m in all_stations:
        for p in all_productions:
            model.add_constraint(D[m,p] <= C_max)

def setup_objective(model):
    C_max = model.C_max
    model.minimize(C_max)
    

def solve(model):
    sol = model.solve(agent='local')    #force model to solve locally
    print(sol)

def build(**kwargs):
    mdl = Model('dd')
    setup_loaddata(mdl, jobs, workspace, duration, presences)
    setup_variables(mdl)
    setup_constraints(mdl)
    setup_objective(mdl)
    return mdl


if __name__ == '__main__':
    model = build()
    solve(model)
