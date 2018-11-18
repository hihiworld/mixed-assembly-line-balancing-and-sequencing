
num_products=3
num_tasks=5
num_stations=5

jobs=[
    ('(1,1)',1, 1, 4, 1),('(1,2)',1, 2, 4, 1),
    ('(2,1)',2, 1, 2, 2),('(2,2)',2, 2, 2, 2), ('(2,3)', 2, 3, 2, 1),
    ('(3,1)',2, 1, 2, 3),                       ('(3,3)', 3, 3, 2, 1),
    ('(4,1)',4, 1, 2, 1),('(4,2)',4, 2, 2, 1), ('(4,3)', 4, 3, 2, 2),
                        ('(5,2)',5, 2, 4, 2), ('(5,3)', 5, 3, 4, 2)
]

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
jobsCapableStation = {'(1,1)':(1,2,3,4,5), '(1,2)':(1,2,3,4,5),
                      '(2,1)':(1,2,4),'(2,2)':(1,2,4),'(2,3)':(1,2,4),
                      '(3,1)':(1,2,3,5),'(3,3)':(1,2,3,5),
                      '(4,1)':(1,3,4,5),'(4,2)':(1,3,4,5),'(4,3)':(1,3,4,5),
                      '(5,2)':(2,3,4,5),'(5,3)':(2,3,4,5)}
                      
tasksCapableStation = [(1,2,3,4,5), (1,2,4), (1,2,3,5), (1,3,4,5), (2,3,4,5)]

setofTasks = [(1,2,3,4), (1,2,3,5), (1,3,4,5), (1,2,4,5), (1,3,4,5)]


from collections import namedtuple
from docplex.mp.model import Model
from docplex.mp.environment import Environment


Tstations = namedtuple('stations',['name','stations_number','totalworkspace'])
Tjobs = namedtuple("jobs", ["name", 'tasks', 'productions', 'duration', 'workspace'])


def setup_loaddata(model, jobs_, stations_, workspace_,duration_,presences_):
    model.jobs = [Tjobs(*job) for job in jobs_]
    model.stations = [Tstations(*station) for station in stations_]
    model.totalworkspace = {station.name: station.totalworkspace for station in model.stations}
    model.tasks = [task for task in range(1, num_tasks+1)]
    model.workspace = workspace_
    model.duration = duration_
    model.presences = presences_
    model.Nb_stations = {station.stations_number: i for i in range(1, num_stations+1) for station in model.stations}

def setup_variables(model):
    all_jobs, all_stations = model.jobs, model.stations
    tasks = model.tasks

    model.X = model.binary_var_matrix(all_stations, all_jobs, name='X')
    model.Y = model.binary_var_matrix(all_stations, tasks, name='Y')

    # model.C = model.integer_var_matrix(all_stations, all_jobs, lb=0, name='CompletTime')
    model.W = model.integer_var(lb=0, name='workload')

def setup_constraints(model):
    all_jobs, all_stations = model.jobs, model.stations
    all_tasks = model.tasks
    # C = model.C
    X = model.X
    a = model.workspace
    Y = model.Y
    b = model.totalworkspace
    d = model.duration
    W = model.W

    # 约束2
    for job in all_jobs:
        model.add_constraint(model.sum(X[station, job] for station in all_stations if station.stations_number in jobsCapableStation[job.name]) == 1)


    # 约束14
    for s in presences:
        for p in s:
            for job in all_jobs:
                if job.name == p[0]:
                    model.add_constraint(model.sum(station.stations_number * X[station, job] for station in all_stations if station.stations_number in jobsCapableStation[p[0]]
                                               ) >= model.sum(station.stations_number * X[station, job] for station in all_stations if station.stations_number in jobsCapableStation[p[1]])
                                     )

    # 约束17
    for task in all_tasks:
        model.add_constraint(model.sum(Y[station, task] for station in all_stations
                                       if station.stations_number in tasksCapableStation[station.stations_number-1])
                             >= 1)


    #约束18
    for station in all_stations:
        model.add_constraint(model.sum(a[station.stations_number-1][task.name] * Y[station, task] for task in all_tasks
                                       if task in setofTasks)
                             <= b[station.name])

    # 约束19
    for task in all_tasks:
        for station in all_stations:
            if station.stations_number not in tasksCapableStation[station.stations_number-1]:
                model.add_constraint(Y[station, task] == 0)

    #约束29
    for station in all_stations:
        for job in all_jobs:
            if station.stations_number in jobsCapableStation[job.name]:
                model.add_constraint(X[station, job] <= Y[station, job.tasks])

    # 约束30
    for station in all_stations:
        for task in all_tasks:
            model.add_constraint(Y[station, task] <= model.sum(X[station, job] for job in all_jobs if job.tasks == task))


    workload = (model.sum(d[station.stations_number-1][job.tasks-1] * X[station, job] for job in all_jobs for station in all_stations))
    model.add_constraint(workload <= W)


def setup_obj(model):
    model.minimize(model.W)

def solve(model):
    sol = model.solve()
    print(sol)

def build(context=None, **kwargs):
    mdl = Model('dd')
    setup_loaddata(mdl, jobs, stations, workspace, duration, presences)
    setup_variables(mdl)
    setup_constraints(mdl)
    setup_obj(mdl)
    return mdl

if __name__ == '__main__':
    model = build()
    solve(model)
