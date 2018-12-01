from docplex.mp.model import Model
from docplex.util.environment import get_environment


num_tasks = 5
num_stations = 3
num_products = 3

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


def load_data(model, jobs_, duration_, workspace_, totalworkspace_, jobsCapableStation_, tasksCapableStation_,
              setofTasks_, presences_):

    model.jobs = [job for job in jobs_]
    model.stations = [m for m in range(1, num_stations+1)]
    model.productions = [p for p in range(1, num_products+1)]
    model.tasks = [task for task in range(1, num_tasks+1)]
    model.duration = duration_
    model.workspace = workspace_
    model.presence = presences_
    model.jobsCapableStation = jobsCapableStation_
    model.totalworkspace = totalworkspace_
    model.tasksCapableStation = tasksCapableStation_
    model.setofTasks = setofTasks_


def setup_var(model):
    all_jobs = model.jobs
    all_stations = model.stations
    all_tasks = model.tasks


    model.X = model.binary_var_matrix(all_stations, all_jobs, name='X')
    model.Y = model.binary_var_matrix(all_stations, all_tasks, name='Y')

    model.W = model.integer_var(lb=0, name='workload')


def setup_cts(model):

    all_jobs = model.jobs
    all_stations = model.stations
    all_tasks = model.tasks

    X = model.X
    Y = model.Y
    W = model.W
    d = model.duration
    w = model.workspace
    b = model.totalworkspace

    jobsCapableStation = model.jobsCapableStation
    presences = model.presence
    tasksCapableStation = model.tasksCapableStation
    setofTasks = model.setofTasks

    # 约束2
    for j in all_jobs:
        model.add_constraint(model.sum(X[m, j] for m in jobsCapableStation[j]) == 1,
                             'ct2_{}'.format(j))

    # 约束14
    for p in presences:
        model.add_constraint(model.sum(i * X[i, p[0]] for i in jobsCapableStation[p[0]])
                             <= model.sum(m * X[m, p[1]] for m in jobsCapableStation[p[1]]), 'ct14_{}'.format(p))

    # 约束17
    for t in all_tasks:
        model.add_constraint(model.sum(Y[m, t] for m in tasksCapableStation[t - 1]) >= 1,
                             'ct17_{}'.format(t))

    # 约束18
    for m in all_stations:
        model.add_constraint(model.sum(w[m - 1][t - 1] * Y[m, t] for t in setofTasks[m - 1])
                             <= b[m - 1], 'ct18_{}'.format(m))

    # 约束19
    for t in all_tasks:
        for m in all_stations:
            if m not in tasksCapableStation[t - 1]:
                model.add_constraint(Y[m, t] == 0, 'ct19-{}_{}'.format(m, t))

    # 约束29
    for j in all_jobs:
        for m in jobsCapableStation[j]:
            model.add_constraint(X[m, j] <= Y[m, j[0]], 'ct29_{}_{}'.format(m, j))

    # 约束30
    for t in all_tasks:
        for m in all_stations:
            model.add_constraint(Y[m, t] <= model.sum(X[m, j] for j in all_jobs if j[0] == t),
                                 'ct30_{}_{}'.format(m, t))

    # 约束55
    for m in all_stations:
        model.add_constraint(W >= model.sum(d[m-1][j[0]-1] * X[m, j] for j in all_jobs),
                             'ct56_{}'.format(m))

def setup_obj(model):
    model.minimize(model.W)

def solve(model):
    # url='https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/'
    # key='api_6d433c14-5a73-4d9f-8364-ec787ca576d9'
    sol = model.solve()
    print(sol)

def build(**kwargs):
    mdl = Model()
    load_data(mdl,jobs,duration,workspace,totalworkspace,
              jobsCapableStation,tasksCapableStation,setofTasks,presences)
    setup_var(mdl)
    setup_cts(mdl)
    setup_obj(mdl)
    return mdl

if __name__ == '__main__':
    model = build()
    solve(model)
    with get_environment().get_output_stream('A_{}_{}_{}.json'.format(num_tasks,num_products,num_stations)) as fp:
        model.solution.export(fp, 'json')
