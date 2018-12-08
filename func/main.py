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

early = [

    [4, 6, 6, 8, 0],

    [4, 6, 0, 6, 10],

    [0, 2, 4, 4, 8]

]
import solution.loadData
import solution.constraints
import solution.solve
import solution.vars

from docplex.mp.model import Model


model = Model('decomposition')

def run_PJ_mip(filename):
    solution.loadData.load_last_Sol(model, filename)
    solution.loadData.setup_data(model,jobs,duration,workspace,totalworkspace,jobsCapableStation,tasksCapableStation,setofTasks
                                 ,presences,early, num_stations,num_products,num_tasks,filename)
    solution.vars.setup_vars(model)
    solution.solve.setup_obj_PJ_mip(model)
    solution.constraints.setup_cts_PJ_mip(model)
    solution.solve.solve(model)

def run_A_mip(filename):
    solution.loadData.setup_data(model,jobs,duration,workspace,totalworkspace,jobsCapableStation,tasksCapableStation,setofTasks
                                 ,presences,early, num_stations,num_products,num_tasks,filename)
    solution.vars.setup_vars(model)
    solution.solve.setup_obj_A_mip(model)
    solution.constraints.setup_cts_A_mip(model)
    solution.solve.solve(model)

def run_all():
    for t in range(num_tasks):
        for p in range(num_products):
            for m in range(num_stations):
                filename=r'C:\Users\root\PycharmProjects\untitled\paper1' + '\A_{}_{}_{}.json'.format(t, p, m)
                run_PJ_mip(filename)

filename=r'C:\Users\root\PycharmProjects\untitled\paper1\A_5_3_3.json'
run_PJ_mip(filename)
# run_A_mip(filename)