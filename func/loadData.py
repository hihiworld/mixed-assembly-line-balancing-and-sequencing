import json

def load_data(filename):
    return None

def load_last_Sol(model, filename):

    fp = json.load(open(filename, 'r'))['CPLEXSolution']['variables']

    P={}

    for f in fp:

        if 'X' in f['name']:

            P[f['name']]=int(1)

    return P

def setup_data(model, jobs_, duration_, workspace_, totalworkspace_,
               jobsCapableStation_, tasksCapableStation_, setofTasks_, presences_,early_,
               num_stations_, num_products_, num_tasks_, filename_):

    model.jobs = [job for job in jobs_]

    model.stations = [m for m in range(1, num_stations_ + 1)]

    model.productions = [p for p in range(1, num_products_ + 1)]

    model.tasks = [task for task in range(1, num_tasks_ + 1)]

    model.duration = duration_

    model.workspace = workspace_

    model.presence = presences_

    model.jobsCapableStation = jobsCapableStation_

    model.totalworkspace = totalworkspace_

    model.tasksCapableStation = tasksCapableStation_

    model.setofTasks = setofTasks_

    model.num_stations = num_stations_

    model.num_products = num_products_

    model.num_tasks = num_tasks_

    model.P = load_last_Sol(model, filename_)

    model.early = early_