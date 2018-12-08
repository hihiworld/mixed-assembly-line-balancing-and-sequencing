def setup_vars(model):
    all_jobs = model.jobs

    all_stations = model.stations

    all_tasks = model.tasks

    all_productions = model.productions

    model.X = model.binary_var_matrix(all_stations, all_jobs, name='X')

    model.Y = model.binary_var_matrix(all_stations, all_tasks, name='Y')

    model.Z = model.binary_var_cube(all_stations, all_jobs, all_jobs, name='Z')

    model.U = model.binary_var_matrix(all_productions, all_productions, name='U')

    model.C = model.integer_var_matrix(all_stations, all_jobs, name='Compelete')

    model.A = model.integer_var_matrix(all_stations, all_productions, name='ArrivalTime')

    model.D = model.integer_var_matrix(all_stations, all_productions, name='DepartTime')

    model.W = model.integer_var(lb=0, name='workload')

    model.C_max = model.integer_var(lb=0, name='C_max')