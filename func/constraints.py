def setup_cts_A_mip(model):
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
        model.add_constraint(W >= model.sum(d[m - 1][j[0] - 1] * X[m, j] for j in all_jobs),

                             'ct56_{}'.format(m))


def setup_cts_PJ_mip(model):
    all_jobs = model.jobs

    all_stations = model.stations

    all_productions = model.productions

    jobsCapableStation = model.jobsCapableStation

    presences = model.presence

    duration = model.duration

    num_stations = model.num_stations

    early = model.early

    P = model.P

    M = 99999

    Z = model.Z

    U = model.U

    C = model.C

    A = model.A

    D = model.D

    C_max = model.C_max

    # 约束3

    for j in all_jobs:

        for m in jobsCapableStation[j]:

            a = 'X_{}_{}'.format(m, j)

            if a in P.keys():
                model.add_constraint(C[m, j] >= early[j[1] - 1][j[0] - 1] * P[a], 'ct3_{}_{}'.format(m, j))

    # 约束4

    for j in all_jobs:

        for m in jobsCapableStation[j]:

            a = 'X_{}_{}'.format(m, j)

            if a in P:
                model.add_constraint(C[m, j] <= M * P[a], 'ct4_{}_{}'.format(m, j))



    # 约束5

    for j in all_jobs:

        for r in all_jobs:

            if j[1] < r[1]:

                for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):

                    a = 'X_{}_{}'.format(m, j)

                    b = 'X_{}_{}'.format(m, r)

                    if a in P.keys() and b in P.keys():
                        model.add_constraint(C[m, j] + duration[m - 1][r[0] - 1]

                                             * P[b] <= C[m, r] + M * (1 - Z[m, j, r]), 'ct6_{}_{}_{}'.format(m, j, r))

    # 约束6

    for j in all_jobs:

        for r in all_jobs:

            if j[1] < r[1]:

                for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):

                    a = 'X_{}_{}'.format(m, j)

                    b = 'X_{}_{}'.format(m, r)

                    if a in P.keys() and b in P.keys():
                        model.add_constraint(P[a] + P[b] - 2 * (Z[m, j, r] + Z[m, r, j]) >= 0,
                                             'ct6_{}_{}_{}'.format(m, j, r))

    # 约束7

    for j in all_jobs:

        for r in all_jobs:

            if j[1] < r[1]:

                for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):

                    a = 'X_{}_{}'.format(m, j)

                    b = 'X_{}_{}'.format(m, r)

                    if a in P.keys() and b in P.keys():
                        model.add_constraint(P[a] + P[b] <= Z[m, j, r] + Z[m, r, j] + 1, 'ct7_{}_{}_{}'.format(m, j, r))

    # 约束8

    for j in all_jobs:

        for r in all_jobs:

            if j[0] != r[0] and j[1] == r[1]:

                for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):

                    b = 'X_{}_{}'.format(m, r)

                    a = 'X_{}_{}'.format(m, j)

                    if b in P.keys() and a in P.keys():
                        model.add_constraint(C[m, j] + duration[m - 1][r[0] - 1] * P[b]

                                             <= C[m, r] + M * (1 - Z[m, j, r]), 'ct8_{}_{}_{}'.format(m, j, r))

    # 约束9

    for j in all_jobs:

        for r in all_jobs:

            if j[0] != r[0] and j[1] == r[1]:

                for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):

                    a = 'X_{}_{}'.format(m, j)

                    b = 'X_{}_{}'.format(m, r)

                    if a in P.keys() and b in P.keys():
                        model.add_constraint(

                            P[a] + P[b] - 2 * (Z[m, j, r] + Z[m, r, j]) >= 0,

                            'ct9_{}_{}_{}.for'.format(m, j, r))

    # 约束10

    for j in all_jobs:

        for r in all_jobs:

            if j[0] != r[0] and j[1] == r[1]:

                for m in set(jobsCapableStation[j]).intersection(set(jobsCapableStation[r])):

                    a = 'X_{}_{}'.format(m, j)

                    b = 'X_{}_{}'.format(m, r)

                    if a in P.keys() and b in P.keys():
                        model.add_constraint(P[a] + P[b] <= Z[m, j, r] + Z[m, r, j] + 1,

                                             'ct10_{}_{}_{}'.format(m, j, r))

    # 约束11

    for j in all_jobs:

        for m in jobsCapableStation[j]:
            model.add_constraint(C[m, j] <= C_max, 'ct11_{}_{}'.format(m, j))

    # 约束12

    for j in all_jobs:

        for m in all_stations:

            if 'X_{}_{}'.format(m, j) not in P.keys():
                model.add_constraint(C[m, j] <= 0, 'ct12_C_{}_{}'.format(m, j))



    # 约束13

    for p in presences:

        for i in jobsCapableStation[p[0]]:

            for m in jobsCapableStation[p[1]]:

                a = 'X_{}_{}'.format(m, p[1])

                if a in P.keys():
                    model.add_constraint(C[m, p[1]] >= C[i, p[0]] + duration[m - 1][p[1][0] - 1] - M * (1 - P[a]),

                                         'ct13_{}_{}_{}'.format(p, i, m))

    # 约束15

    for j in all_jobs:

        for m in jobsCapableStation[j]:
            model.add_constraint(C[m, j] >= 0, 'ct15_C_{}_{}'.format(m, j))

            model.add_constraint(C_max >= 0, 'ct15_C_max_{}_{}'.format(m, j))

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

                if p != q and v == 1:

                    for m in all_stations:
                        model.add_constraint(D[m, p] - A[m, q] <= M * (2 - U[v, p] - U[v + 1, q]),

                                             'ct23_{}_{}_{}_{}'.format(m, p, q, v))

    # 约束24

    for p in all_productions:

        for q in all_productions:

            for v in all_productions:

                if p != q and v > 1:

                    for m in all_stations:
                        model.add_constraint(D[m, p] - A[m, q] <= M * (2 - U[v - 1, p] - U[v, q]),

                                             'ct24_{}_{}_{}_{}'.format(m, p, q, v))

    # 约束25

    for p in all_productions:

        for m in all_stations:

            if m < num_stations:
                model.add_constraint(A[m + 1, p] == D[m, p], 'ct25_{}_{}'.format(m, p))

    # 约束26

    for p in all_productions:

        for m in all_stations:

            if m == num_stations:
                model.add_constraint(A[m, p] == D[m - 1, p], 'ct26_{}_{}'.format(m, p))

    # 约束28

    for m in all_stations:

        for p in all_productions:
            model.add_constraint(A[m, p] >= 0, 'ct28_{}_{}'.format(m, p))

            model.add_constraint(D[m, p] >= 0, 'ct28_{}_{}'.format(m, p))

    # 约束31

    for j in all_jobs:

        for m in jobsCapableStation[j]:

            a = 'X_{}_{}'.format(m, j)

            if a in P.keys():
                model.add_constraint(

                    A[m, j[1]] <= C[m, j] - duration[m - 1][j[0] - 1] * P[a] + M * (1 - P[a]),

                    'ct31_{}_{}'.format(m, j))

    # 约束32

    for j in all_jobs:

        for m in jobsCapableStation[j]:

            a = 'X_{}_{}'.format(m, j)

            if a in P.keys():
                model.add_constraint(D[m, j[1]] >= C[m, j], 'ct32_{}_{}'.format(m, j))

    # 约束33

    for m in all_stations:

        for p in all_productions:
            model.add_constraint(

                D[m, p] >= A[m, p] +

                model.sum(duration[m - 1][j[0] - 1] * P['X_{}_{}'.format(m, j)] for j in all_jobs

                          if j[1] == p and 'X_{}_{}'.format(m, j) in P.keys()),

                'ct33_{}_{}'.format(m, p))

    # 约束34

    for m in all_stations:

        for p in all_productions:
            model.add_constraint(D[m, p] <= C_max, 'ct34_{}_{}'.format(m, p))

def setup_cts_J_mip(model):
    return
