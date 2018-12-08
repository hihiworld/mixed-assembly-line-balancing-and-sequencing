def setup_obj_A_mip(model):
    model.minimize(model.W)

def setup_obj_PJ_mip(model):
    model.minimize(model.C_max)

def solve(model):
    sol = model.solve()
    print(sol)