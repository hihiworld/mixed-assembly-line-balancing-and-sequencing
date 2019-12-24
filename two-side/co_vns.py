'''
@Description: 
@Author: yaologos
@Github: https://github.com/hihiworld
@Date: 2019-10-16 16:05:59
@LastEditors  : yaologos
@LastEditTime : 2019-12-24 15:29:06
'''
import copy
import random

import dataloader


num_tasks, num_stations, num_products, P0, AL, AR, AE, side_task, P, S,times, workspace, totalworkspace = dataloader.load_data_EA(
    9,3,3)


def initSolution():
    '''
    @description: 初始化生成任务序列，方向序列和产品序列
    @param {type} 
    @return: 
    @usage: 
    '''

    def init_task(product):
        '''
        @description: 生成初始解，依据次序约束，平均时间和容量限制

        @product {int}: 此时是依据产品的种类生成其任务序列 
        
        @return: [1, 1, 2, 1, 2, 2, 3, 3, 3]

        @usage: print(init_task(1))
        '''
        j = 1
        seq_task = [0]*num_tasks
        candidate = list(P0)
        assigned = []
        tasks = [i for i in range(1, num_tasks+1)]
        FIXEDTIME = round(sum(times[product-1])/num_stations, 2)

        TIME = 0
        CAPACITY = 0

        while 0 in seq_task:
            if j == num_stations:
                for t in set(tasks).difference(set(assigned)):
                    seq_task[i-1] = j
                break
            t = random.choice(candidate)
            TIME += times[product-1][t-1]
            CAPACITY += workspace[product-1][t-1]
            if CAPACITY <= totalworkspace[j-1]:
                if TIME <= FIXEDTIME:
                    seq_task[t-1] = j
                else:
                    j += 1
                    TIME = 0
                    CAPACITY = 0

                    if random.normalvariate(0, 1) < 0.5:
                        seq_task[t-1] = j
                        TIME += times[product-1][t-1]
                        CAPACITY += workspace[product-1][t-1]

                    else:
                        seq_task[t-1] = j-1

            else:
                j += 1
                TIME = 0
                CAPACITY = 0
                seq_task[t-1] = j
                TIME += times[product-1][t-1]
                CAPACITY += workspace[product-1][t-1]

            candidate.remove(t)
            assigned.append(t)

            for i in S[t]:
                if set(P[i]).issubset(assigned):
                    candidate.append(i)

        if 0 in seq_task:
            for k, i in enumerate(seq_task):
                if i == 0:
                    seq_task[k] = num_stations
        return seq_task


    def init_side():
        '''
        @description: 初始化边，对边进行编码，
        @param {type} 
        @return: [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
        '''
        seq_sides = [0] * num_tasks
        for i in range(1,num_tasks+1):
            if i in AL:
                seq_sides[i-1] = 0
            elif i in AR:
                seq_sides[i-1] = 1
            else:
                if random.random() > 0.5:
                    seq_sides[i-1] = 0
                else:
                    seq_sides[i-1] = 1
        return seq_sides
    def init_products():
        seq = [i for i in range(1, num_products+1)]
        random.shuffle(seq)
        return seq

    seq_products = init_products()
    seq_tasks = []
    seq_sides = []
    for i in range(1,num_products+1):
        seq_task = init_task(i)
        seq_tasks.append(seq_task)
        seq_sides.append(init_side())
    return seq_tasks,seq_sides, seq_products

def getSetOfTasksWithWeight(seq_task, weight):
    '''
    @description: 依据编码 [1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2] 和
                其对应任务权重 [961, 510, 968, 713, 994, 52, 945, 805, 242]
                    生成一个每个工作站被分配的任务集合

    @return: [[1, 3, 2, 6, 6], [7，4, 8, 9, 10, 11, 12], []]
    '''
    setoftasks = [[] for i in range(num_stations)]
    for m in range(1, num_stations+1):
        for k, t in enumerate(seq_task):
            if t == m:
                setoftasks[m-1].append(k+1)
    nums = [[] for i in range(num_stations)]
    candidate_tasks = list(P0)
    assigned_tasks = []
    while candidate_tasks:
        r = 0
        w = 0
        if len(candidate_tasks) == 1:
            r = candidate_tasks[0]
        else:
            for i in candidate_tasks:
                if w < weight[i-1]:
                    r = i
                    w = weight[i-1]
        assigned_tasks.append(r)
        candidate_tasks.remove(r)
        for i in S[r]:
            if set(P[i]).issubset(assigned_tasks):
                candidate_tasks.append(i)
    for k, seq in enumerate(setoftasks):
        for i in assigned_tasks:
            if i in seq:
                nums[k].append(i)
    return nums

def getSetOfTasksWithoutWeight(seq_task):
    '''
    @description: 返回无权重的每个工作站上加工的任务集合
    @param {seq_task} [2, 1, 1, 2, 1, 1, 3, 2, 2, 3, 2, 2]
    @return: [[2, 3, 5, 6], [1, 4, 8, 9, 11, 12], [7, 10]]
    @usage: print(getSetOfTasksWithoutWeight([2, 1, 1, 2, 1, 1, 3, 2, 2, 3, 2, 2]))
    '''
    nums = [[] for m in range(num_stations)]
    for m in range(num_stations):
        for k,task in enumerate(seq_task):
            if task == m+1:
                nums[m].append(k+1)
    return nums

def getWeight():
    '''
    @description: 生成任务权重，可以拓展为依据启发式规则生成权重
    @param {type} 
    @return: 
    @usage: 
    '''
    weight = []
    for i in range(num_tasks):
        rand = random.randint(1, 1000)
        if rand not in weight:
            weight.append(rand)
    return weight

def decode(seq_tasks, seq_sides, seq_products, times, weight):
    def searchInsertedPosition(nowTask, assigned_dict: dict):
        '''
        @description: 搜索满足次序关系，且可以插入的位置
        @param {type}
        @return:
        '''
        task = 0
        maxIndex = 0

        for i in P[nowTask]:
            if i in assigned_dict:
                temp = list(assigned_dict).index(i)
                if temp > maxIndex:
                    maxIndex = temp
                    task = i
        return maxIndex, task

    def insertDict(dic, index, key, value):
        '''
        @description: 向字典的指定位置插入键值对,例如将9:[9,10]插入到dic的第三个位置, insertDict(dic,2,9,[9,10])
        @param {type} 
            dic = {0: [0, 0], 5: [3, 4], 4: [6, 8]}
        @return: {0: [0, 0], 5: [3, 4], 9: [9, 10], 4: [6, 8]}
        '''
        num = list(dic)
        num.insert(index, key)
        new_dic = {}
        for i in num:
            if i not in dic.keys():
                new_dic[i] = value
            else:
                new_dic[i] = dic[i]
        return new_dic

    def updateIdleTime(task_time, CT):
        '''
        @description: 更新idle time的序列
        @param {type} 
            param1 - task_time = {0:[0,0], 5:[3,4],6:[4,6],9:[8,9]}
        @return: {0: 3, 5: 0, 6: 2, 9: 3.5}
        '''
        idleTime = {0: CT}
        num = list(task_time)
        for i in range(len(num)):
            if i < len(num)-1:
                idleTime[num[i]] = task_time[num[i+1]][0] - \
                    task_time[num[i]][1]
            else:
                idleTime[num[i]] = CT - task_time[num[i]][1]
        return idleTime

    def update_task_workstation(nowTask, currentTaskList, oppositeTasksList, time):
        '''
        @description: 根据待分配的task，更新当前边的任务列表，规则基于最小化空闲时间插入任务
        @param {type} 
            @currentTaskList: 当前边上已安排的工作列表，就是task_time_workstation[边]的结果是一个dict，形如{0: [0, 0], 5: [3, 4], 4: [6, 8]}
        @return: 
        '''
        # 当前任务的紧前任务在已分配工作列表中的index
        currentIndex, currentTask = searchInsertedPosition(
            nowTask, currentTaskList)  # 0,0
        oppositeIndex, oppositeTask = searchInsertedPosition(
            nowTask, oppositeTasksList)  # 2,3

        idleTimeList = []  # 为了获得当前任务能够安排的空闲时间列表，比如task 6的time=3，则此时5可以安排在4的后面
        # 查找当前已安排任务后的空闲时间，例如{0:0, 1:1, 5:0,4:5.5}
        idle_time_left = updateIdleTime(currentTaskList, 9999)
        for task, idletime in idle_time_left.items():
            if time[nowTask-1] <= idletime:  # 搜索满足空闲时间的可以插入的位置
                idleTimeList.append([task, idletime])

        # 除了满足空闲时间的约束，还要满足次序关系，即插入的空闲位置不能违反次序关系，下面判断空闲位置是否满足次序关系
        currentLatestTime = currentTaskList[currentTask][1]  # 0
        # 紧前任务的完成时间
        oppositeLatestTime = oppositeTasksList[oppositeTask][1]  # 5
        for task, idletime in iter(idleTimeList):  # task=4, idletime=5.5
            # 可插入的空闲时间的任务的index
            idleIndex = list(currentTaskList).index(task)  # index = 3
            # 可分配空闲位置任务的完成时间
            # idleTime = 7
            idleEarlyCurrentTime = currentTaskList[task][1]
            last_task = list(currentTaskList)[-1]
            if task != last_task:
                # 空闲位置任务的完成时间必须大于在当前边上紧前任务的完成时间，否则开始判断下一个空闲位置
                if idleEarlyCurrentTime >= currentLatestTime:  # 7>=0
                    # 如果可供插入的位置同时满足相反边的紧前任务已完成，则直接插入该贡献位置
                    if idleEarlyCurrentTime >= oppositeLatestTime:  # 7 >= 5
                        start_time = idleEarlyCurrentTime  # 7
                        end_time = start_time+time[nowTask-1]  # 7+1=8
                        currentTaskList = insertDict(currentTaskList, idleIndex+1, nowTask,
                                                     [start_time, end_time])  # 插入，此时左边task列表变为{0：[0,0], 1:[0,2], 5:[3,4], 4:[4,7], 6:[7,8]}
                    else:  # 如果空闲位置任务的完成时间小于相反边的紧前任务完成时间，

                        nextIdleTask = list(currentTaskList)[
                            idleIndex+1]
                        # 则还应该判断扣除相反边紧前任务时间的剩余空闲时间可否插入该任务
                        # 空闲位置的结束时间，也就是空闲位置任务的下一个任务的开始时间，减去相反边紧前任务的完成时间，剩余时间仍然可供插入，则插入
                        if currentTaskList[nextIdleTask][0]-oppositeLatestTime > time[nowTask-1]:
                            start_time = oppositeLatestTime
                            end_time = start_time+time[nowTask-1]
                            currentTaskList = insertDict(currentTaskList, idleIndex+1, nowTask,
                                                         [start_time, end_time])

            else:  # 判断空闲位置任务不能为当前边的最后一个任务，需要进行单独判断
                # 当前边的任务的最大完成时间
                last_time = currentTaskList[last_task][1]
                maxTime = max(oppositeLatestTime, last_time)
                start_time = maxTime  # 3
                end_time = start_time+time[nowTask-1]  # 4
                currentTaskList = insertDict(currentTaskList, idleIndex+1, nowTask,
                                             [start_time, end_time])
        return currentTaskList
    # 解码过程，根据任务序列，产品序列和边序列，获得其makespan

    # 1. 获得seq_products[0]在所有工作站的完工时间，分配的时候还应保证不违反次序关系

    def single(seq_task, seq_side, time):
        '''
        @description: 将单个产品的任务依据分配的工作站，时间，方向以最小空闲时间的方式进行分配
        @param {type} 
        @return: 
        @usage: 
        '''
        setoftasks = getSetOfTasksWithWeight(seq_task,weight)
        task_time_workstation = [{0: [0, 0]} for i in range(2*num_stations)]
        for m in range(num_stations):
            assign = setoftasks[m]
            for t in assign:
                if seq_side[t-1] == 0:
                    task_time_workstation[2*m] = update_task_workstation(
                        t, task_time_workstation[2*m], task_time_workstation[2*m+1], time)
                else:
                    task_time_workstation[2*m+1] = update_task_workstation(
                        t, task_time_workstation[2*m+1], task_time_workstation[2*m], time)
        return task_time_workstation

    def create_task_of_product():
        '''
        @description: 产生每个产品的任务分配解
        @param {type} 
        @return: 
        @usage: 
        '''
        task_of_product = []
        for p in range(num_products):
            task_time_workstation = single(
                seq_tasks[p], seq_sides[p], times[p])
            task_of_product.append(task_time_workstation)
        return task_of_product

    # 2. 比较seq_products[0]在该工作站的完工时间和seq_products[1]在前一工作站的完工时间的大小，取最大值，作为seq_products[1]
    #    的在该工作站的开始时间
    # 3. 迭代，直到所有的信息都被安排，最大值为makespan

    def createGnat(task_of_product):
        '''
        @description: 生成产品的调度图
        @param {type} 
        @return: 
        @usage: 
        '''

        product_time = [[i for i in range(num_stations)]
                        for i in range(num_products)]
        for p in range(num_products):
            for m in range(num_stations):
                leftTime = task_of_product[p][2 *
                                              m][list(task_of_product[p][2*m])[-1]][1]
                rightTime = task_of_product[p][2*m +
                                               1][list(task_of_product[p][2*m+1])[-1]][1]

                process_time = max(leftTime, rightTime)
                product_time[p][m] = process_time
        # product_time = [[3, 0, 4], [0, 3, 4], [3, 5, 0]]
        product_workstation_time = [{0: [0, 0]} for i in range(num_stations)]
        for m in range(1, num_stations+1):
            if m == 1:
                for p in seq_products:
                    t = list(product_workstation_time[m-1])[-1]
                    start_time = product_workstation_time[m-1][t][1]
                    end_time = start_time+product_time[p-1][m-1]
                    product_workstation_time[m-1][p] = [start_time, end_time]
            else:
                for p in seq_products:
                    t = list(product_workstation_time[m-1])[-1]
                    station_time = product_workstation_time[m-1][t][1]
                    precedence_time = product_workstation_time[m-2][p][1]
                    maxtime = max(station_time, precedence_time)
                    start_time = maxtime
                    end_time = start_time+product_time[p-1][m-1]
                    product_workstation_time[m-1][p] = [start_time, end_time]
        makespan = 0
        for seq in product_workstation_time:
            t = list(seq)[-1]
            end_time = seq[t][1]
            if end_time > makespan:
                makespan = end_time
        return makespan

    task_of_product = create_task_of_product()
    makespan = createGnat(task_of_product)
    return makespan

def precedence_repair(seq_task):
    '''将违反次序关系的任务的工作站数字进行调换'''
    for t in range(1, num_tasks+1):
        for i in P[t]:
            if i != 0:
                if seq_task[i-1] > seq_task[t-1]:
                    seq_task[i-1], seq_task[t - 1] = seq_task[t-1], seq_task[i-1]
    return seq_task

def capacity_repair(seq_task, product):
    '''将违反容量约束的任务进行修复'''
    nums = getSetOfTasksWithoutWeight(seq_task)
    capacity = []  # 获取每个工作站上当前的已占用容量
    for i in range(num_stations):
        temp = 0
        for j in nums[i]:
            temp += workspace[product-1][j-1]
        capacity.append(temp)

    # 检查超出容量的工作站上的任务，将次序关系靠后的任务的工作站+1或者-1

    for m in range(1, num_stations+1):
        if m == 1:
            if capacity[m-1] > totalworkspace[m-1]:
                seq_task[nums[m-1][-1]-1] = seq_task[nums[m-1][-1]-1]+1
        elif m > 1 and m < num_stations:
            if capacity[m-1] > totalworkspace[m-1]:
                if random.normalvariate(0, 1) < 0.5:
                    seq_task[nums[m-1][-1]-1] = seq_task[nums[m-1][-1]-1]+1
                else:
                    seq_task[nums[m-1][0]-1] = seq_task[nums[m-1][0]-1]-1
        else:  # m == num_stations
            if capacity[m-1] > totalworkspace[m-1]:
                seq_task[nums[m-1][0]-1] = seq_task[nums[m-1][0]-1]-1
    return seq_task
def repair(seq_task, product):
    '''
    @description: 修复操作，因为任务序列的邻域操作可能导致违反约束关系，因此使用此修复操作

    @param: 表示该产品对应的任务序列 

    @return: [1, 1, 1, 2, 2, 2, 2, 3, 3]

    @usage: print(repair([1, 2, 2, 2, 2, 2, 2, 2, 3],1))
    '''
    original = copy.deepcopy(seq_task)
    repaired = capacity_repair(seq_task,product)
    repaired = precedence_repair(repaired)
    if original == repaired:
        return repaired
    else:
        return repair(repaired,product)

def task_swap(seq_tasks):
    '''
    @description: 对所有产品都应用同一种交换方式
    @param {type}
    @return:
    @usage:
    '''
    for seq_task in seq_tasks:
        nums = []
        setoftasks = getSetOfTasksWithoutWeight(seq_task)

        for k,i in enumerate(setoftasks):
            if len(i) != 0:
                nums.append(k+1)
        
        k1 = random.choice(nums)
        nums.remove(k1)
        k2 = random.choice(nums)
        t1 = random.choice(setoftasks[k1-1])
        t2 = random.choice(setoftasks[k2-1])
        seq_task[t1-1], seq_task[t2-1] = seq_task[t2-1], seq_task[t1-1]
    gg = []
    for k, i in enumerate(seq_tasks):
        n = repair(i,k+1)
        gg.append(n)
    return gg

def task_mutation(seq_tasks):
    for seq_task in seq_tasks:
        num = random.choice([i for i in range(num_tasks)])
        list_station = [i for i in range(1, num_stations+1)]
        # print(seq_task[num])
        list_station.remove(seq_task[num])
        seq_task[num] = random.choice(list_station)
    dd = []
    for k, i in enumerate(seq_tasks):
        n = repair(i,k+1)
        dd.append(n)
    return dd

def precedence_swap(weight):
    '''
    @description: 交换权重，从而修改选择任务的顺序
    @weight {list}: 权重列表 [941, 699, 238, 356, 650, 857, 278, 760, 621, 36, 961, 706]
    @return: [941, 699, 238, 356, 650, 857, 36, 760, 621, 278, 961, 706]
    @usage: 
    '''
    r = random.randint(1,num_tasks)
    t = random.randint(1,num_tasks)
    if t != r:
        weight[r-1],weight[t-1] = weight[t-1],weight[r-1]
    return weight

def side_re_select(seq_sides):
    '''
    @description: 对E-type任务重新选择边
    @param {type} 
    @return: 
    '''
    for side in seq_sides:
        r = random.choice(AE)
        if side[r-1] == 0:
            side[r-1] = 1
        else:
            side[r-1] = 0
    return seq_sides

def product_insert(seq_products:list):
    '''
    @description: 随机选取一个产品，随机插入某个位置，并且与原来的序列不同
    @param {type} 
    @return: 
    @usage: print(product_insert([2,1,3]))
    '''
    old = copy.deepcopy(seq_products)
    product = random.choice(seq_products)
    seq_products.remove(product)
    seq_products.insert(random.randint(0, num_products-1), product)
    if seq_products == old:
        return product_insert(old)
    else:
        return seq_products
# 邻域操作
def neighborhood_search(k, seq_tasks, seq_sides, seq_products, weight):
    '''
    @description: 邻域操作，如果可以写成自适应的邻域操作
    @param {type} 
    @return: 
    @usage: 
    '''
    if k == 1: # 边重定向
        seq_sides = side_re_select(seq_sides)
        return seq_tasks, seq_sides, seq_products,weight
    if k == 2: # 任务重定向：交换操作
        new_tasks1 = task_swap(seq_tasks)
        return new_tasks1, seq_sides, seq_products, weight
    elif k == 3: # 任务重定向：变异操作
        new_tasks2 = task_mutation(seq_tasks)
        return new_tasks2, seq_sides, seq_products, weight
    else:
        new_weight = precedence_swap(weight)
        return seq_tasks, seq_sides, seq_products, new_weight

# 生成邻域
def create_neighborhood(num_b,num_s):
    '''
    @description: 生成平衡个体群和排序个体群
    @param {type} 
    @return: 
    @usage: 
    '''
    balance_neighborhood = []
    sequence_neighborhood = []
    while len(balance_neighborhood) != num_b:
        seq_tasks, seq_sides, seq_products = initSolution()
        init_solution = [seq_tasks,seq_sides]
        if init_solution not in balance_neighborhood:
            balance_neighborhood.append(init_solution)
    while len(sequence_neighborhood) != num_s:
        seq_tasks, seq_sides, seq_products = initSolution()
        if seq_products not in sequence_neighborhood:
            sequence_neighborhood.append(seq_products)
    return balance_neighborhood,sequence_neighborhood

def local_search(seq_tasks, product):
    '''
    @description: 局域搜索
    @param {type} 
    @return: 
    @usage: 
    '''
    
    pass

def main():
    # 设置迭代次数
    t = 1
    tMax = 100000
    # 生成邻域
    pop_b,pop_s = create_neighborhood(9,3)
    # 暂且使用统一的权重
    weight = getWeight()

    # 生成最优解
    best_b = 0
    best_s = 0
    best_fitness = 500
    for balance in pop_b:
        for sequence in pop_s:
            fitness = decode(balance[0],balance[1],sequence,times,weight)
            if fitness < best_fitness:
                best_fitness = fitness
                best_b = balance
                best_s = sequence
    # 生成邻域中每个个体的适应度值
    b_fit = [i for i in range(9)]
    s_fit = [i for i in range(3)]
    for balance in pop_b:
        fitness = decode(balance[0], balance[1], best_s, times, weight)
        b_fit.append(fitness)
    for sequence in pop_s:
        fitness = decode(best_b[0], best_b[1], sequence, times, weight)

    while t < tMax:
        if t % 1000 == 0:
            for balance in pop_b:
                for sequence in pop_s:
                    fitness = decode(balance[0], balance[1], sequence, times, weight)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_b = balance
                        best_s = sequence
            print(best_fitness)
        random.shuffle(weight)
        
        # 对平衡邻域进行搜索
        # 从最佳组合中确定产品序列，进行平衡解的迭代
        for b in range(len(pop_b)):
            # 当前适应度值
            local_fitness = b_fit[b]
            balance = pop_b[b]
            # 邻域搜索
            k = random.randint(1,3)
            a,b,c,weight = neighborhood_search(k, balance[0], balance[1], best_s, weight)
            
            fitness = decode(balance[0], balance[1], best_s, times, weight)

            # 更新pop-b邻域
            
            # 如果发现更优的结果，则更新当前解为新的解
            if fitness < local_fitness:
                balance[0] = a
                balance[1] = b
                b_fit[b] = fitness

        # 排序解的搜索
        # 从最佳个体中固定平衡解，对排序解进行邻域搜索
        for s in range(len(pop_s)):
            local_fitness = s_fit[s]
            sequence = pop_s[s]

            random.shuffle(sequence)
            fitness = decode(best_b[0], best_b[1], sequence, times, weight)
            if fitness < local_fitness:
                s_fit[s] = fitness
                pop_s[s] = sequence

        t+=1

    print(best_fitness)

main()
