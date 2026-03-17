import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys, getopt
import numpy as np
from itertools import combinations

def argument_parser(argv):
    inputfile = ""
    inputfile2 = ""
    timelimit = 10800
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--ifile", help="Input file 1")
        parser.add_argument("-j", "--jfile", help="Input file 2")
        parser.add_argument("-t", "--timelimit", help="Time Limit")
        
        args = parser.parse_args()
        
        if args.ifile:
            inputfile = "instance/"+args.ifile
        if args.jfile:
            inputfile2 = "instance/"+args.jfile
        if args.timelimit:
            timelimit = args.timelimit
            
    except getopt.GetoptError as error:
        print("Error", error)
        sys.exit(2)
    
    return inputfile, inputfile2, timelimit

#Function to read the files contain the value s
def read_instance(filename):
    with open(filename, 'r') as file:
        dimension = int(file.readline().strip())
        length = list(map(int, file.readline().strip().split(',')))
        
        cost_matrix = []
        for _ in range(dimension):
            row = list(map(int, file.readline().strip().split(',')))
            cost_matrix.append(row)
    
    costs = {}
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                costs[(i+1, j+1)] = cost_matrix[i][j]  
    
    return length, costs

def process_instance(filename1, filename2):
    l , c1 = read_instance(filename1)
    _ , c2 = read_instance(filename2)
    n = len(l)
    return c1, c2, l , n

def getStablePermutation(x_vars, n, tol=1e-6):
    """
    Reconstructs a stable department sequence from binary x[i,j,k] solution.
    Ensures k is between i and j for all x[i,j,k] == 1 triples.
    
    Args:
        x_vars: Gurobi x[i,j,k] variables
        n: number of departments
        tol: tolerance to consider x == 1
    
    Returns:
        permutation: list of department indices (1-based)
    """
    # Collect all active triples
    triples = []
    for i in range(0,n):
        for j in range(i+1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                if x_vars[i, j, k].X > 1 - tol:
                    triples.append((i, j, k))
    
    # Start with arbitrary sequence if triples exist
    if triples:
        i, j, k = triples[0]
        seq = [i, k, j]
    else:
        seq = list(range(n))
    
    changed = True
    while changed:
        changed = False
        for i, j, k in triples:
            # ensure i and j are in sequence
            if i not in seq:
                seq.insert(0, i)
                changed = True
            if j not in seq:
                seq.append(j)
                changed = True
            if k not in seq:
                # put k between i and j
                idx_i = seq.index(i)
                idx_j = seq.index(j)
                left = min(idx_i, idx_j)
                right = max(idx_i, idx_j)
                seq.insert(right, k)
                changed = True
            else:
                # if k already in seq, check if it is between i and j
                idx_i = seq.index(i)
                idx_j = seq.index(j)
                idx_k = seq.index(k)
                left = min(idx_i, idx_j)
                right = max(idx_i, idx_j)
                if not (left < idx_k < right):
                    # move k between i and j
                    seq.pop(idx_k)
                    seq.insert(right if idx_k > right else left + 1, k)
                    changed = True
    
    # Ensure all departments are included
    for d in range(n):
        if d not in seq:
            seq.append(d)
    
    # Convert to 1-based indexing
    permutation = [d + 1 for d in seq]
    return permutation


# Example usage
def Amaral_Model(c1, c2, l, n, w, time_limit):  
    
    model_time = time.time()       
    # Create a new GP instance
    m = gp.Model("SRFLP")

    # Add variables, the variable is continnous here
    x = m.addVars(n, n, n, vtype=GRB.CONTINUOUS, name="x",  lb=0, ub=1)

    # Add constraint:
    m.addConstrs((x[i, j, k] + x[i, k, j] + x[j, k, i] == 1 for i in range(0,n) for j in range(i+1,n) for k in range(j+1,n) if (i < j and j < k)), name='7')

    m.addConstrs((-x[i, j, d] + x[i, k, d] + x[j, k, d] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.1')
    m.addConstrs((x[i, j, d] - x[i, k, d] + x[j, k, d] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.2')
    m.addConstrs((x[i, j, d] + x[i, k, d] - x[j, k, d] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.3')

    m.addConstrs((x[i, j, d] + x[i, k, d] + x[j, k, d] <= 2 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(k+1,n) if (i < j and j < k and k < d and d!=i and d!=j and d!=k)), name='9')
    
    constant_1 = gp.quicksum(float(c1[i+1, j+1])*(l[i] + l[j]) for i in range(0,n-1) for j in range(i+1,n))/2
    objective_1 = gp.quicksum((float(c1[i+1, j+1]) * l[k] * x[i, j, k]) for i in range(0,n-1) for j in range(i+1,n) for k in range(0,n) if (k != i and k != j))
  
    constant_2 = gp.quicksum(float(c2[i+1, j+1]) *(l[i] + l[j]) for i in range(0,n-1) for j in range(i+1,n))/2
    objective_2 = gp.quicksum((float(c2[i+1, j+1]) * l[k] * x[i, j, k]) for i in range(0,n-1) for j in range(i+1,n) for k in range(0,n) if (k != i and k != j))
    
    obj1 = objective_1 + constant_1
    obj2 = objective_2 +constant_2
    
    #Write the objective function as a weighted sum shape with "w" which shows the weight of each objective
    m.setObjective(w[0] * obj1 + w[1] * obj2, GRB.MINIMIZE)
    
    time_remain = time_limit - (time.time() - model_time)
    if time_remain < 0:
        time_remain = 0
        
    try:
        m.setParam('TimeLimit', time_remain)
        m.setParam(GRB.Param.Threads, 1)
        m.optimize()
        LP_iter = m.IterCount
        
        # Adding cutting planes if non-integer solution found
        tolerance  = 0.000001
        beta = 6
        non_integer_detected = any(abs(v.X) > tolerance and abs(v.X - 1) > tolerance for v in m.getVars())
        if non_integer_detected:
            sets_S = list(combinations(range(n), beta))
            for S in sets_S:
                for r in S:
                    # Partition S \ {r} into S1 and S2
                    S_minus_r = [i for i in S if i != r]
                            
                    # Iterate over all possible S1 subsets of size beta // 2
                    S1 = S_minus_r[:beta // 2]
                    S2 = S_minus_r[beta // 2:]  
                                
                    lhs_1 = gp.quicksum(x[p, q, r] for p in S1 for q in S1 if p < q)
                    lhs_2 = gp.quicksum(x[p, q, r] for p in S2 for q in S2 if p < q)
                    rhs = gp.quicksum(x[p, q, r] for p in S1 for q in S2 if p < q)

                    # Add the inequality constraint
                    m.addConstr(lhs_1 + lhs_2 + rhs <= 6, name ='extra1')
        
            time_remain = time_remain - (time.time() - model_time)
            if time_remain < 0:
                time_remain = 0
            m.setParam('TimeLimit', time_remain)         
            m.setParam(GRB.Param.Threads, 1)        
            m.optimize()
            LP_iter = m.IterCount

            
    except Exception as e:
        print("There is an error: ", e)
         
    perm = None
    best_obj = None
    best_bound = None
    gap = None
    
    if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        try:
            perm = getStablePermutation(x, n)
            best_obj = m.ObjVal
            best_bound = m.ObjBound  # Lower/upper bound depending on min/max
            gap = m.MIPGap * 100 if m.SolCount > 0 else 0.0
        except Exception as e:
            print("Could not extract permutation:", e)
                
    # Get solution
    if m.Status == GRB.INFEASIBLE:
        print("No solution found")
        m.dispose()
        return None, None, None, GRB.INFEASIBLE, None , None, None, None, None
    elif m.Status == GRB.TIME_LIMIT:
        print("Time Limit reached")
        m.dispose()
        return None, None, None, GRB.INFEASIBLE, LP_iter, perm, best_obj, best_bound, gap
    else:
        obj_val = m.objVal
        obj1 = obj1.getValue()
        obj2 = obj2.getValue()
        status = m.Status
        m.dispose()
        return obj_val, obj1, obj2, status, LP_iter, perm, best_obj, best_bound, gap
 

def main(argv):
     
    inputfile, inputfile2, timelimit = argument_parser(argv)
    
    output_file_basename = inputfile.split("/")[-1] + "_WS"
    
    print("Input file 1 is: ", inputfile)
    print("Input file 2 is: ", inputfile2)
    print("\n")

    c1, c2, l, n = process_instance(inputfile, inputfile2)
    
    # global lists and parameters       
    Start_Time = time.time()
    y_list = []
    stats_list = []
    epsilon = []
    NDPoints = 0
    time_remain = int(timelimit)
    
    # first point with large w for obj1
    Start_Time_Point = time.time()
    
    w1 =(100000,0.000001)
    objval1, y_a1, y_a2, status, LP_iter, perm, best_obj, best_bound, gap = Amaral_Model(c1,c2,l,n,w1, time_remain)
    Total_Time_Point = time.time() - Start_Time_Point
    time_remain -= Total_Time_Point
    y_a =(round(y_a1, 5), round(y_a2, 5))
    y_list.append(y_a)
    epsilon.append(y_a) 
    NDPoints += 1 
    print("First Node:", y_a)
    print("ND-Point:", NDPoints)
    stats_list.append({'OBJ1': round(y_a1, 3),
                       'OBJ2': round(y_a2, 3),
                       'LP_iterations': LP_iter,
                       'Total_Time_MIP[s]' : round(Total_Time_Point, 3),
                       'Permutation':  "-".join(map(str, perm)),
                       'Best_Obj': round(best_obj, 3), 
                       'Best_Bound': round(best_bound, 3),
                       'Gap[%]': gap})
    print("\n-----------------------------------------------------------------------------------------------------------------")
  
    #second point with large w for obj2
    Start_Time_Point = time.time()
    
    w2 = (0.000001, 100000) 
    obj_val2, y_b1, y_b2, status, LP_iter, perm, best_obj, best_bound, gap = Amaral_Model(c1,c2,l,n,w2, time_remain)
    Total_Time_Point = time.time() - Start_Time_Point
    time_remain -= Total_Time_Point
    y_b = (round(y_b1, 5), round(y_b2, 5))
    y_list.append(y_b)
    epsilon.append(y_b) 
    print("Second Node:", y_b)
    NDPoints += 1
    print("ND-Point:", NDPoints)
    stats_list.append({'OBJ1': round(y_b1, 3), 
                       'OBJ2': round(y_b2, 3),
                       'LP_iterations': LP_iter,
                       'Total_Time_MIP[s]' : round(Total_Time_Point, 3),
                       'Permutation':  "-".join(map(str, perm)),
                        'Best_Obj': round(best_obj, 3), 
                        'Best_Bound': round(best_bound, 3),
                       'Gap[%]': gap})
    print("\n------------------------------------------------------------------------------------------------------------")
    
    #start of formula
    while True:
        Start_Time_Point = time.time()    
        w = (epsilon[0][1]- epsilon[1][1], epsilon[1][0] - epsilon[0][0])
        print('Weight:',w)
        obj_val, obj1, obj2, status,LP_iter, perm, best_obj, best_bound, gap= Amaral_Model(c1,c2,l,n,w, time_remain)
        if status == GRB.INFEASIBLE and time_remain > 0:
            break
        Total_Time_Point = time.time() - Start_Time_Point
        y_c = (round(obj1, 5), round(obj2, 5))
        #y_c = (obj1,obj2)
        print("The new node:", y_c)
        if y_c not in y_list and obj_val < (w[0] * epsilon[0][0] + w[1] * epsilon[0][1]):
            y_list.append(y_c)
            NDPoints += 1
            stats_list.append({'OBJ1': round(obj1, 3),
                               'OBJ2': round(obj2, 3),
                               'LP_iterations': LP_iter, 
                               'Total_Time_MIP[s]' : round(Total_Time_Point, 3),
                               'Permutation':  "-".join(map(str, perm)),
                               'Best_Obj': round(best_obj, 3), 
                               'Best_Bound': round(best_bound, 3),
                               'Gap[%]': gap})
            epsilon.insert(1,y_c)
            print("epsilon_list:",epsilon)  
        else:
            epsilon.pop(0)
            print("epsilon_list:",epsilon)     
               
        print("ND-Point:", NDPoints)
        print('ylist:',y_list)
        if len(epsilon) < 2:  # For example, stop if epsilon has fewer than 2 points
           break 
        time_remain -= Total_Time_Point
        print("\n-------------------------------------------------------------------------------------------------------------")
    
    TotalTime = time.time() - Start_Time
        
    print("\n")
    print("Total running time:", TotalTime)
        
    
    #Table of final results 
    dfResults = pd.DataFrame(stats_list)
    print(dfResults)
    dfResults.to_latex("./output/" + output_file_basename + ".tex", index=False)

      
    #Plotting    
    plt.scatter(dfResults["OBJ1"], dfResults["OBJ2"], color='green', label="Supported Points", s=8)
    plt.legend()
    filename = "./output/" + output_file_basename+".png"
    print(filename)
    plt.savefig(filename) 
    plt.close()
    
    csv_file = "./output/" + output_file_basename+".csv"
    dfResults[["OBJ1", "OBJ2"]].to_csv(csv_file, index=False)
    
    #Save all results in a unique fill
    with open("./results/" + "results_WS" + ".tex", 'a') as file:
        if file.tell() == 0:  # This checks if the file is empty
            file.write("Instance,#departments,#ND-points,t-MIP [s], Last-Point-Gap% \n")  # CSV header
        file.write(f"{output_file_basename},{n},{NDPoints},{round(TotalTime, 4)}, {0.00}\n")
        
        
if __name__ == "__main__":
    main(sys.argv[1:])
