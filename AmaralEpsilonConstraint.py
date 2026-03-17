import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys, getopt
import numpy as np

# Function to get the input files for each objective form directory and considering the time limmit to 3 hours
def argument_parser(argv):
    inputfile = ""
    inputfile2 = ""
    timelimit = 10800  # Default time limit of 3 hours
    
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

#Function to read instances data
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
    l, c1 = read_instance(filename1) ,# length is read from first file, contain the first objective flow values
    _, c2 = read_instance(filename2)
    n = len(l)
    return c1, c2, l , n

#Function to find the permutation related to each optimal solution
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
    #Collect all active triples
    triples = []
    for i in range(0,n):
        for j in range(i+1,n):
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

# Example usage, the main model of Amaral
def Amaral_Model(c1, c2, l, n, v, time_limit):         
    
    model_time = time.time()
    # Create a new GP instance
    m = gp.Model("SRFLP")

    # Add variables
    x = m.addVars(n, n, n, vtype=GRB.BINARY, name="x",  lb=0.0, ub=1.0)
    

    # Add constraint:
    m.addConstrs((x[i, j, k] + x[i, k, j] + x[j, k, i] == 1 for i in range(0,n) for j in range(i+1,n) for k in range(j+1,n) if (i < j and j < k)), name='7')

    m.addConstrs((-x[i, j, d] + x[j, k, d] + x[i, k, d] >= 0 for i in range(0,n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.1')
    m.addConstrs((x[i, j, d] - x[j, k, d] + x[i, k, d] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.2')
    m.addConstrs((x[i, j, d] + x[j, k, d] - x[i, k, d] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.3')

    m.addConstrs((x[i, j, d] + x[j, k, d] + x[i, k, d] <= 2 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='9')
    
    
    # Set objective_1 as main and esilon_constraint for objective_2 
    constant_1 = gp.quicksum(float(c1[i+1, j+1])*(l[i] + l[j]) for i in range(0,n-1) for j in range(i+1,n))/2
    objective_1 = gp.quicksum((float(c1[i+1, j+1])*l[k]*x[i,j,k]) for i in range(0,n-1) for j in range(i+1,n) for k in range(n) if (k != i and k != j))
  
    constant_2 = gp.quicksum(c2[i+1, j+1]*(l[i] + l[j]) for i in range(0,n-1) for j in range(i+1,n))/2
    objective_2 = gp.quicksum(c2[i+1, j+1] * l[k] * x[i,j,k] for i in range(0,n-1) for j in range(i+1,n) for k in range(n) if (k != i and k != j))
    
    obj1 = objective_1 + constant_1
    obj2 = objective_2 + constant_2
    
    m.setObjective(obj1, GRB.MINIMIZE)
    
    
    # the epsilon-constriant
    epsilon = 1
    m.addConstr(obj2 <= v - epsilon)
    
    time_remain = time_limit - (time.time() - model_time)
    
    if time_remain < 0:
        time_remain = 0

    try:
        m.setParam('TimeLimit', time_remain)
        m.setParam(GRB.Param.Threads, 1)
        m.optimize()
        BNB_Nodes = m.NodeCount

    except Exception as e:
        print("Error during optimization:", e)

    perm = None
    if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
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
        return None, None, None, GRB.INFEASIBLE, None, None, None, None, None, None
    elif m.Status == GRB.TIME_LIMIT:
        print("Time Limit reached")
        m.dispose()
        return None, None, None, GRB.INFEASIBLE, None, BNB_Nodes, perm, best_obj, best_bound, gap
    else:
        obj1 = m.ObjVal
        obj2_val = obj2.getValue()
        obj2_K = objective_2.getValue()
        status = m.Status
        m.dispose()
        return obj1, obj1, obj2_K, status, obj2_val, BNB_Nodes, perm, best_obj, best_bound, gap


def main(argv):
    
    inputfile, inputfile2, timelimit = argument_parser(argv)
    
    output_file_basename = inputfile.split("/")[-1]+"_E"
    
    print("Input file 1 is: ", inputfile)
    print("Input file 2 is: ", inputfile2)
    print("\n")

    c1, c2, l, n = process_instance(inputfile, inputfile2)
    # the value of objective 2
    v = 99999999
    Start_Time = time.time()
    objective1 = []
    objective2 = []
    stats_list = []
    NDPoints = 0

    time_remain = int(timelimit)
    time_out = False

    obj_val, obj1, objective_2, status, obj2, BNB, perm, best_obj, best_bound, gap = Amaral_Model(c1,c2,l,n,v, time_remain)
        
    Total_Time_Point = time.time() - Start_Time
    time_remain -= (time.time() - Start_Time)
    
    while status != GRB.INFEASIBLE and time_remain > 0:
        Start_Time_Point = time.time() 
        
        #v = objective_2
        v = obj2
        NDPoints += 1
        print("flow1: ", obj1)
        print("flow2: ", obj2)
        print("ND-Point:", NDPoints)
        objective1.append(obj1)
        objective2.append(obj2)

        
        stats_list.append({'OBJ1': round(obj1, 3),
                           'OBJ2': round(obj2, 3),
                           'BNB_Nodes_MIP': BNB,
                           'Total_Time_MIP[s]' : round(Total_Time_Point, 3),
                           'Permutation': "-".join(map(str, perm)),
                           'Best_Obj_MIP': round(best_obj, 3),
                           'Best_Bound_MIP': round(best_bound, 3),
                           'Gap_MIP%': gap})
        print("\n-----------------------------------------")
        try:
            obj_val, obj1, objective_2, status, obj2, BNB,perm, best_obj, best_bound, gap = Amaral_Model(c1,c2,l,n,v, time_remain)
        except:
            status == GRB.INFEASIBLE
            
        End_time = time.time()
        Total_Time_Point = End_time - Start_Time_Point
        time_remain -= Total_Time_Point
        
    TotalTime = time.time() - Start_Time
        
    print("\n")
    print("Total running time:", TotalTime)
        
    
    #Table of final results 
    to_remove = []
    for i in range(0, len(stats_list)-1):
        if stats_list[i]["OBJ1"] == stats_list[i+1]["OBJ1"]:
            to_remove.append(stats_list[i])
            NDPoints-=1
    for i in to_remove:
        stats_list.remove(i)
        
    print("STAH, Instance, #departments, #ND-Points, t-MIP")
    print(f"STAT,{output_file_basename},{n},{NDPoints},{round(TotalTime, 4)}")
        
    dfResults = pd.DataFrame(stats_list)
    print(dfResults)
    dfResults.to_latex("./outputE/" + output_file_basename + ".tex", index=False)
        
    #Plotting      
    plt.scatter(dfResults["OBJ1"], dfResults["OBJ2"], color='blue', label="Non-Dominated Points", s=8)
    plt.legend()
    filename = "./outputE/" + output_file_basename+".png"
    print(filename)
    plt.savefig(filename) 
    plt.close()
    
    csv_file = "./outputE/" + output_file_basename+".csv"
    dfResults[["OBJ1", "OBJ2"]].to_csv(csv_file, index=False)
    
    #Save all results in a unique fill
    with open("./results/" + "results_Epsilon" + ".tex", 'a') as file:
        if file.tell() == 0:  # This checks if the file is empty
            file.write("Instance,#departments,#ND-points,t-MIP [s], Last-Point-Gap%\n")  # CSV header
        file.write(f"{output_file_basename},{n},{NDPoints},{round(TotalTime, 4)}, {0.00}\n")
        
if __name__ == "__main__":
    main(sys.argv[1:])
