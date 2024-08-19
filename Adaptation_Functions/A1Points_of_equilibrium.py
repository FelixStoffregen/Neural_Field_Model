from .A0Basic_Functions import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root

import matplotlib.pyplot as plt
import time as time


#####################################################
# Computing the trajectories g_e/g_i and their intersection
#####################################################

def remove_infinity(arr, corresponding_arr):
    # Step 1: Find indices "infinitys"
    inf_indices = np.where(np.abs(arr) > 100)[0]
    if len(inf_indices) == 0:
        return arr, corresponding_arr

    # Step 2: Identify start and end indices of consecutive infinity intervals
    diff = np.diff(inf_indices)
    boundaries = np.where(diff > 1)[0]
    start_indices = inf_indices[np.append(0, boundaries + 1)]
    end_indices = inf_indices[np.append(boundaries, len(inf_indices) - 1)]

    # Step 3: Mark indices that are not at the boundaries for removal
    # We use NaN as a placeholder for infinity to remove; ensure arr is float for NaN support
    for start, end in zip(start_indices, end_indices):
        if end - start > 1:  # More than one zero in the interval
            arr[start+1:end] = np.nan  # Mark inner infinity

    # Step 4: Remove marked values, convert back to original dtype if needed
    filtered_arr = arr[~np.isnan(arr)]
    filtered_corresponding_arr = corresponding_arr[~np.isnan(arr)]
    return filtered_arr, filtered_corresponding_arr

def g_e_line(param, I_i, ui_lim=(-0.5, 0.5), du=5e-5):
    ui_min, ui_max = ui_lim
    u_i = np.arange(ui_min, ui_max, du) 
    u_e = g_e(u_i, param, I_i)
    u_e, u_i = remove_infinity(u_e, u_i)
    return u_e, u_i

def g_i_line(param, I_e, ue_lim=(-0.5, 0.5), du=5e-5):
    ue_min, ue_max = ue_lim
    ue = np.arange(ue_min, ue_max, du)
    ui = g_i(ue, param, I_e)
    ui, ue = remove_infinity(ui, ue)
    return ue, ui

def get_intersection(ue1, ui1, ue2, ui2, warning=True):
    # Step 1: Interpolate trajectories
    f1 = interp1d(ue1, ui1, kind='linear')
    f2 = interp1d(ue2, ui2, kind='linear')
    

    # Step 2: Numerically find intersections (simple approach)
    ue_min = max(min(ue1), min(ue2))
    ue_max = min(max(ue1), max(ue2))
    ue_values = np.linspace(ue_min, ue_max, 1000)  # Adjust granularity as needed
    diff = f1(ue_values) - f2(ue_values)
    sign_changes = np.diff(np.sign(diff))

    intersection_indices = np.where(sign_changes != 0)[0]
    intersection_ue = ue_values[intersection_indices]
    intersection_ui = f1(intersection_ue)
    
    tol=0.1
    if warning:
        for ue, ui in zip(intersection_ue, intersection_ui):
            if np.abs(ue - ue_min) < tol or np.abs(ue - ue_max) < tol:
                print("It is possible, that points of equilibrium have been missed:")
                print(f"We search for ue in [{ue_min:.2f}, {ue_max:.2f}]:")
                print(f"Point of Equilibrium close to boundary: ue: {ue:.2f}, ui: {ui:.2f}")
    return np.sort(intersection_ue), np.sort(intersection_ui)

#####################################################
# Compute the points of Equilibrium
#####################################################

def root_rhs(u, param, I_e, I_i):
    # interpret rhs of the local limit as function of u=(u_e, u_i)
    u_e, u_i = u
    du_e = - u_e + param["nu_ee"] * P_e(u_e - param["theta_e"], param) - param["nu_ie"] * P_i(u_i - param["theta_i"], param) - param["g"] * relu(u_e) + I_e
    du_i = - u_i + param["nu_ei"] * P_e(u_e - param["theta_e"], param) - param["nu_ii"] * P_i(u_i - param["theta_i"], param) + I_i
    return np.array([du_e, du_i])

def root_rhs_Jacobian(u, param):
    # jacobian of the rhs of the local limit as function of u=(u_e, u_i)
    u_e, u_i = u
    d_ee = - 1 + param["nu_ee"] * dP_e(u_e - param["theta_e"], param) - param["g"] * drelu(u_e)
    d_ii = - 1 - param["nu_ii"] * P_i(u_i - param["theta_i"], param)
    d_ei = - param["nu_ie"] * dP_i(u_i - param["theta_i"], param)
    d_ie = + param["nu_ei"] * dP_e(u_e - param["theta_e"], param)
    return np.array([[d_ee, d_ei],
                     [d_ie, d_ii]])


def get_adapt_equilibrium_points(param,
                                 I_e=0, I_i=0,
                                 ue_lim=(-0.9, 0.9), ui_lim=(-0.9, 0.9),
                                 du=5e-4,
                                 iterations=5,
                                 perturbation_strength=1e-2,
                                 warning=True,
                                 improve_solution=True):
    
    ue1, ui1 = g_e_line(param, I_i, ui_lim=ui_lim, du=du)
    ue2, ui2 = g_i_line(param, I_e, ue_lim=ue_lim, du=du)
    ue, ui = get_intersection(ue1, ui1, ue2, ui2, warning=warning)

    if improve_solution:
        # Improve solution using the roots function from scipy
        for i in range(len(ue)):
            x0 = np.array([ue[i], ui[i]])
            current_sol = x0
            current_error = np.abs(np.sum(root_rhs(param=param, u=x0, I_e=I_e, I_i=I_i)))
            # Try different starting values for root and pick the best solution.
            for _ in range(iterations):
                if current_error < 1e-13:
                    break
                # Small perturbation on starting value, to improve solution
                perturbation = (2 * np.random.rand(2, ) - 1) * perturbation_strength
                starting_value = x0 + perturbation
                # First try always without any perturbation
                if _ == 0:
                    starting_value = x0

                solution = root(fun = lambda u: root_rhs(u, param, I_e, I_i),
                                x0 = starting_value,
                                jac = lambda u: root_rhs_Jacobian(u, param))
                
                # Check if the found solution is closest to an already found solution, or the x0
                dist = np.abs(np.sum(x0-solution.x))
                unique = True
                for j in range(len(ue)):
                    if np.abs(np.sum(np.array([ue[j], ui[j]])-solution.x)) < dist:
                        unique = False
                # If a unique solution is found, check if it is better than the previously calculated solutinos 
                if unique:
                    new_error = np.abs(np.sum(root_rhs(param=param, u=solution.x, I_e=I_e, I_i=I_i)))
                    # new improved solution found => update current solution
                    if new_error < current_error:
                        current_error = new_error
                        current_sol = solution.x

            ue[i], ui[i] = current_sol
    return ue, ui