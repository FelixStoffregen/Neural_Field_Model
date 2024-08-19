from .A0Basic_Functions import *
import numpy as np


######################################
# Local Stability Calculations
######################################

def J_matrix(param, ue_eq, ui_eq):
    dPe = dP_e(ue_eq - param["theta_e"], param)
    dPi = dP_i(ui_eq - param["theta_i"], param)
    nu_ee = param["nu_ee"]
    nu_ei = param["nu_ei"]
    nu_ie = param["nu_ie"]
    nu_ii = param["nu_ii"]
    tau_i =  param["tau_i"]
    tau_e =  param["tau_e"]
    tau_alpha =  param["tau_alpha"]
    g = param["g"]
    H = drelu(ue_eq)

    J = [[tau_e**(-1)*(-1 + nu_ee * dPe),         tau_e**(-1)*(-nu_ie * dPi) ,     tau_e**(-1)*(-g * H)],
            [tau_i**(-1) * nu_ei * dPe,   -tau_i**(-1) * (1 + nu_ii * dPi),             0],
            [tau_alpha**(-1),           0,                            -tau_alpha**(-1)]]
    return np.array(J)

def J_polynomial_coefficients(param, ue_eq, ui_eq):
    J = J_matrix(param, ue_eq, ui_eq)
    return np.poly(J)

def J_characteristic_polynomial(x, param, ue_eq, ui_eq):
    q3, q2, q1, q0 = J_polynomial_coefficients(param, ue_eq, ui_eq)
    return q3 * x**3 + q2 * x**2 + q1 * x + q0

def J_eigenvalues(param, ue_eq, ui_eq):
    J = J_matrix(param, ue_eq, ui_eq)
    return np.linalg.eigvals(J)

def local_stability(param, ue_eq, ui_eq):
    eig_values = J_eigenvalues(param=param, ue_eq=ue_eq, ui_eq=ui_eq)
    for value in eig_values:
        if value.real > 0:
            return False
    return True

##############################################################
# Num Solver for Trajectory
##############################################################

def Runge_Kutta_step(u: np.ndarray,
                     dt: float,
                     rhs: callable,
                     param: dict,
                     I):
    
    """
    Classical Runge Kutta method of order 4.

    Args:
        u (np.ndarray): Two dimensional array
                        u[0]: Membrane Potential of the excitatory population at time t
                        u[1]: Membrane Potential of the inhibitory population at time t
        dt (float): timestep of the simulation
        rhs (callable): right hand side of the differential equation (independent from t)
        param (dict): Parameters of the simulation given as a dictionary

    Returns:
        np.ndarray: Membrane potential of excitatory and inhibitory population in the next timestep
    """

    K1 = rhs(u=u, param=param, I=I)
    K2 = rhs(u = u + 0.5 * dt * K1, param = param, I=I)
    K3 = rhs(u = u + 0.5 * dt * K2, param = param, I=I)
    K4 = rhs(u = u + dt * K3, param = param, I=I)
    return u + dt/6 * (K1 + 2 * K2 + 2 * K3 + K4)


def num_solver(u0, 
               param,
               rhs = local_rhs,
               dt: float=1e-2, 
               start_time: float=0., 
               stop_time: float=1., 
               step_solver: callable=Runge_Kutta_step,
               I=[0,0]):
    
    """
    Numerical simulation of the Wilson-Cowan equation in the local limit.

    Args:
        u0 (np.ndarray): Starting values given as two dimensional array
                        u[0]: Membrane Potential of the excitatory population at starting time
                        u[1]: Membrane Potential of the inhibitory population at starting time
        param (dict, optional): paramters of the simulation.
        dt (float, optional): timestep. Defaults to 1e-2.
        start_time (float, optional): starting time of the simulation. Defaults to 0..
        stop_time (float, optional): stopping time of the simulation. Defaults to 1..
        rhs (callable, optional): right hand side of the equation. Defaults to local_rhs.
        step_solver (callable, optional): numerical method to calculate the next timestep. Defaults to Runge_Kutta_step.

    Returns:
       np.ndarray, np.ndarray: Membrane potential of the inhibitory and exicatory population and time discretization.
    """

    t = np.arange(start_time, stop_time, dt)
    u = np.zeros((t.size, 3))
    u[0] = u0
    for i in range(len(u)-1):
        u[i+1] = step_solver(u = u[i], dt = dt, rhs=rhs, param=param, I=I)
    return u, t