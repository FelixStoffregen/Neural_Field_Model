from .A0Basic_Functions import *
import numpy as np

####################################################
# Right Hand Side of the spatial system
####################################################

def spatial_rhs(u_e,
                u_i,
                alpha,
                I_e,
                I_i,
                w_mn,
                param, 
                P_e: callable=P_e,
                P_i: callable=P_i):
    """
    Computes the right hand side voltage-based neural fiel model with linear adaptation.
    It is computed using a spatial connectivity matrix.

    Args:
        u_e (np.ndarray): vector of excitatory voltages at time t.
        u_i (np.ndarray): vector of inhibitory voltages at time t.
        alpha (np.ndarray): adaptation of the excitatory population
        I_e (float): external input in the excitatory population
        I_i (float): external input in the inhibitory population
        w_mn (dict): dictionary of spatial connectivity matricies
        param (dict): parameters of simulation
        P_e (callable, optional): excitatory firing rate function. Defaults to P_e.
        P_i (callable, optional): inhibitory firing rate function. Defaults to P_i.

    Returns:
        du_e, du_i: right hand side at timestep t.
    """

    p_e = P_e(u_e - param["theta_e"], param)
    p_i = P_i(u_i - param["theta_i"], param)
    
    du_e = - u_e + w_mn["ee"].dot(p_e) - w_mn["ie"].dot(p_i) + I_e - param["g"]* relu(alpha)
    du_i = - u_i + w_mn["ei"].dot(p_e) - w_mn["ii"].dot(p_i) + I_i
    dalpha = - alpha + u_e

    du_i *= 1/param["tau_i"]
    du_e *= 1/param["tau_e"]
    dalpha *= 1/param["tau_alpha"]

    return du_e, du_i, dalpha


def spatial_rhs_fft(u_e,
                u_i,
                alpha,
                I_e,
                I_i,
                w_mn,
                param,
                P_e: callable=P_e,
                P_i: callable=P_i):

    """
    Computes the right hand side voltage-based neural fiel model with linear adaptation.
    The convolution is computed using the FFT.
    
    Args:
        u_e (np.ndarray): vector of excitatory activities at time t.
        u_i (np.ndarray): vector of inhibitory activities at time t.
        w_mn (dict): dictionary of spatial connectivity matricies
        param (dict): parameters of simulation
        P_e (callable, optional): excitatory firing rate function. Defaults to P_e.
        P_i (callable, optional): inhibitory firing rate function. Defaults to P_i.

    Returns:
        du_e, du_i: right hand side at timestep t.
    """

    p_e_hat = np.fft.rfft(P_e(u_e - param["theta_e"], param))
    p_i_hat = np.fft.rfft(P_i(u_i - param["theta_i"], param))

    n = len(u_e)
    conv_ee = np.real(np.fft.irfft(p_e_hat * w_mn["ee"], n=n))
    conv_ie = np.real(np.fft.irfft(p_i_hat * w_mn["ie"], n=n))
    conv_ei = np.real(np.fft.irfft(p_e_hat * w_mn["ei"], n=n))
    conv_ii = np.real(np.fft.irfft(p_i_hat * w_mn["ii"], n=n))
    
    du_e = - u_e + conv_ee - conv_ie - param["g"] * relu(alpha) + I_e
    du_i = - u_i + conv_ei - conv_ii + I_i
    dalpha = -alpha + u_e

    dalpha /= param["tau_alpha"]
    du_e *= 1/param["tau_e"]
    du_i /= param["tau_i"]
    
    return du_e, du_i, dalpha


################################################
# Step Solvers
###############################################


def spatial_adapt_Runge_Kutta(u_e,
                              u_i,
                              alpha,
                              I_e,
                              I_i,
                              w_mn,
                              param,
                              dt,
                              rhs: callable=spatial_rhs,
                              P_e: callable=P_e,
                              P_i: callable=P_i):
    """
    Spatial Runge Kutta Method of order 4 

    Args:
        u_e (np.ndarray): vector of excitatory activities at time t.
        u_i (np.ndarray): vector of inhibitory activities at time t.
        w_mn (dict): dictionary of spatial connectivity matricies
        param (dict): parameters of simulation
        dt (float): timestep
        rhs (callable, optional): method for computing the right hand side. Defaults to spatial_rhs.
        P_e (callable, optional): excitatory firing rate function. Defaults to P_e.
        P_i (callable, optional): inhibitory firing rate function. Defaults to P_i.

    Returns:
        u_e, u_i, alpha: next values in simulation
    """
                        
    K1_e, K1_i, K1_alpha = rhs(u_e=u_e, u_i=u_i, alpha=alpha, I_e=I_e, I_i=I_i, param=param, w_mn=w_mn, P_e=P_e, P_i=P_i)

    K2_e, K2_i, K2_alpha = rhs(u_e=u_e + 0.5 * dt * K1_e, u_i=u_i + 0.5 * dt * K1_i, alpha=alpha + 0.5 * dt * K1_alpha,
                               I_e=I_e, I_i=I_i, param=param, w_mn=w_mn, P_e=P_e, P_i=P_i)

    K3_e, K3_i, K3_alpha = rhs(u_e=u_e + 0.5 * dt * K2_e, u_i=u_i + 0.5 * dt * K2_i, alpha=alpha + 0.5 * dt * K2_alpha,
                               I_e=I_e, I_i=I_i, param=param, w_mn=w_mn, P_e=P_e, P_i=P_i)

    K4_e, K4_i, K4_alpha = rhs(u_e=u_e + 0.5 * dt * K3_e, u_i=u_i + 0.5 * dt * K3_i, alpha=alpha + 0.5 * dt * K3_alpha,
                               I_e=I_e, I_i=I_i, param=param, w_mn=w_mn, P_e=P_e, P_i=P_i)
    
    step_e = 1/6 * (K1_e + 2 * K2_e + 2 * K3_e + K4_e)
    step_i = 1/6 * (K1_i + 2 * K2_i + 2 * K3_i + K4_i)
    step_alpha = 1/6 * (K1_alpha + 2 * K2_alpha + 2 * K3_alpha + K4_alpha)

    return u_e + dt * step_e, u_i + dt * step_i, alpha + dt * step_alpha

def spatial_adapt_explicit_euler(u_e,
                                 u_i,
                                 alpha,
                                 I_e,
                                 I_i,
                                 w_mn,
                                 param,
                                 dt,
                                 rhs: callable=spatial_rhs,
                                 P_e: callable=P_e,
                                 P_i: callable=P_i):

    """
    Explicit Euler method

    Args:
        u_e (np.ndarray): vector of excitatory activities at time t.
        u_i (np.ndarray): vector of inhibitory activities at time t.
        w_mn (dict): dictionary of spatial connectivity matricies
        param (dict): parameters of simulation
        dt (float): timestep
        rhs (callable, optional): method for computing the right hand side. Defaults to spatial_rhs.
        P_e (callable, optional): excitatory firing rate function. Defaults to P_e.
        P_i (callable, optional): inhibitory firing rate function. Defaults to P_i.

    Returns:
        u_e, u_i, alpha: next values in simulation
    """
    step_e, step_i, step_alpha = rhs(u_e=u_e, u_i=u_i, alpha=alpha, I_e=I_e, I_i=I_i, param=param, w_mn=w_mn, P_e=P_e, P_i=P_i)

    return u_e + dt * step_e, u_i + dt * step_i, alpha + dt * step_alpha

#######################################
# Spatial Solver
######################################

def spatial_adapt_num_solver(ue_0,
                             ui_0,
                             alpha0,
                             I_e,
                             I_i,
                             param,
                             fft: bool=True,
                             normalized: bool=False,
                             t: np.ndarray=np.linspace(0, 10, 1000),
                             x: np.ndarray=np.linspace(-5, 5, 200),
                             w: callable=exponential_connectivity,
                             P_e: callable=P_e,
                             P_i: callable=P_i,
                             spatial_step_solver: callable=spatial_adapt_Runge_Kutta):
    """
    Solves the integrodifferential equations of the voltage based neural field model with linear adaptation using
    one-step methods.

    Args:
        u_e0 (np.ndarray): starting_values of excitatory population
        u_i0 (np.ndarray): starting_values of inhibitory population 
        alpha0 (np.ndarray): starting_values of adaptation
        I_e (float): external input in excitatory population
        I_i (float): external input in inhibitory population
        param (dict): parameters of simulation
        fft (bool, optional): True, if the convolution shall be computed with the FFT. Defaults to True.
        t (np.ndarray, optional): temporal discretization. Defaults to np.arange(0,10,1e-2).
        x (np.ndarray, optional): spatial discretization. Defaults to np.arange(-5,5,2e-2).
        rhs (callable, optional): method for computing the right hand side. Defaults to spatial_rhs.
        w (callable, optional): spatial connectivity function. Defaults to exponential_connectivity.
        P_e (callable, optional): excitatory firing rate function. Defaults to P_e.
        P_i (callable, optional): inhibitory firing rate function. Defaults to P_i.
        spatial_step_solver (callable, optional): spatial one-step-method used for simulation. Defaults to spatial_Runge_Kutta.

    Returns:
        u_e, u_i: array of excitatory and inhibitory populations, where the columns of represent the spatial distribution at each step.
    """
                       
    dist = np.array([min(x[i]-x[0], x[-i]-x[0]) for i in range(len(x))])
    dx = x[1] - x[0]

    u_e = np.zeros((len(x),len(t)))
    u_i = np.zeros((len(x),len(t)))
    alpha = np.zeros((len(x), len(t)))
    
    u_e[:,0] = ue_0
    u_i[:,0] = ui_0
    alpha[:,0] = alpha0

    mn = ["ee", "ei", "ie", "ii"]

    if normalized:
        param = param | {"nu_ee": 1, "nu_ii": 1, "nu_ie": 1, "nu_ei": 1}

    # Create connectivity dictionaries either with FFT, or as a full matrix
    if fft:
        rhs = spatial_rhs_fft
        if len(x) % 2 != 0: print("The length of x is not even, which will impact performance negatively.")
        w_mn = {sub: dx * np.fft.rfft(w(x=dist, sigma=param["sigma_"+sub], nu=param["nu_"+sub]))  for sub in mn}
    else:
        rhs = spatial_rhs
        dist_matrix = np.zeros((len(dist), len(dist)))
        for i in range(len(dist)):
            dist_matrix[i,:] = np.roll(dist, i)
        w_mn = {sub: dx * w(x=dist_matrix, sigma=param["sigma_"+sub], nu=param["nu_"+sub]) for sub in mn}

    for j in range(len(t)-1):
        # next step with one-step method
        u_e[:,j+1], u_i[:,j+1], alpha[:,j+1]= spatial_step_solver(u_e=u_e[:,j], u_i=u_i[:,j], alpha=alpha[:,j], I_e=I_e, I_i=I_i,
                                        dt=t[j+1]-t[j], rhs=rhs, w_mn=w_mn, P_e=P_e, P_i=P_i, param=param)
    return u_e, u_i, alpha


