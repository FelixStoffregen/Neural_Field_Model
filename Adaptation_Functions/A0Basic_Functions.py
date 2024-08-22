import numpy as np

######################################
# RHS of the local limit
######################################

def local_rhs(u, param, I=[0,0]):
    I_e, I_i = I
    u_e, u_i, alpha = u[0], u[1], u[2]
    du_e = - u_e + param["nu_ee"] * P_e(u_e - param["theta_e"], param) - param["nu_ie"] * P_i(u_i - param["theta_i"], param) - param["g"] * relu(alpha) + I_e
    du_i = - u_i + param["nu_ei"] * P_e(u_e - param["theta_e"], param) - param["nu_ii"] * P_i(u_i - param["theta_i"], param) + I_i
    dalpha = - alpha + u_e
    
    du_i *= 1/param["tau_i"]
    du_e *= 1/param["tau_e"]
    dalpha *= 1/param["tau_alpha"]
    
    return np.array([du_e, du_i, dalpha])


######################################
# Functions P_e and P_i 
######################################

def P_e(u, param):
    return 0.5 * (1 + np.tanh(param["beta_e"] * u))

def dP_e(u, param):
    return 0.5 * param["beta_e"] * (1 - np.tanh(param["beta_e"] * u)**2)

def P_i(u, param):
    return 0.5 * (1 + np.tanh(param["beta_i"] * u))

def dP_i(u, param):
    return 0.5 * param["beta_i"] * (1 - np.tanh(param["beta_i"] * u)**2)

def inverse_P_e(u, param):
    x = 2 * u - 1
    result = np.empty_like(x)

    lower_bound = x < -1 + 1e-8
    upper_bound = x > 1 - 1e-8
    valid_mask = ~(lower_bound | upper_bound)

    result[lower_bound] = -1e10
    result[upper_bound] = 1e10
    result[valid_mask] = 1 / param["beta_e"] * np.arctanh(x[valid_mask])

    return result

def inverse_P_i(u, param):
    x = 2 * u - 1
    result = np.empty_like(x)

    lower_bound = x < -1 + 1e-8
    upper_bound = x > 1 - 1e-8
    valid_mask = ~(lower_bound | upper_bound)

    result[lower_bound] = -1e10
    result[upper_bound] = 1e10
    result[valid_mask] = 1 / param["beta_i"] * np.arctanh(x[valid_mask])

    return result


#####################################
# Exponential, Gaussian auf Fourier Connectivity (W_mn and hat(w_mn))
######################################

def exponential_connectivity(x, sigma, nu=1):
    return nu * 1/(2*sigma) * np.exp(-np.abs(x/sigma))

def fourier_exponential_connectivity(k, sigma, nu=1):
    return nu * 1/(1 + sigma**2 * k**2)

def gaussian_connectivity(x, sigma, nu=1):
    return nu * (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(((x**2))/(2*(sigma**2))))

def fourier_gaussian_connectivity(k, sigma, nu=1):
    return nu * np.exp(-(1/2) * (sigma**2) * (k**2))

def rectangular_connectivity(x, sigma, nu=1):
    h = x / sigma
    return nu * np.where(np.abs(h) < 1, 1 / (2*sigma), 0)

def fourier_rectangular_connectivity(k, sigma, nu=1):
    denominator = k * sigma
    result = np.where(np.isclose(denominator, 0), 1, np.sin(denominator) / denominator)
    result = np.where(np.isnan(result), 1, result)  # Handle NaN values from previous step
    return nu * result


######################################
# Linear Rectifier function
######################################

def relu(x):
    if type(x) == np.ndarray:
        return np.where(x < 0, 0, x)
    elif isinstance(x, (int, float)):
        return max(x, 0)


def drelu(x):
    if type(x) == np.ndarray:
        return np.where((x < 0) | (x > 1), 0, 1)
    elif isinstance(x, list):
        return [0 if i < 0 else 1 for i in x]
    elif isinstance(x, (int, float)):
        if x < 0:
            return 0
        return 1
    else:
        raise Exception('No valid input')

######################################
# Functions g_i and g_e
######################################

def g_i(u_e, param, I_e):
    temp = - u_e - param["g"] * relu(u_e) + param["nu_ee"] * P_e(u_e - param["theta_e"], param) + I_e
    return inverse_P_i(1/param["nu_ie"] * temp, param) + param["theta_i"]

def g_e(u_i, param, I_i):
    temp = u_i + param["nu_ii"] * P_i(u_i - param["theta_i"], param) - I_i
    return inverse_P_e(1/param["nu_ei"] * temp, param) + param["theta_e"]
