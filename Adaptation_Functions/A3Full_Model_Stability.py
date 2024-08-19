from .A0Basic_Functions import *
from .A1Points_of_equilibrium import get_adapt_equilibrium_points
from .A2Local_Model_Stability import local_stability

import numpy as np

################################################
# Stability Matrix
################################################

def A_matrix(k,
             param,
             ue_eq,
             ui_eq,
             wmn_hat: callable=fourier_exponential_connectivity):

    # Scalars
    dPe = dP_e(ue_eq - param["theta_e"], param)
    dPi = dP_i(ui_eq - param["theta_i"], param)
    
    # Vectors dependent on k
    wee_hat = wmn_hat(k=k, sigma=param["sigma_ee"], nu=param["nu_ee"])
    wie_hat = wmn_hat(k=k, sigma=param["sigma_ie"], nu=param["nu_ie"])
    wei_hat = wmn_hat(k=k, sigma=param["sigma_ei"], nu=param["nu_ei"])
    wii_hat = wmn_hat(k=k, sigma=param["sigma_ii"], nu=param["nu_ii"]) 
    
    tau_i = param['tau_i']
    tau_e = param['tau_e']
    tau_alpha = param['tau_alpha']
    g = param['g']
    H = drelu(ue_eq)
    if type(k) != np.ndarray:
        A = [[tau_e**(-1)*(-1 + wee_hat * dPe),     tau_e**(-1)*(-wie_hat * dPi), tau_e**(-1)*(-g * H)],
            [tau_i**(-1) * wei_hat * dPe,   -tau_i**(-1) * (1 + wii_hat  * dPi), 0],
            [tau_alpha**(-1),           0,                            -tau_alpha**(-1)]]
        return np.array(A)

    # Since dPe and dPi are scalars, they will be broadcasted over the arrays
    A = np.array([
        [tau_e**(-1)*(-1 + wee_hat * dPe), tau_e**(-1)*(-wie_hat * dPi) , -np.ones(k.shape) * tau_e**(-1) * g * H],
        [tau_i**(-1) * wei_hat * dPe, -tau_i**(-1) * (1 + wii_hat * dPi), np.zeros(k.shape)],
        [tau_alpha**(-1)*np.ones(k.shape), np.zeros(k.shape), -tau_alpha**(-1) * np.ones(k.shape)]
    ])
    # Transpose the array to shape (len(k), 3, 3)
    A = A.transpose(2, 0, 1)
    return A

def A_polynomial_coefficients(k,
                              param,
                              ue_eq,
                              ui_eq,
                              wmn_hat: callable=fourier_exponential_connectivity):
    A = A_matrix(k=k, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat)
    return np.poly(A)

def A_characteristic_polynomial(x,
                                k,
                                param,
                                ue_eq,
                                ui_eq,
                                wmn_hat: callable=fourier_exponential_connectivity):
    q3, q2, q1, q0 = A_polynomial_coefficients(k=k, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat)
    return q3 * x**3 + q2 * x**2 + q1 * x + q0

def A_eigenvalues(k,
                  param,
                  ue_eq,
                  ui_eq,
                  wmn_hat: callable=fourier_exponential_connectivity):
    eig = np.linalg.eigvals(A_matrix(k=k, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat))
    if eig.ndim == 1:
        return np.array(sorted(eig, key=lambda x: x.real))
    sorting_indices = np.argsort(np.real(eig), axis=1)
    sorted_eigenvalues = np.take_along_axis(eig, sorting_indices, axis=1)
    return sorted_eigenvalues

def A_polynomial_coefficients_by_hand(k,
                                      param,
                                      ue_eq,
                                      ui_eq,
                                      wmn_hat: callable=fourier_exponential_connectivity):
    dPe = dP_e(ue_eq - param["theta_e"], param)
    dPi = dP_i(ui_eq - param["theta_i"], param)
    wee_hat = wmn_hat(k=k, sigma=param["sigma_ee"], nu=param["nu_ee"])
    wie_hat = wmn_hat(k=k, sigma=param["sigma_ie"], nu=param["nu_ie"])
    wei_hat = wmn_hat(k=k, sigma=param["sigma_ei"], nu=param["nu_ei"])
    wii_hat = wmn_hat(k=k, sigma=param["sigma_ii"], nu=param["nu_ii"])
    tau_e =  param["tau_e"]
    tau_i =  param["tau_i"]
    tau_alpha =  param["tau_alpha"]
    g = param["g"]
    H = drelu(ue_eq)
    q0 = 1 + g * H - wee_hat * dPe + wii_hat * dPi + g * H * wii_hat * dPi + wei_hat * wie_hat * dPe * dPi - wee_hat * wii_hat * dPe * dPi
    q1 = tau_e + tau_i + tau_alpha + (g * H * tau_i) + (tau_e * wii_hat * dPi) - (tau_i * wee_hat * dPe) - (tau_alpha * wee_hat * dPe) + (wii_hat * dPi * tau_alpha) - (tau_alpha * wee_hat * wii_hat * dPe * dPi) + (tau_alpha * wei_hat * wie_hat * dPe * dPi)
    q2 = (tau_e * tau_i) + (tau_e * tau_alpha) + (tau_i * tau_alpha) + (tau_alpha * tau_e * wii_hat * dPi) - (tau_i * tau_alpha * wee_hat * dPe)
    q3 = tau_i * tau_e * tau_alpha
    if type(q0) == np.ndarray:
        q3 = q3 * np.ones(q0.shape)
    coeff = np.array([q3, q2, q1, q0])
    return 1/(tau_e * tau_i * tau_alpha) * coeff



################################################
# Computing Stability 
###############################################

def Routh_Hurwitz(coeff):
    if coeff.size == 4:
        q3, q2, q1, q0 = coeff
        return q2 > 0 and q1 > 0 and q0 > 0 and q2 * q1 > q0
    else:
        # Extract each coefficient
        q3 = coeff[0, :]
        q2 = coeff[1, :]
        q1 = coeff[2, :]
        q0 = coeff[3, :]

        # Conditions to check
        condition1 = np.all(q0 > 0)
        condition2 = np.all(q1 > 0)
        condition3 = np.all(q2 > 0)
        condition4 = np.all(q2 * q1 > q0)

        # Check if all conditions hold for every set of coefficients
        return condition1 & condition2 & condition3 & condition4

def adapt_full_model_stability(k, param: dict, ue_eq: float, ui_eq: float, wmn_hat: callable=fourier_exponential_connectivity):
    return Routh_Hurwitz(A_polynomial_coefficients_by_hand(k=k, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat))
def adapt_full_model_stability2(k, param: dict, ue_eq: float, ui_eq: float, wmn_hat: callable=fourier_exponential_connectivity):
    eig_values = A_eigenvalues(k=k, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat)
    return np.all(np.real(eig_values) <= 0)

def adapt_global_stability(param: dict, ue_eq: float, ui_eq: float, k_stop=30, wmn_hat: callable=fourier_exponential_connectivity):
    return Routh_Hurwitz(A_polynomial_coefficients_by_hand(k=np.linspace(0, k_stop, 1000), param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat))
# def adapt_global_stability(param: dict, ue_eq: float, ui_eq: float, k_stop=30, wmn_hat: callable=fourier_exponential_connectivity):
#     return np.all([adapt_full_model_stability(ki, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat) for ki in np.linspace(0, k_stop, 1000)])

################################################
# Compute gain-bands and get type of stability
###############################################

def get_gain_bands(param,
                   ue_eq,
                   ui_eq,
                   k_stop=30,
                   wmn_hat: callable=fourier_exponential_connectivity):
    if not local_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq): return "Locally Unstable"
    k_values = np.linspace(0, k_stop, 1000)
    all_eigenvalues = A_eigenvalues(k_values, param, ue_eq, ui_eq, wmn_hat)
    # all_eigenvalues = [A_eigenvalues(k=k, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat) for k in k_values]
    k_bands = []
    i = 0
    while i < len(k_values):
        while i < len(k_values) and np.all(all_eigenvalues[i].real < 0):
            i += 1

        current_k_band = []
        while  i < len(k_values) and np.any(all_eigenvalues[i].real >= 0):
            current_k_band.append(k_values[i])
            i += 1

        if current_k_band != []:
            k_bands.append(current_k_band)

    return k_bands

def type_of_instability(param,
                        ue_eq,
                        ui_eq,
                        tol=1e-5,
                        k_stop=30,
                        wmn_hat: callable=fourier_exponential_connectivity):
    if not local_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq): return ["Locally Unstable"]
    if adapt_global_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat, k_stop=k_stop): return ["Stable"]
    # return ["Turing"]
    gain_bands = get_gain_bands(param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat, k_stop=k_stop)
    # The Tuple refers to the imaginary part at the crossing of the imaginary axis.
    # i.e. (True, False) the first eigenvalue in the gain band with positive real part
    # has a nonzero imaginary part and the last one is purely real
    classify_type = {
        (True, True): "Dynamic Turing",
        (False, False): "Static Turing",
        (True, False): "Mixed Turing", 
        (False, True): "Mixed Turing"
    }
    
    instability_of_gain_band = []
    for band in gain_bands:
        entry_imag = False
        exit_imag = False
        entry_eigenvalues = A_eigenvalues(k=band[0], param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat)
        exit_eigenvalues = A_eigenvalues(k=band[-1], param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat)
        if np.any((entry_eigenvalues.real > 0) & (np.abs(entry_eigenvalues.imag) > 0)):
            entry_imag = True
        if np.any((exit_eigenvalues.real > 0) & (np.abs(exit_eigenvalues.imag) > tol)):
            exit_imag = True
        instability_of_gain_band.append((entry_imag, exit_imag))
    return [classify_type[elem] for elem in instability_of_gain_band]

def adapt_points_of_equilibrium_and_stability(param,
                                              I_e,
                                              I_i,
                                              k_stop=30,
                                              lim =2,
                                              wmn_hat: callable=fourier_exponential_connectivity):
    ue_lim = (-lim, lim)
    ui_lim = (-lim, lim) 
    ue, ui = get_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i, ue_lim=ue_lim, ui_lim=ui_lim)
    for ue_eq, ui_eq in zip(ue, ui):
        stable = adapt_global_stability(param=param, k_stop=k_stop, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat)
        if stable:
            print(f"Stable Point of Equilibrium: ue_eq={ue_eq:.3f}, ui_eq={ui_eq:.2f}")
        else:
            print(f"Unstable Point of Equilibrium: ue_eq={ue_eq:.3f}, ui_eq={ui_eq:.3f}")

