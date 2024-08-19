from .A0Basic_Functions import *
from . import A1Points_of_equilibrium as A1
from . import A2Local_Model_Stability as A2
from . import A3Full_Model_Stability as A3
from . import A4State_Space as A4
from . import A5Simulation as A5
from . import Parameters as Parameters

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import time

#####################################################
# Plot g_e and g_i
#####################################################

def plot_adapt_equilibrium_points(param,
                                  I_e=0,
                                  I_i=0,
                                  ue_lim=(-0.5, 0.5),
                                  ui_lim=(-0.5, 0.5),
                                  du=1e-5,
                                  wmn_hat: callable=fourier_exponential_connectivity,
                                  show_local_stability=False,
                                  show_full_stability=False,
                                  k_stop=30):

    ue1, ui1 = A1.g_e_line(param, I_i, ui_lim=ui_lim, du=du)
    ue2, ui2 = A1.g_i_line(param, I_e, ue_lim=ue_lim, du=du)
    
    ue_eq, ui_eq = A1.get_intersection(ue1, ui1, ue2, ui2, warning=False)
    ue_eq, ui_eq = A1.get_adapt_equilibrium_points(param=param,
                                                   I_e=I_e, I_i=I_i, du=du,
                                                   ue_lim=ue_lim, ui_lim=ui_lim, warning=False,
                                                   improve_solution=True)
    
    line1, = plt.plot(ue1, ui1)
    line2, = plt.plot(ue2, ui2)
    equilibrium_plot, = plt.plot(ue_eq, ui_eq, "ro", markersize=5)
    


    invisible_line = Line2D([0], [0], color='w', marker='o', markerfacecolor='w', label="")
    plt.xlabel(r"$u_e$")
    plt.ylabel(r"$u_i$")
    plt.xlim((ue_lim))
    plt.ylim((ui_lim))

    # Creating a custom legend for the equilibrium points and adding stability
    num_points = len(ue_eq)
    equilibrium_1 = f"{num_points} Steady State(s):"
    if show_local_stability:
        plt.title("Steady States with Local Stability")
        equilibrium_2 = ""
        i = 0
        for ue, ui in zip(ue_eq, ui_eq):
            stable = A2.local_stability(param=param, ue_eq=ue, ui_eq=ui)
            if stable:
                equilibrium_2 += f"State {i+1} is stable:\n $u_e^{i+1}={ue_eq[i]:.3f}$ \n $u_i^{i+1} = {ui_eq[i]:.3f}$\n"
            else:
                equilibrium_2 += f"State {i+1} is unstable:\n $u_e^{i+1}={ue_eq[i]:.3f}$ \n $u_i^{i+1} = {ui_eq[i]:.3f}$\n"
            i += 1
    elif show_full_stability:
        plt.title("Steady States with Full-Model Stability")
        equilibrium_2 = ""
        i = 0
        for ue, ui in zip(ue_eq, ui_eq):
            stable = A3.adapt_global_stability(k_stop=k_stop, param=param, ue_eq=ue, ui_eq=ui, wmn_hat=wmn_hat)
            locally_stable = A2.local_stability(param=param, ue_eq=ue, ui_eq=ui)
            if stable:
                equilibrium_2 += f"Linearly Stable\n"
            elif locally_stable:
                equilibrium_2 += "Turing unstable:\n"
            else:
                equilibrium_2 += "Locally Unstable:\n"
            equilibrium_2 += f"$u_e^{i+1}={ue_eq[i]:.3f}$\n$u_i^{i+1} = {ui_eq[i]:.3f}$ \n"
            i += 1
    else:
        plt.title("Steady States")
        equilibrium_2 =  "".join([f"$u_e^{i+1}={ue_eq[i]:.3f}$ \n $u_i^{i+1} = {ui_eq[i]:.3f}$ \n" for i in range(num_points)])


    # Add custom legend for equilibrium points along with existing labels
    plt.legend(handles=[line1, line2, equilibrium_plot, invisible_line],
               labels=[r"$g_e(u_i)$", r"$g_i(u_e)$", equilibrium_1 , equilibrium_2],
               loc="upper left",
               bbox_to_anchor=(1,1))
    plt.show()


######################################
# Local Stability Plot
######################################

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(
    x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def plot_limit_cycle(starting_values: list, 
                     param: dict,
                     u1_lim: tuple=(-0.1, 0.5),
                     u2_lim: tuple=(0,0.15),
                     plot_alpha: bool=False,
                     I=[0, 0],
                     automatic_bounds=False,
                     stop_time: float=20.,
                     vector_field: bool=False,
                     num_of_arrows: int=30) -> None:
    """Creates a plot of the limit cycle for different starting values

    Args:
        starting_values (list): list of tuples of starting values
        param (dict): parameters of simulation
    """

    # calculate trajectories for all starting values
    trajectories = []
    for u0 in starting_values:
        u_temp, t = A2.num_solver(u0=u0, stop_time=stop_time, param=param, rhs=A2.local_rhs, I=I)
        trajectories.append(u_temp)
    
    # set bounds automatically
    if automatic_bounds:
        u1_max = max([max(u[:,0]) for u in trajectories]) + 0.01
        u1_min = min([min(u[:,0]) for u in trajectories]) - 0.01
        u1_lim = (u1_min, u1_max)
        u2_max = max([max(u[:,1]) for u in trajectories]) + 0.01
        u2_min = min([min(u[:,1]) for u in trajectories]) - 0.01
        u2_lim = (u2_min, u2_max)

    plt.xlim(u1_lim)
    plt.ylim(u2_lim)
    plt.xlabel("Membrane potential: $u_e$")
    plt.ylabel("Membrane potential: $u_i$")
    plt.title(r'Trajectories')
    
    # plot trajectories
    for u in trajectories:
        colorline(u[:,0], u[:,1])
    
    # Plot the vector field
    if vector_field:
        x, y = np.meshgrid(np.linspace(u1_lim[0], u1_lim[1], num_of_arrows), 
                           np.linspace(u2_lim[0], u2_lim[1], num_of_arrows))
        if param["g"] < 1e-5:
            # If the adaptation-strength is negligible we can plot the vector field.
            alpha = [[0]*num_of_arrows]*num_of_arrows
            dx_dt, dy_dt, dalpha_dt = local_rhs(u = np.array([x,y,alpha]), param=param, I=I)
            norm = np.sqrt(dx_dt**2 + dy_dt**2)
            dx_dt, dy_dt = dx_dt / norm, dy_dt / norm
            plt.quiver(x, y, dx_dt, dy_dt, angles="xy")
        else:
            plt.grid()
    
    # calculate and mark the points of equilibrium
    excitatory_eq, inhibitory_eq= A1.get_adapt_equilibrium_points(param,
                                                                    I_e=I[0],
                                                                    I_i=I[1])
    for ue_eq, ui_eq in zip(excitatory_eq, inhibitory_eq):
        plt.scatter(ue_eq, ui_eq, color="r")

    # mark starting points
    for u0 in starting_values:
        plt.scatter(u0[0], u0[1], facecolors='none', edgecolors="b")

    # Plot alpha over time
    if plot_alpha:
        plt.show()
        for u in trajectories:
            plt.plot(t, u[:,2])
        for i, ue_eq in enumerate(excitatory_eq):
            plt.axhline(ue_eq, linestyle="--", color="black")
        plt.xlabel("Time t")
        plt.ylabel(r"Adaptation $alpha$ (t)")
        plt.title("Adaptation-strength of Trajectories")
        plt.show()


def local_stability_trajectory(param,
                               automatic_bounds=True,
                               trajectories=3,
                               plot_alpha=False,
                               stop_time=20,
                               u1_lim=(-1, 1),
                               u2_lim=(-1, 1),
                               I_e=-0.04,
                               I_i=-0.04,
                               pert=1e-2):
    rd.seed(21)
    I = [I_e, I_i]
    
    ue, ui = A1.get_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i)
    plot_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i, show_local_stability=True)
    
    indices = [i+1 for i in range(len(ue))]
    for ui_eq, ue_eq, index in zip(ui, ue, indices):
        print(f"Steady State {index}: ui_eq: {ui_eq:.2}, ue_eq: {ue_eq:.2}")
        J = A2.J_matrix(param, ue_eq, ui_eq)
        eig_val = np.linalg.eigvals(J)
        print(f"Eigenvalues: {eig_val}")
        stable = A2.local_stability(param, ue_eq=ue_eq, ui_eq=ui_eq)
        print(f"Stability: {stable}")

        start = []
        for i in range(trajectories):
            start.append((ue_eq + rd.uniform(-pert, pert), ui_eq + rd.uniform(-pert, pert), ue_eq + rd.uniform(-pert, pert)))
        plot_limit_cycle(starting_values=start,
                                param=param,
                                automatic_bounds=automatic_bounds,
                                I=I,
                                plot_alpha=plot_alpha,
                                stop_time=stop_time,
                                u1_lim=u1_lim,
                                u2_lim=u2_lim,
                                vector_field=True)
        
###############################################
# Stability Plots
###############################################

def plot_eigenvalues_full_model(param: dict,
                                compute_eq=False,
                                ue_eq_list=[],
                                ui_eq_list=[],
                                I_e=0,
                                I_i=0,
                                k_stop=10,
                                re_lim1=None,
                                re_lim2=None,
                                im_lim=None,
                                n=2000, 
                                lim = 2,
                                marker = "o",
                                show=True,
                                title=None,
                                subtitle=True,
                                axs_label=True,
                                wmn_hat=None):
    ue_lim = (-lim, lim)
    ui_lim = (-lim, lim)
    if compute_eq:
        ue_eq_list, ui_eq_list = A1.get_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i, ue_lim=ue_lim, ui_lim=ui_lim, du=1e-4, iterations=10)
    
    ms=1
    for ue_eq, ui_eq in zip(ue_eq_list, ui_eq_list):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})
        
        k = np.linspace(0, k_stop, n)

        eig1, eig2, eig3 = zip(*[A3.A_eigenvalues(k=k_i, param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat) for k_i in k])
        eig1, eig2, eig3 = map(np.array, [eig1, eig2, eig3])

        full = {"k": k, "eig1": eig1, "eig2": eig2, "eig3": eig3}
        stable_dissected = {"k1": [], "k2": [], "k3": [], "eig1": [], "eig2": [], "eig3": []}
        unstable_dissected = {"k1": [], "k2": [], "k3": [], "eig1": [], "eig2": [], "eig3": []}

        eig = ["eig1", "eig2", "eig3"]
        k_i = ["k1", "k2", "k3"]

        def dissect(i):
            mask_stable = np.real(full[eig[i]]) < 0
            mask_unstable = ~mask_stable
            n = len(k)
            j = 0
            while j < n:
                eig_band, k_band, j = get_band(mask_stable, full[eig[i]], k, j)
                stable_dissected[eig[i]].append(eig_band)
                stable_dissected[k_i[i]].append(k_band)
                eig_band, k_band, j = get_band(mask_unstable, full[eig[i]], k, j)
                unstable_dissected[eig[i]].append(eig_band)
                unstable_dissected[k_i[i]].append(k_band)

        def get_band(mask, eigenvalues, k, j):
            eig_band = []
            k_band = []
            n = len(mask)
            while j < n and mask[j]:
                eig_band.append(eigenvalues[j])
                k_band.append(k[j])
                j +=1
            return eig_band, k_band, j

        # Dissect the eigenvalues and k-values in stable and unstable
        for i in range(3):
            dissect(i)
            
        # Mark Starting points
        axs[0].plot(np.real(eig1)[1], np.imag(eig1[1]), color="green", marker="o", ms=8)
        axs[0].plot(np.real(eig2)[1], np.imag(eig2[1]), color="green", marker="o", ms=8)
        axs[0].plot(np.real(eig3)[1], np.imag(eig3[1]), color="green", marker="o", ms=8)

        # Plot all three eigenvalues in complex plane in blue, if stable and red if unstable
        for i in range(3):
            for j in range(len(stable_dissected[eig[i]])):
                eig_stable = stable_dissected[eig[i]][j]
                axs[0].plot(np.real(eig_stable), np.imag(eig_stable), color="blue", marker=marker, ms=ms, linestyle='None')
                axs[0].set_label(f"eigenvalue {i+1}")
            for j in range(len(unstable_dissected[eig[i]])):
                eig_unstable = unstable_dissected[eig[i]][j]
                axs[0].plot(np.real(eig_unstable), np.imag(eig_unstable), color="red", marker=marker, ms=ms, linestyle='None')

        if title == None:
            title = r"Steady State: $u_e =$" + f"{ue_eq:.3f}" + r"$, u_i =$" + f"{ui_eq:.3f}"
        fig.suptitle(title)
        if subtitle == True:
            axs[0].set_title("Eigenvalues")
        if axs_label:
            axs[0].set_ylabel(r"Im $\lambda_i$")
            axs[0].set_xlabel(r"Re $\lambda_i$")
        else:
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
        if re_lim1 != None:
            axs[0].set_xlim(re_lim1)
            print(re_lim2)
        if im_lim != None:
            axs[0].set_ylim(im_lim)
        axs[0].axhline(0, color="black", linewidth=0.8)
        axs[0].axvline(0, color="black", linewidth=0.8)

        # Plot the real part of all three eigenvalues in blue, if stable and red if unstable
        for i in range(3):
            for j in range(len(stable_dissected[eig[i]])):
                eig_stable = stable_dissected[eig[i]][j]
                k_stable = stable_dissected[k_i[i]][j]
                axs[1].plot(k_stable, np.real(eig_stable), color="blue",  marker=marker, ms=ms, linestyle='None')
                axs[1].set_label(f"eigenvalue {i+1}")
            for j in range(len(unstable_dissected[eig[i]])):
                eig_unstable = unstable_dissected[eig[i]][j]
                k_unstable = unstable_dissected[k_i[i]][j]
                axs[1].plot(k_unstable, np.real(eig_unstable), color="red",  marker=marker, ms=ms, linestyle='None')

        if subtitle:
            axs[1].set_title("Real Part of Eigenvalues")
        if axs_label:
            axs[1].set_ylabel(r"Re $\lambda_i$")
            axs[1].set_xlabel("k")
            axs[1].set_xlim(0, max(k))
            axs[1].axhline(0, color="black", linewidth=0.8)
            axs[1].set_xticks(range(k_stop + 1))
        else:
            axs[1].set_xlim(0, max(k))
            axs[1].set_xticklabels([])
            axs[1].set_yticklabels([])
        if re_lim2 != None:
            axs[1].set_ylim(re_lim2)
        if show: plt.show()

############################################################################################
# Plot the eigenvalues depending on the external currents
############################################################################################

def plot_varying_I_i(param,
                     I_e,
                     wmn_hat=fourier_exponential_connectivity,
                     limit=4,
                     start_I_i=0,
                     stop_I_i=0.5,
                     points=200,):
    I_i_values = np.linspace(start_I_i, stop_I_i, points)
    # Initialize lists to hold the data for plotting
    ue_stable = []
    ue_unstable = []
    ue_locally_unstable = []
    ui_stable = []
    ui_unstable = []
    ui_locally_unstable = []
    I_i_stable = []
    I_i_unstable = []
    I_i_locally_unstable = []

    # Iterate over I_i values, compute equilibrium points and store them with corresponding colors
    ue_lim = (-limit, limit)
    ui_lim = (-limit, limit)
    for I_i in I_i_values:
        ue, ui = A1.get_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i, ue_lim=ue_lim, ui_lim=ui_lim)
        for ue_eq, ui_eq in zip(ue, ui):
            if A3.adapt_global_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat):
                ue_stable.append(ue_eq)
                ui_stable.append(ui_eq)
                I_i_stable.append(I_i)
            elif not A2.local_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq):
                ue_locally_unstable.append(ue_eq)
                ui_locally_unstable.append(ui_eq)
                I_i_locally_unstable.append(I_i)
            else:
                ue_unstable.append(ue_eq)
                ui_unstable.append(ui_eq)
                I_i_unstable.append(I_i)

    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for x_values depending on I_i
    axs[0].scatter(I_i_stable, ue_stable, color='green', label='Stable')
    axs[0].scatter(I_i_locally_unstable, ue_locally_unstable, color="red", label="Locally unstable")
    axs[0].scatter(I_i_unstable, ue_unstable, color='blue', label='Turing unstable')
    axs[0].set_xlabel('I_i')
    axs[0].set_ylabel(r"$u_e^{eq}$")
    axs[0].set_title(r"$u_e^{eq}$ Depending on $I_i$")
    axs[0].legend()
    axs[0].grid(True)

    # Plot for y_values depending on I_i
    axs[1].scatter(I_i_stable, ui_stable, color='green', label='Stable')
    axs[1].scatter(I_i_locally_unstable, ui_locally_unstable, color="red", label="Locally unstable")
    axs[1].scatter(I_i_unstable, ui_unstable, color='blue', label='Turing unstable')
    axs[1].set_xlabel('I_i')
    axs[1].set_ylabel(r"$u_i^{eq}$")
    axs[1].set_title(r"$u_i^{eq}$ Depending on $I_i$")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_varying_I_e(param:dict,
                     I_i:float,
                     wmn_hat=fourier_exponential_connectivity,
                     limit=4,
                     start_I_e=0,
                     stop_I_e=0.5,
                     points=200,):
    I_e_values = np.linspace(start_I_e, stop_I_e, points)
    # Initialize lists to hold the data for plotting
    ue_stable = []
    ue_unstable = []
    ue_locally_unstable = []
    ui_stable = []
    ui_unstable = []
    ui_locally_unstable = []
    I_e_stable = []
    I_e_unstable = []
    I_e_locally_unstable = []
    ue_lim = (-limit, limit)
    ui_lim = (-limit, limit)

    # Iterate over I_e values, compute equilibrium points and store them with corresponding colors
    for I_e in I_e_values:
        ue, ui = A1.get_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i, ue_lim=ue_lim, ui_lim=ui_lim)
        for ue_eq, ui_eq in zip(ue, ui):
            if A3.adapt_global_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat):
                ue_stable.append(ue_eq)
                ui_stable.append(ui_eq)
                I_e_stable.append(I_e)
            elif not A2.local_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq):
                ue_locally_unstable.append(ue_eq)
                ui_locally_unstable.append(ui_eq)
                I_e_locally_unstable.append(I_e)
            else:
                ue_unstable.append(ue_eq)
                ui_unstable.append(ui_eq)
                I_e_unstable.append(I_e)

    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for x_values depending on I_e
    axs[0].scatter(I_e_stable, ue_stable, color='green', label='Stable')
    axs[0].scatter(I_e_locally_unstable, ue_locally_unstable, color="red", label="Locally Unstable")
    axs[0].scatter(I_e_unstable, ue_unstable, color='blue', label='Locally stable, but Unstable in the Full Model')
    axs[0].set_xlabel('I_e')
    axs[0].set_ylabel(r"$u_e^{eq}$")
    axs[0].set_title(r"$u_e^{eq}$ Depending on $I_e$")
    axs[0].legend()
    axs[0].grid(True)

    # Plot for y_values depending on I_e
    axs[1].scatter(I_e_stable, ui_stable, color='green', label='Stable')
    axs[1].scatter(I_e_locally_unstable, ui_locally_unstable, color="red", label="Locally unstable")
    axs[1].scatter(I_e_unstable, ui_unstable, color='blue', label='Turing unstable')
    axs[1].set_xlabel('I_e')
    axs[1].set_ylabel(r"$u_i^{eq}$")
    axs[1].set_title(r"$u_i^{eq}$ Depending on $I_e$")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

######################################################################################################
# Plot the eigenvalues depending on a parameter and plot the difference between the up- and down-state
######################################################################################################

def plot_varying_parameter(param,
                           I_i,
                           I_e,
                           varying_parameter,
                           start=0.01,
                           stop=1.2,
                           points=200,
                           wmn_hat=fourier_exponential_connectivity,
                           limit=2):
    range_of_parameter = np.linspace(start, stop, points)
    # Initialize lists to hold the data for plotting
    ue_stable = []
    ue_unstable = []
    ue_locally_unstable = []
    ui_stable = []
    ui_unstable = []
    ui_locally_unstable = []
    param_stable = []
    param_unstable = []
    param_locally_unstable = []
    diff_e = np.zeros_like(range_of_parameter)
    diff_i = np.zeros_like(range_of_parameter)
    ue_lim = (-limit, limit)
    ui_lim = (-limit, limit)

    # Iterate over I_e values, compute equilibrium points and store them with corresponding colors
    for idx, parameter in enumerate(range_of_parameter):
        param[varying_parameter] = parameter
        ue, ui = A1.get_adapt_equilibrium_points(param=param, I_e=I_e, I_i=I_i, ue_lim=ue_lim, ui_lim=ui_lim)
        if len(ue) == 3:
            diff_e[idx] = ue[2] - ue[0]
            diff_i[idx] = ui[2] - ui[0]
        for ue_eq, ui_eq in zip(ue, ui):
            if A3.adapt_global_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq, wmn_hat=wmn_hat):
                ue_stable.append(ue_eq)
                ui_stable.append(ui_eq)
                param_stable.append(parameter)
            elif not A2.local_stability(param=param, ue_eq=ue_eq, ui_eq=ui_eq):
                ue_locally_unstable.append(ue_eq)
                ui_locally_unstable.append(ui_eq)
                param_locally_unstable.append(parameter)
            else:
                ue_unstable.append(ue_eq)
                ui_unstable.append(ui_eq)
                param_unstable.append(parameter)

    # Create the plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for x_values of the eigenvalues
    axs[0].scatter(param_stable, ue_stable, color='green', label='Stable')
    axs[0].scatter(param_locally_unstable, ue_locally_unstable, color="violet", label="Locally unstable")
    # axs[0].scatter(param_unstable, ue_unstable, color='blue')
    axs[0].scatter(param_unstable, ue_unstable, color='cyan', label='Turing unstable')
    axs[0].set_xlabel(varying_parameter)
    axs[0].set_ylabel(r"$u_e^{st}$")
    axs[0].set_title(r"$u_e^{st}$ Dependent on " + varying_parameter)
    axs[0].legend()
    axs[0].grid(True)

    # Plot for y_values of the eigenvalues
    axs[1].scatter(param_stable, ui_stable, color='green', label='Stable')
    axs[1].scatter(param_locally_unstable, ui_locally_unstable, color="violet", label="Locally unstable")
    axs[1].scatter(param_unstable, ui_unstable, color='cyan', label='Turing unstable')
    # axs[1].scatter(param_unstable, ui_unstable, color='blue')
    axs[1].set_xlabel(varying_parameter)
    axs[1].set_ylabel(r"$u_i^{st}$")
    axs[1].set_title(r"$u_i^{st}$ Dependent on " + varying_parameter)
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Plot the difference between the up- and down-state
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    mask = diff_e != 0.0
    axs[0].plot(range_of_parameter[mask], diff_e[mask], label=r"$u_e$", color='blue')
    axs[0].set_xlabel(varying_parameter)
    axs[0].set_title("Difference of "  + r"$u_e$" + " between up-and down-state")
    axs[0].legend()
    axs[1].plot(range_of_parameter[mask], diff_i[mask], label=r"$u_i$", color='red')
    axs[1].set_xlabel(varying_parameter)
    axs[1].set_title("Difference of "  + r"$u_i$" + " between up-and down-state")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

####################################################################
# Simulation Plot
####################################################################

def create_spatial_adapt_plot(x: np.ndarray,
                              t: np.ndarray,
                              ue_0: np.ndarray=None,
                              ui_0: np.ndarray=None,
                              alpha0:np.ndarray=None,
                              I_e: float=None,
                              I_i: float=None,
                              param: dict=None,
                              w: callable=exponential_connectivity,
                              spatial_step_solver: callable=A5.spatial_adapt_Runge_Kutta,
                              k_max: int=10,
                              run_sim: bool=True,
                              fft: bool=True,
                              show_fft: bool=True,
                              averages: bool=False,
                              steps=1,
                              steps_x=1,
                              u_e=None,
                              u_i=None,
                              alpha=None,
                              save=False,
                              filename="temp"):

    if run_sim:
        neccessary_parameters = [ue_0, ui_0, alpha0, I_e, I_i, param]
        if any(_ is None for _ in neccessary_parameters):
            raise Exception("Not all neccessary parameters are given for the simulation to run.")
        if np.shape(ue_0) != np.shape(x) or np.shape(ui_0) != np.shape(x) or np.shape(alpha0) != np.shape(x):
            raise Exception(f"Shape of starting values ({np.shape(ue_0)} does not match the defined discretization ({np.shape(x)})")
        u_e, u_i, alpha = A5.spatial_adapt_num_solver(ue_0=ue_0, ui_0=ui_0, alpha0=alpha0,
                                                      I_e=I_e, I_i=I_i,
                                                      fft=fft, w=w,
                                                      t=t, x=x, 
                                                      spatial_step_solver=spatial_step_solver,
                                                      param=param)
    else:
        if u_e is None or u_i is None or alpha is None:
            raise Exception("At least one result of the simulation is not given and run_sim is False.")

    def round_to_first_significant_digit(x):
        if x == 0:
            return 0
        else:
            # Determine the factor to multiply by to get a number between 1 and 10
            factor = -int(np.floor(np.log10(abs(x))))
            # Round the number to one significant figure
            x_rounded = round(x, factor)
            # Return the number divided by the factor
            return x_rounded
    
    # Don't plot in full resolution to save time
    def countourf_plot(x, y, data, levels, colors="viridis"):
        plt.contourf(x[::steps_x], y[::steps], data[::steps, ::steps_x], levels=levels, cmap=colors)
    
    Nx = len(x)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    start_x, stop_x = int(np.round(x[0])), int(np.round(x[-1]))
    start_t, stop_t =  int(np.round(t[0])), int(np.round(t[-1]))
    x_ticks = list(range(start_x, stop_x+1, (stop_x - start_x)//10))
    t_ticks = np.array(range(start_t, stop_t+1, 50 * max(int((stop_t-start_t)/(50 * 20)), 1)))/1000


    ################################### Temporal-Spatial Plot ###################################
    fig, ax = plt.subplots(1,3, figsize=(15, 7.5), sharex=True, sharey=True)

    voltage_max = np.round(max(u_e.max(), u_i.max()), 1) + 0.1
    voltage_min = np.round(min(u_e.min(), u_i.min()), 1) - 0.1
    alpha_max = np.round(alpha.max(), 1) + 0.1
    alpha_min = np.round(alpha.min(), 1) - 0.1
    voltage_levels = round_to_first_significant_digit((voltage_max - voltage_min)/100)
    alpha_levels = round_to_first_significant_digit((alpha_max - alpha_min)/100)

    plt.subplot(1,3,1)
    plt.title(r"$u_e (x, t)$")
    countourf_plot(x=x, y=t/1000, data=u_e.T, levels=np.arange(voltage_min, voltage_max, voltage_levels))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xticks(x_ticks)
    plt.yticks(t_ticks)
    plt.xlabel("Position x")
    plt.ylabel('Time [s]')

    plt.subplot(1,3,2)
    plt.title(r"$u_i (x, t)$")
    countourf_plot(x=x, y=t/1000, data=u_i.T, levels=np.arange(voltage_min, voltage_max, voltage_levels))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xticks(x_ticks)
    plt.yticks(t_ticks)
    plt.xlabel("Position x")
    # plt.ylabel('Time [s]')

    plt.subplot(1,3,3)
    plt.title(r"$\alpha (x, t)$")
    countourf_plot(x=x, y=t/1000, data=alpha.T, levels=np.arange(alpha_min, alpha_max, alpha_levels))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xticks(x_ticks)
    plt.yticks(t_ticks)
    plt.xlabel("Position x")
    # plt.ylabel('Time [s]')
    plt.tight_layout()  

    if save:
        plt.savefig(filename+".jpeg", format="jpeg")
        plt.close()
    else:
        plt.show()
    
    ################################### Average Spatial Plot ###################################
    if averages:
        fig, ax = plt.subplots(1,3, figsize=(20,10))

        plt.subplot(1,3,1)
        avrg_u_e = np.sum(u_e, axis=1) / len(t)
        plt.title("Average Activity of $u_e$")
        plt.plot(x, avrg_u_e)
        plt.xticks(x_ticks)
        plt.xlabel("Position x")
        plt.ylabel("Activity")

        plt.subplot(1,3,2)
        avrg_u_i = np.sum(u_i, axis=1) / len(t)
        plt.title(r"Average Activity of $u_i$")
        plt.plot(x, avrg_u_i)
        plt.xticks(x_ticks)
        plt.xlabel("Position x")
        plt.ylabel("Activity")

        plt.subplot(1,3,3)
        avrg_alpha = np.sum(alpha, axis=1) / len(t)
        plt.title(r"Average Activity of $alpha$")
        plt.plot(x, avrg_alpha)
        plt.xticks(x_ticks)
        plt.xlabel("Position x")
        plt.ylabel("Activity")
        plt.show()

    ################################### Temporal Fourier Plot ###################################
    if show_fft:
    
        fig, ax = plt.subplots(1,3, figsize=(20,10))

        plt.subplot(1,3,1)
        u_e_hat = np.fft.rfft(u_e.T, axis=1)
        u_e_hat = np.abs(u_e_hat)
        f = np.fft.rfftfreq(Nx, dx)
        k = 2 * np.pi * f
        index = np.where((k > 0.05) & (k <= k_max))[0][-1]
        k = k[:index+1]
        u_e_hat = u_e_hat[:, :index+1]

        plt.contourf(k[1:], t/1000, u_e_hat[:,1:], levels=np.linspace(0, u_e_hat[1:,1:].max()), cmap="viridis")
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(r"Fourierspectrum of $u_e$")
        plt.yticks(t_ticks)
        plt.xticks(list(range(1,k_max)))
        plt.xlabel("Wave number k")
        plt.ylabel('Time t [s]')

        plt.subplot(1,3,2)
        u_i_hat = np.fft.rfft(u_i, axis=0).T
        u_i_hat = np.abs(u_i_hat)
        u_i_hat = u_i_hat[:, :index+1]
        countourf_plot(x=k[1:], y=t/1000, data=u_i_hat[:, 1:], levels=np.linspace(0, u_i_hat[1:,1:].max()))
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(r"Fourierspectrum of $u_i$")
        plt.yticks(t_ticks)
        plt.xticks(list(range(1,k_max)))
        plt.xlabel("Wave number k")
        plt.ylabel('Time t [s]')

        plt.subplot(1,3,3)
        alpha_hat = np.fft.rfft(alpha, axis=0).T
        alpha_hat = np.abs(alpha_hat)
        alpha_hat = alpha_hat[:, :index+1]
        # plt.contourf(k[1:], t, u_i_hat[:,1:], levels=np.linspace(0, u_i_hat[1:,1:].max()), cmap="viridis")
        countourf_plot(x=k[1:], y=t/1000, data=alpha_hat[:, 1:], levels=np.linspace(0, u_i_hat[1:,1:].max()))
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(r"Fourierspectrum of $alpha$")
        plt.yticks(t_ticks)
        plt.xticks(list(range(1,k_max)))
        plt.xlabel("Wave number k")
        plt.ylabel('Time t [s]')

        plt.show()

    ################################### Average Fourier Plot ###################################
    if averages and show_fft:

        fig, ax = plt.subplots(1,3, figsize=(20,10))
        steady_time = int(400//dt)

        plt.subplot(1,3,1)
        avrg_u_e_hat = np.sum(u_e_hat[steady_time:,:], axis=0)/len(t)
        plt.title("Average Fourierspectrum of $u_e$")
        plt.plot(k[1:], avrg_u_e_hat[1:])
        plt.xticks(list(range(1,k_max)))
        plt.xlabel("wave number")
        max_indx = np.argmax(avrg_u_e_hat[1:])
        k_max_e = k[max_indx + 1]
        plt.axvline(k_max_e, linestyle="--", color="r", label=f"Dominant wave number: {k_max_e:.2f}")
        plt.legend()
        
        plt.subplot(1,3,2)
        avrg_u_i_hat = np.sum(u_i_hat[steady_time:,:], axis=0)/len(t)
        plt.title("Average Fourierspectrum of $u_i$")
        plt.plot(k[1:], avrg_u_i_hat[1:])
        plt.xticks(list(range(1,k_max)))
        plt.xlabel("wave number")
        max_indx = np.argmax(avrg_u_i_hat[1:])
        k_max_i = k[max_indx + 1]
        plt.axvline(k_max_i, linestyle="--", color="r", label=f"Dominant wave number: {k_max_i:.2f}")
        plt.legend()

        plt.subplot(1,3,3)
        avrg_alpha_hat = np.sum(alpha_hat[steady_time:,:], axis=0)/len(t)
        plt.title("Average Fourierspectrum of $alpha$")
        plt.plot(k[1:], avrg_alpha_hat[1:])
        plt.xticks(list(range(1,k_max)))
        plt.xlabel("wave number")
        max_indx = np.argmax(avrg_alpha_hat[1:])
        k_max_alpha = k[max_indx + 1]
        plt.axvline(k_max_i, linestyle="--", color="r", label=f"Dominant wave number: {k_max_alpha:.2f}")
        plt.legend()

        plt.show()

    ################################### Average Activity and Dominant Frequency ###################################

    if averages and show_fft:

        fig, ax = plt.subplots(1,3, figsize=(20,10))

        plt.subplot(1,3,1)
        plt.title("Average Activity of $u_e$ and Dominant Frequency")
        plt.plot(x, avrg_u_e, label = "Average activity")
        dominant_frequency = - np.cos(x * k_max_e)
        center = np.mean(avrg_u_e)
        dominant_frequency *= (max(avrg_u_e) - min(avrg_u_e))/2
        dominant_frequency += center
        plt.plot(x, dominant_frequency, color="r", label="Dominant Frequency (shifted and scaled)")
        plt.xticks(x_ticks)
        plt.xlabel("Position x")
        plt.ylabel("Activity")
        plt.legend()

        plt.subplot(1,3,2)
        plt.title("Average Activity of $u_i$ and Dominant Frequency")
        plt.plot(x, avrg_u_i, label="Average Activity")
        dominant_frequency = - np.cos(x * k_max_i)
        amplitude = (max(avrg_u_i) - min(avrg_u_i))/2
        dominant_frequency *= amplitude
        dominant_frequency += max(avrg_u_i) - amplitude
        plt.plot(x, dominant_frequency, color="r", label="Dominant Frequency (shifted and scaled)")
        plt.xticks(x_ticks)
        plt.xlabel("Position x")
        plt.ylabel("Activity")
        plt.legend()

        plt.subplot(1,3,3)
        plt.title("Average Activity of $alpha$ and Dominant Frequency")
        plt.plot(x, avrg_alpha, label="Average Activity")
        dominant_frequency = - np.cos(x * k_max_alpha)
        amplitude = (max(avrg_alpha) - min(avrg_alpha))/2
        dominant_frequency *= amplitude
        dominant_frequency += max(avrg_alpha) - amplitude
        plt.plot(x, dominant_frequency, color="r", label="Dominant Frequency (shifted and scaled)")
        plt.xticks(x_ticks)
        plt.xlabel("Position x")
        plt.ylabel("Activity")
        plt.legend()
        
    return u_e, u_i, alpha