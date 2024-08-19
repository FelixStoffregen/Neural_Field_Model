from .A0Basic_Functions import *
from .A1Points_of_equilibrium import get_adapt_equilibrium_points
from .A3Full_Model_Stability import type_of_instability

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import time

######################################################
# Main function
######################################################

def plot_I_plane(param: dict,
                 wmn_hat: callable=fourier_exponential_connectivity,
                 I_e_start=-0.5,
                 I_e_stop=0.5,
                 dI_e=2e-2, 
                 I_i_start=-0.5,
                 I_i_stop=0.5, 
                 dI_i=2e-2,
                 k_stop=50,
                 return_res=False,
                 return_img=False,
                 limit=1.2,
                 plot=True,
                 print_times=True,
                 print_progress=False,
                 save=False,
                 show=True,
                 legend=True,
                 equilibrium_point_precision=5e-4,
                 only_up_states = True,
                 small_ticks=True,
                 title="",
                 filename="stability_plot_varying_I"):

    I_e_list = np.arange(I_e_start, I_e_stop + dI_e, dI_e)
    I_i_list = np.arange(I_i_start, I_i_stop + dI_i, dI_i)

    Ne = np.size(I_e_list)
    Ni = np.size(I_i_list)
    res = np.zeros((Ne, Ni), dtype=np.int8)

    T1 = []
    T2 = []
    
    if only_up_states:
        encoding_func = encoding_only_up
    else:
        encoding_func = encoding_full

    for e, I_e in enumerate(I_e_list):
        if e == Ne//10 and print_progress:
            print("10%")
        if e == Ne//4 and print_progress:
            print("25%")
        if e == Ne//2 and print_progress:
            print("50%")
        if e == 3 * Ne // 4 and print_progress:
            print("75%")
        for i, I_i in enumerate(I_i_list):
            stability_list = []
            t1 = time.time()
            if I_e > 1.0 and I_i > 0:
                pass
            ue, ui = get_adapt_equilibrium_points(param=param,
                                                  I_e=I_e, I_i=I_i,
                                                  ue_lim=(-limit, limit),
                                                  ui_lim=(-limit, limit),
                                                  du=equilibrium_point_precision,
                                                  warning=True)
            t2 = time.time()
            for ue_eq, ui_eq in zip(ue, ui):
                current_instabilitiy = type_of_instability(param=param,
                                                           k_stop=k_stop,
                                                           ue_eq=ue_eq,
                                                           ui_eq=ui_eq,
                                                           wmn_hat=wmn_hat)
                if len(current_instabilitiy) == 1:
                    stability_list.append(current_instabilitiy[0])
                else:
                    stability_list.append("Multiple Gain Bands")
                
            t3 = time.time()
            T1.append(t2-t1)
            T2.append(t3-t2)
            res[e, i] = encoding_func(stability_list)

            if res[e, i] == -1:
                # An Exception has been thrown
                print("These values has caused an issue:")
                print(f"I_e: {I_e:.4} and I_i: {I_i:.4}")
                print(f"ue: {ue}")
                print(f"ui: {ui}")
                
    if print_progress: print("100%")
    if print_times: print(f"Time to get points of Equilibrium: {sum(T1):.3f}")
    if print_times: print(f"Time for classification: {sum(T2):.3f}")

    if plot: img = plot_equilibria(res=res,
                                     y=I_i_list,
                                     y_axis=r"$I_i$",
                                     x=I_e_list,
                                     x_axis=r"$I_e$",
                                     only_up_states=only_up_states,
                                     small_ticks=small_ticks,
                                     legend=legend,
                                     title=title,
                                     show=show)

    if save: plot_equilibria(res=res,
                             y=I_i_list,
                             y_axis=r"$I_i$",
                             x=I_e_list,
                             x_axis=r"$I_e$",
                             only_up_states=only_up_states,
                             small_ticks=small_ticks,
                             title=title,
                             save=save,
                             show=show,
                             filename=filename)

    if return_img: return img
    if return_res: return res

######################################################
# Plotting function
######################################################

def plot_equilibria(res,
                    x,
                    x_axis,
                    y,
                    y_axis,
                    save=False,
                    legend=True,
                    only_up_states = True,
                    title='State Space of I-Plane',
                    small_ticks=True,
                    show=True,
                    filename=""):
    if only_up_states:
        cases = cases_func_only_up()
        colors_for_cases = colors_for_cases_func_only_up()
    else:
        cases = cases_func_full()
        colors_for_cases = colors_for_cases_func_full()

    
    # Extract the colors and labels for the colormap and colorbar
    cmap_list = [colors_for_cases[case] for case in cases]
    cmap = mcolors.ListedColormap(cmap_list)

    # Adjust the bounds to correctly reflect the encoded values
    bounds = np.arange(-0.5, len(cases), 1.)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the results
    img = plt.imshow(np.transpose(res), extent=(x.min(), x.max(), y.min(), y.max()),
                    origin='lower', cmap=cmap, norm=norm, aspect='equal')

    # Create a grid
    plt.grid(color='black', linestyle='-', linewidth=0.25)

    # Create colorbar ticks and labels
    if legend:
        cbar_ticks = range(len(cases))
        cbar = plt.colorbar(img, ticks=cbar_ticks, spacing='uniform')
        cbar.ax.set_yticklabels(cases)

    # Set the labels for x and y axes
    plt.xlabel(x_axis)
    
    if small_ticks: plt.xticks(np.arange(np.round(x.min(), 1), x.max(), step=0.2))
    plt.ylabel(y_axis)
    if small_ticks: plt.yticks(np.arange(np.round(y.min(), 1) + 0.1, y.max(), step=0.2))
    plt.title(title)
    
    # Save or show, depending on the settings
    if save:
        plt.savefig(filename+".jpeg", format="jpeg")
        plt.close()
    if show:
        plt.figure(figsize=(8, 6))
        plt.show()
    return img

######################################################
# Encoding, Cases and Colors for upper region only
######################################################

def cases_func_only_up():
    cases = [
            # One Point of Equilibrium
            "1, Stable state",
            "1, Locally unstable state",
            "1, Turing unstable state",
            # Three Points of Equilibrium
            "3, No Stable state",

            "3, Stable up-state, Locally unstable down-state",
            "3, Stable up-state, Static-Turing unstable down-state",
            "3, Stable up-state, Dynamic-Turing unstable down-state",
            "3, Stable up-state, Mixed-Turing unstable down-state",
            "3, Stable up-state, Multiple Gain Bands in down-state",

            "3, Stable down-state, Locally unstable up-state",

            "3, Stable up-state, Stable down-state",
    ]
    return cases

def colors_for_cases_func_only_up():
    colors_for_cases = {
                        "3, Stable up-state, Multiple Gain Bands in down-state": "magenta",
                        # One Point of Equilibrium
                        "1, Stable state": "blue",
                        "1, Locally unstable state": "lightblue",
                        "1, Turing unstable state": "turquoise",
                        # Three Points of Equilibrium
                        "3, No Stable state": "bisque",

                        "3, Stable up-state, Locally unstable down-state": "lightsalmon",
                        "3, Stable up-state, Static-Turing unstable down-state": "green",
                        "3, Stable up-state, Dynamic-Turing unstable down-state": "lime",
                        "3, Stable up-state, Mixed-Turing unstable down-state": "yellowgreen",

                        "3, Stable down-state, Locally unstable up-state": "red",

                        "3, Stable up-state, Stable down-state": "darkred",
    }
    return colors_for_cases

def encoding_only_up(stability_list):
    cases = cases_func_only_up()

    num_points_of_eq = len(stability_list)
    num_stable_points = stability_list.count("Stable")
    num_local_instabilities = stability_list.count("Locally Unstable")
    # num_Turing_instabilities = stability_list.count("Turing")

    Dynamic_Turing = stability_list.count("Dynamic Turing")
    Static_Turing = stability_list.count("Static Turing")
    Mixed_Turing = stability_list.count("Mixed Turing")
    num_Turing_instabilities = Dynamic_Turing + Static_Turing + Mixed_Turing

    # if Mixed_Turing > 0: print('Mixed Turing')
    # if Dynamic_Turing > 0: print('Dynamic Turing')
    # if Static_Turing > 0: print('Static Turing')
    
    if num_points_of_eq == 0:
        return cases.index("The Rest")
    
    if "Multiple Gain Bands" in stability_list:
        # print()
        # print("Exception:")
        # print("There is a case with multiple gain bands:")
        # print(stability_list)
        if num_points_of_eq == 1:
            return cases.index('1, Turing unstable state')
        if num_points_of_eq == 3:
            if stability_list[2] == "Stable":
                return cases.index("3, Stable up-state, Multiple Gain Bands in down-state")
            else:
                return cases.index("3, No Stable state")        
        print(stability_list)
        return cases.index("The Rest")
    
    if num_points_of_eq == 2:
        # print()
        # print("Exception:")
        print("Two Points of Equilibrium!")
        print(stability_list)
        return cases.index("The Rest")

    # Case 1: One Point of Equilibrium
    if num_points_of_eq == 1:
        if num_stable_points == 1:
            return cases.index("1, Stable state")
        if num_Turing_instabilities == 1: 
            return cases.index("1, Turing unstable state")

        return cases.index("1, Locally unstable state")

    # Case 2: Three Points of Equilibrium
    if num_points_of_eq == 3:
        if num_stable_points == 0:
            return cases.index("3, No Stable state")
        
        if num_stable_points == 2:
            return cases.index("3, Stable up-state, Stable down-state")
        
        if num_stable_points == 1:
            # One Stable Point
            if stability_list[2] == "Stable" and stability_list[1] == "Locally Unstable":
                # Stable up-state and locally unstable middle state
                if Dynamic_Turing == 1:
                    return cases.index("3, Stable up-state, Dynamic-Turing unstable down-state")
                elif Static_Turing == 1:
                    return cases.index("3, Stable up-state, Static-Turing unstable down-state")
                elif Mixed_Turing == 1:
                    return cases.index("3, Stable up-state, Mixed-Turing unstable down-state")
                else:
                    return cases.index("3, Stable up-state, Locally unstable down-state")
                
            elif stability_list[0] == "Stable" and stability_list[1] == "Locally Unstable":
                #  Stable down-state and locally unstable middle state
                if stability_list[2] == "Locally Unstable":
                    return cases.index("3, Stable down-state, Locally unstable up-state")

            else:
                print()
                print("Exception:")
                print("The middle point is stable, or middle point is not locally unstable.")
                return cases.index("The Rest") # Exception

        print()
        print("Exception:")
        print("Three stable points of Equilibrium")
    return cases.index("The Rest") # Exception

######################################################
# Encoding, Cases and Colors for both regions
######################################################

def cases_func_full():
    cases = [
            # One Point of Equilibrium
            "1, Stable state",
            "1, Locally unstable state",
            "1, Turing unstable state",
            # Three Points of Equilibrium
            "3, No Stable state",

            "3, Stable up-state, Turing unstable down-state",
            "3, Stable up-state, Locally unstable down-state",

            "3, Stable down-state, Turing unstable up-state",
            "3, Stable down-state, Locally unstable up-state",

            "3, Stable up-state, Stable down-state",
    ]
    return cases

def colors_for_cases_func_full():
    colors_for_cases = {
                        "3, Stable up-state, Multiple Gain Bands in down-state": "magenta",
                        # One Point of Equilibrium
                        "1, Stable state": "blue",
                        "1, Locally unstable state": "lightblue",
                        "1, Turing unstable state": "turquoise",
                        # Three Points of Equilibrium
                        "3, No Stable state": "bisque",

                        "3, Stable up-state, Locally unstable down-state": "lightsalmon",
                        "3, Stable up-state, Turing unstable down-state": "greenyellow",

                        "3, Stable down-state, Locally unstable up-state": "darkorange",
                        "3, Stable down-state, Turing unstable up-state": "darkolivegreen",

                        "3, Stable up-state, Stable down-state": "darkred",
    }
    return colors_for_cases

def encoding_full(stability_list):
    cases = cases_func_full()

    num_points_of_eq = len(stability_list)
    num_stable_points = stability_list.count("Stable")
    num_local_instabilities = stability_list.count("Locally Unstable")
    # num_Turing_instabilities = stability_list.count("Turing")

    Dynamic_Turing = stability_list.count("Dynamic Turing")
    Static_Turing = stability_list.count("Static Turing")
    Mixed_Turing = stability_list.count("Mixed Turing")
    num_Turing_instabilities = Dynamic_Turing + Static_Turing + Mixed_Turing

    # if Mixed_Turing > 0: print('Mixed Turing')
    # if Dynamic_Turing > 0: print('Dynamic Turing')
    # if Static_Turing > 0: print('Static Turing')
    
    if num_points_of_eq == 0:
        return cases.index("The Rest")
    
    if "Multiple Gain Bands" in stability_list:
        if num_points_of_eq == 1:
            return cases.index('1, Turing unstable state')
        if num_points_of_eq == 3:
            if stability_list[2] == "Stable":
                return cases.index("3, Stable up-state, Turing unstable down-state")
            if stability_list[0] == "Stable":
                return cases.index("3, Stable down-state, Turing unstable up-state")
            else:
                return cases.index('3, No Stable state')        
        return cases.index("The Rest")
    
    if num_points_of_eq == 2:
        # print()
        # print("Exception:")
        # print("Two Points of Equilibrium!")
        return cases.index("The Rest")

    # Case 1: One Point of Equilibrium
    if num_points_of_eq == 1:
        if num_stable_points == 1:
            return cases.index("1, Stable state")
        if num_Turing_instabilities == 1: 
            return cases.index('1, Turing unstable state')
        return cases.index("1, Locally unstable state")

    # Case 2: Three Points of Equilibrium
    if num_points_of_eq == 3:
        if num_stable_points == 0:
            return cases.index("3, No Stable state")
        
        if num_stable_points == 2:
            return cases.index("3, Stable up-state, Stable down-state")
        
        if num_stable_points == 1:
            # One Stable Point
            if stability_list[2] == "Stable" and stability_list[1] == "Locally Unstable":
                # Stable up-state and locally unstable middle state
                if num_Turing_instabilities == 1:
                    return cases.index("3, Stable up-state, Turing unstable down-state")
                else:
                    return cases.index("3, Stable up-state, Locally unstable down-state")
                
            elif stability_list[0] == "Stable" and stability_list[1] == "Locally Unstable":
                if num_Turing_instabilities == 1:
                    return cases.index("3, Stable down-state, Turing unstable up-state")
                else:
                    return cases.index("3, Stable down-state, Locally unstable up-state")
            else:
                print()
                print("Exception:")
                print("The middle point is stable, or middle point is not locally unstable.")
                return cases.index("The Rest") # Exception

        print()
        print("Exception:")
        print("Three stable points of Equilibrium")
    return cases.index("The Rest") # Exception
