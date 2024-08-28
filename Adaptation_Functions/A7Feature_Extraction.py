import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import find_peaks

####################################################################
# Compute Frequency with zero crossings
####################################################################

def frequency_zero_crossing(u, t):
    """
    Estimate the frequency of a signal using the zero-crossing method.

    Parameters:
    u (array): The signal values.
    t (array): The time steps corresponding to the signal values.

    Returns:
    float: Estimated frequency of the signal.
    """

    # Center the signal around zero
    u_centered = u - np.mean(u)

    # Detect zero crossings
    zero_crossings = np.where(np.diff(np.sign(u_centered)))[0]

    # Calculate the time duration of the signal
    duration = t[-1] - t[0]

    # print('crossing: ', len(zero_crossings)/2)
    # print('duration: ', duration)

    # Estimate frequency
    # Each complete wave cycle has two zero crossings, hence the division by 2
    frequency = len(zero_crossings) / (2 * duration)

    return frequency

####################################################################
# Plot values of the first node and the spectrum
####################################################################

def plot_first_node(t,
                    u_e,
                    u_i,
                    alpha,
                    t_start=0,
                    t_stop=None,
                    cutoff_frequency = 20,
                    filename="temp",
                    save=False):

    t_new = t/1000
    if t_stop==None: t_stop=t_new[-1]
    mask = np.where((t_new >= t_start) & (t_new <= t_stop))
    t_new = t_new[mask]

    if len(u_e.shape) == 2:
        ue_first = u_e[0, :][mask]
        ui_first = u_i[0, :][mask]
        alpha_first = alpha[0, :][mask]
    else:
        ue_first = u_e[mask]
        ui_first = u_i[mask]
        alpha_first= alpha[mask]
    
    ue_color = "b"
    ui_color = "darkred"
    alpha_color = "darkorange"

    plt.subplots(2,1, figsize=(6, 5))
    plt.subplot(2,1,1)
    plt.plot(t_new, ue_first, label=r"$u_e$", color=ue_color)
    plt.plot(t_new,ui_first, label=r"$u_i$", color=ui_color)
    plt.plot(t_new, alpha_first, label=r"$\alpha$", color=alpha_color)
    plt.xlabel("Time [s]")
    plt.xticks(np.arange(t_start, t_stop, 5))
    plt.ylabel("Input Voltage/ Adaptation")
    plt.legend(loc="upper right")

    N = len(t_new)
    print(t_start)
    print(t_stop)
    print(N)
    dt = t_new[1] - t_new[0]

    # Compute the corresponding frequencies
    f = np.fft.rfftfreq(N, dt)
    # Cut off the higher and negative frequencies
    indices = np.where((f <= cutoff_frequency) & (f > 0.01))

    # Compute the magnitude of the FFT
    window = np.hanning(N//2 + 1)
    u_e_hat = np.abs(window * np.fft.rfft(ue_first))[indices]
    u_i_hat = np.abs(window * np.fft.rfft(ui_first))[indices]
    alpha_hat = np.abs(window * np.fft.rfft(alpha_first))[indices]
    f = f[indices]

    plt.subplot(2,1,2)
    plt.plot(f, u_e_hat, label="$u_e$", color=ue_color)
    plt.plot(f, u_i_hat, label="$u_i$", color=ui_color)
    plt.plot(f, alpha_hat, label=r"$\alpha$", color=alpha_color)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")

    # Compute and mark maximal Frequency
    max_idx = np.argmax(u_e_hat)
    # dominant_freq = f[max_idx]
    dominant_freq = frequency_zero_crossing(ue_first, t_new)
    plt.axvline(dominant_freq, color="r", linestyle="--", label=f"Dominant frequency [Hz]: {dominant_freq:.3f}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save:
        plt.savefig(filename+".jpeg", format="jpeg")
        plt.close()
    else:
        plt.show()
    return dominant_freq

####################################################################
# Compute and plot the wave speed
####################################################################

def wave_speed(u, x, dt, max_peak_movement=100, return_peaks=False):
    height = np.mean(u)
    tracked_peaks = []
    first_peaks, _ = find_peaks(u[0, :])

    if len(first_peaks) == 0:
        if return_peaks: return [], 0
        return 0
    
    tracked_peaks = [[start_peak] for start_peak in first_peaks]

    for idx_row, row in enumerate(u):
        if idx_row==0: continue
        peaks, _ = find_peaks(row, height=height)
        for idx_tracked in range(len(tracked_peaks)):
            last_peak = tracked_peaks[idx_tracked][-1]
            distances = np.abs(last_peak - peaks)
            closest_peak_idx = np.argmin(distances)
            if distances[closest_peak_idx] * dt <= max_peak_movement:
                tracked_peaks[idx_tracked].append(peaks[closest_peak_idx])
            else:
                # Mark not updated tracked-peaks for deletion
                tracked_peaks[idx_tracked] = []
        # Delete the tracked peaks that haven't been updated
        tracked_peaks = [elem for elem in tracked_peaks if elem != []]
    
    v = []
    delta_x = x[-1] - x[0]
    for peaks in tracked_peaks:
        first_peak = peaks[0]
        last_peak = peaks[-1]
        delta_t = (last_peak - first_peak) * dt
        v.append(delta_x/delta_t)
    
    if return_peaks: return tracked_peaks, np.mean(np.array(v))
    return np.mean(np.array(v))

def wave_speed_plot(u, x, t):
    dt = t[1] - t[0]
    tracked_peaks, speed = wave_speed(u=u, x=x, dt=dt, return_peaks=True)
    # Plot only u_e, just like in create_spatial_addapt_plot
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
    start_x, stop_x = int(np.round(x[0])), int(np.round(x[-1]))
    start_t, stop_t =  int(np.round(t[0])), int(np.round(t[-1]))
    x_ticks = list(range(start_x, stop_x+1, (stop_x - start_x)//10))
    t_ticks = list(range(start_t, stop_t+1, 50 * max(int((stop_t-start_t)/(50 * 20)), 1)))
    data_max = np.round(u.max(), 1) + 0.1
    data_min = np.round(u.min(), 1) - 0.1
    step = round_to_first_significant_digit((data_max - data_min)/100)
    plt.contourf(x, t, u.T, levels=np.arange(data_min, data_max, step), cmap="viridis")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xticks(x_ticks)
    plt.yticks(t_ticks)
    plt.xlabel("Position x")
    plt.ylabel('Time t [ms]')
    # Check if any peaks are actually found
    if len(tracked_peaks) == 0:
        plt.show()
        return

    # left and right y-values of second-peak
    lhs = t[tracked_peaks[1][0]]
    rhs = t[tracked_peaks[1][-1]]
    # Connect Points
    plt.plot([start_x, stop_x], [lhs, rhs], color="red")
    # horizontal line from right hand side
    plt.axhline(rhs, linestyle="--", color="red")

    # Arrow - from left point to top of horizontal line
    arrow = FancyArrowPatch((start_x + 0.1, lhs), (start_x + 0.1, rhs), 
                            arrowstyle='<|-|>', mutation_scale=15, color='r')
    plt.gca().add_patch(arrow)
    # Add the distance as text
    plt.annotate(xy=(start_x + 0.2, (lhs + rhs)/2), text=f"{np.abs(lhs-rhs):.2f} [ms]", color="red", fontsize="large")
    plt.show()
    
def wave_features(u, x, dt):
    mean = np.mean(u)
    speed = wave_speed(u=u, x=x, dt=dt)
    N_t = u.shape[0]
    min_u = np.min(u[0, N_t:])
    max_u = np.max(u[0, N_t:])
    amplitude = max_u - min_u
    middle = min_u + 1/2 * amplitude

    # We want to consider only peaks that are sufficiently large and spaced out.
    # The largest possible frequency should be 20Hz
    distance = 1/20 * 1000/dt
    first_peaks, _ = find_peaks(u[0, :], height=0.1, distance=distance)
    # Calculate the dominant frequency using peak intervals
    intervals = np.diff(first_peaks) * dt/1000
    dominant_freq = 1 / np.mean(intervals)
    
    res = {
        "mean": mean,
        "middle": middle,
        "min": min_u,
        "max": max_u,
        "wave_speed": speed,
        "amplitude": amplitude,
        "frequency": dominant_freq,
    }
    return res
