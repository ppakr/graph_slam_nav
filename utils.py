import gtsam
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import DataLoader
from gtsam.symbol_shorthand import B, V, X
from gtsam import NavState, Pose3, Rot3, Point3
from gtsam.utils.plot import plot_pose3, set_axes_equal
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List
from functools import partial


def plot_trajectory_fast(
    values: gtsam.Values, title: str = "Trajectory", fignum: int = 1, show: bool = True
):
    # Create/Get the figure
    fig = plt.figure(fignum, figsize=(10, 8))

    # Clear the figure to remove any existing 2D axes
    fig.clf()

    # Explicitly add a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = [], [], []
    i = 0

    # Extract data
    while values.exists(X(i)):
        pose = values.atPose3(X(i))
        p = pose.translation()

        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
        i += 1

    if len(xs) == 0:
        print("No poses found!")
        return

    # Plot the path line
    ax.plot(xs, ys, zs, "-b", linewidth=1, label="Predicted Path")

    # Mark Start (Green) and End (Red)
    # Note: On a 3D axis, scatter accepts (x, y, z)
    ax.scatter(xs[0], ys[0], zs[0], c="g", marker="o", s=50, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], c="r", marker="x", s=50, label="End")

    # Labels and Title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()

    # Equal Aspect Ratio (Try/Except in case utils are missing)
    try:
        gtsam.utils.plot.set_axes_equal(fignum)
    except:
        pass

    if show:
        plt.show()


def get_initial_state_from_data(query_time, odom_data, dvl_data):
    """
    Interpolates Odometry and DVL data to get a robust initial estimate
    for Pose and Velocity at a specific time.
    """
    # Find nearest index in Odometry for Position/Orientation
    # (Using searchsorted is faster than argmin for sorted time)
    idx = np.searchsorted(odom_data["time"], query_time)
    if idx == 0:
        idx = 1
    if idx >= len(odom_data["time"]):
        idx = len(odom_data["time"]) - 1

    # Linear Interpolation factor
    t0 = odom_data["time"][idx - 1]
    t1 = odom_data["time"][idx]
    ratio = 0.0
    if t1 > t0:
        ratio = (query_time - t0) / (t1 - t0)

    # Interpolate Position
    pos0 = np.array(
        [odom_data["x"][idx - 1], odom_data["y"][idx - 1], odom_data["z"][idx - 1]]
    )
    pos1 = np.array([odom_data["x"][idx], odom_data["y"][idx], odom_data["z"][idx]])
    pos_interp = pos0 + ratio * (pos1 - pos0)

    # Get Rotation (Nearest neighbor is sufficient for initialization, or SLERP if precise)
    # Using nearest for stability
    q_idx = idx if ratio > 0.5 else idx - 1
    rot = gtsam.Rot3.Quaternion(
        odom_data["qw"][q_idx],
        odom_data["qx"][q_idx],
        odom_data["qy"][q_idx],
        odom_data["qz"][q_idx],
    )

    pose_est = gtsam.Pose3(rot, gtsam.Point3(*pos_interp))

    # Get Velocity from DVL
    # DVL measures body velocity. NavState needs World velocity.
    # V_world = R_body_to_world * V_body
    v_idx = np.searchsorted(dvl_data["time"], query_time)
    if v_idx >= len(dvl_data["time"]):
        v_idx = len(dvl_data["time"]) - 1

    # Nearest DVL reading
    v_body = np.array(
        [dvl_data["vx"][v_idx], dvl_data["vy"][v_idx], dvl_data["vz"][v_idx]]
    )

    vel_world_est = rot.rotate(gtsam.Point3(*v_body))

    return pose_est, vel_world_est


def plot_trajectory_fast_multi(
    values: gtsam.Values,
    title: str = "Trajectory",
    fignum: int = 1,
    show: bool = True,
    # New arguments for multi-plot support
    clear: bool = False,
    color: str = "b",
    label: str = "Predicted Path",
):
    """
    Plots a 3D trajectory from GTSAM values.

    Args:
        values: GTSAM Values object containing Pose3s with keys X(0), X(1)...
        title: Title of the plot
        fignum: Figure number to plot on
        show: Whether to call plt.show() at the end
        clear: If True, clears the figure before plotting (start fresh).
               If False, adds to the existing plot (overlay).
        color: Color of the trajectory line
        label: Legend label for this specific trajectory
    """
    # Create/Get the figure
    fig = plt.figure(fignum, figsize=(10, 8))

    # CLEAR logic: Only clear if explicitly requested
    if clear:
        fig.clf()

    # Get or Create 3D Axes
    # If the figure has no axes, create one. If it has one, reuse it.
    if len(fig.axes) == 0:
        ax = fig.add_subplot(111, projection="3d")
        # Set labels only when creating new axes
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
    else:
        ax = fig.axes[0]
        # Ensure the existing axis is 3D
        if ax.name != "3d":
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(title)

    xs, ys, zs = [], [], []
    i = 0

    # Extract data
    while values.exists(X(i)):
        pose = values.atPose3(X(i))
        p = pose.translation()

        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
        i += 1

    if len(xs) == 0:
        print(f"No poses found for {label}!")
        return

    # Plot the path line using the custom color and label
    ax.plot(xs, ys, zs, c=color, linestyle="-", linewidth=1, label=label)

    # Mark Start (Circle) and End (X)
    # We use the same color as the line for the markers to distinguish valid sets,
    # or you can stick to Green/Red if you prefer fixed start/end colors.
    # We set label=None ("_nolegend_") to avoid cluttering the legend with "Start/End" for every single line.
    ax.scatter(xs[0], ys[0], zs[0], c=color, marker="o", s=50, label="_nolegend_")
    ax.scatter(xs[-1], ys[-1], zs[-1], c=color, marker="x", s=50, label="_nolegend_")

    ax.legend()

    # Equal Aspect Ratio
    try:
        gtsam.utils.plot.set_axes_equal(fignum)
    except:
        pass

    if show:
        plt.show()


# Moving average data given window size only looks at past data points
def moving_average(data: List[float], window_size: int) -> List[float]:
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    averaged_data = []
    cumulative_sum = 0.0
    for i in range(len(data)):
        cumulative_sum += data[i]
        if i >= window_size:
            cumulative_sum -= data[i - window_size]
            averaged_data.append(cumulative_sum / window_size)
        else:
            averaged_data.append(cumulative_sum / (i + 1))
    return averaged_data
