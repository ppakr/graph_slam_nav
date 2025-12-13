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
from utils import plot_trajectory_fast, plot_trajectory_fast_multi, moving_average
from auv_graph_slam import AUVGraphSLAM


# Create main function
def main():
    # Initialize data directory
    DATA_DIR = "./full_dataset"
    DataLoader.get_dir(DATA_DIR)

    imu_adis_ros_file = "imu_adis_ros.csv"
    imu_xsens_file = "imu_xsens_mti_ros.csv"
    dvl_file = "dvl_linkquest.csv"
    depth_file = "depth_sensor.csv"
    odom_file = "odometry.csv"

    imu_adis_ros = DataLoader.read_imu(imu_adis_ros_file)
    imu_xsens_ros = DataLoader.read_imu(imu_xsens_file)
    dvl = DataLoader.read_dvl(dvl_file)
    depth = DataLoader.read_depth(depth_file)
    odom = DataLoader.read_odom(odom_file)

    # time synchronization
    start_time = imu_adis_ros["time"][0]
    # start_time = imu_xsens_ros['time'][0]

    # print(f"Start time: {start_time}")
    # offeset all timestamps to start from zero
    imu_adis_ros["time"] -= start_time
    imu_xsens_ros["time"] -= start_time
    dvl["time"] -= start_time
    depth["time"] -= start_time
    odom["time"] -= start_time

    # show new time
    print(f"New start time: {imu_adis_ros['time'][0]}")
    print(f"New end time: {imu_adis_ros['time'][-1]}")

    # convert nanoseconds to seconds
    imu_adis_ros["time"] /= 1e9
    dvl["time"] /= 1e9
    depth["time"] /= 1e9
    odom["time"] /= 1e9
    imu_xsens_ros["time"] /= 1e9

    # store original imu data for comparison
    imu_adis_ros_original = imu_adis_ros.copy()
    imu_xsens_ros_original = imu_xsens_ros.copy()

    # smooth imu data using moving average filter
    imu_adis_ros["ax"] = moving_average(imu_adis_ros["ax"], window_size=15)
    imu_adis_ros["ay"] = moving_average(imu_adis_ros["ay"], window_size=15)
    imu_adis_ros["az"] = moving_average(imu_adis_ros["az"], window_size=15)
    imu_adis_ros["omega_x"] = moving_average(imu_adis_ros["omega_x"], window_size=15)
    imu_adis_ros["omega_y"] = moving_average(imu_adis_ros["omega_y"], window_size=15)
    imu_adis_ros["omega_z"] = moving_average(imu_adis_ros["omega_z"], window_size=15)

    # smooth xsens imu data using moving average filter
    imu_xsens_ros["ax"] = moving_average(imu_xsens_ros["ax"], window_size=5)
    imu_xsens_ros["ay"] = moving_average(imu_xsens_ros["ay"], window_size=5)
    imu_xsens_ros["az"] = moving_average(imu_xsens_ros["az"], window_size=5)
    imu_xsens_ros["omega_x"] = moving_average(imu_xsens_ros["omega_x"], window_size=5)
    imu_xsens_ros["omega_y"] = moving_average(imu_xsens_ros["omega_y"], window_size=5)
    imu_xsens_ros["omega_z"] = moving_average(imu_xsens_ros["omega_z"], window_size=5)

    # store original dvl data for comparison
    dvl_original = dvl.copy()

    # smooth dvl data using moving average filter
    dvl["vx"] = moving_average(dvl["vx"], window_size=5)
    dvl["vy"] = moving_average(dvl["vy"], window_size=5)
    dvl["vz"] = moving_average(dvl["vz"], window_size=5)

    # plot smoothed DVL data for verification
    plt.figure()
    plt.plot(dvl["time"], dvl["vx"], label="vx smoothed")
    plt.plot(dvl["time"], dvl["vy"], label="vy smoothed")
    plt.plot(dvl["time"], dvl["vz"], label="vz smoothed")
    plt.plot(
        dvl_original["time"],
        dvl_original["vx"],
        label="vx original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        dvl_original["time"],
        dvl_original["vy"],
        label="vy original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        dvl_original["time"],
        dvl_original["vz"],
        label="vz original",
        linestyle="--",
        alpha=0.5,
    )
    plt.title("DVL Velocities Smoothed")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot imu data for verification
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(
        imu_adis_ros["time"],
        imu_adis_ros_original["ax"],
        label="ax original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_adis_ros["time"],
        imu_adis_ros_original["ay"],
        label="ay original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_adis_ros["time"],
        imu_adis_ros_original["az"],
        label="az original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(imu_adis_ros["time"], imu_adis_ros["ax"], label="ax")
    plt.plot(imu_adis_ros["time"], imu_adis_ros["ay"], label="ay")
    plt.plot(imu_adis_ros["time"], imu_adis_ros["az"], label="az")
    plt.title("ADIS IMU Accelerations")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.legend()

    # plot angular velocities
    plt.subplot(2, 1, 2)
    plt.plot(
        imu_adis_ros["time"],
        imu_adis_ros_original["omega_x"],
        label="omega_x original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_adis_ros["time"],
        imu_adis_ros_original["omega_y"],
        label="omega_y original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_adis_ros["time"],
        imu_adis_ros_original["omega_z"],
        label="omega_z original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(imu_adis_ros["time"], imu_adis_ros["omega_x"], label="omega_x")
    plt.plot(imu_adis_ros["time"], imu_adis_ros["omega_y"], label="omega_y")
    plt.plot(imu_adis_ros["time"], imu_adis_ros["omega_z"], label="omega_z")
    plt.title("ADIS IMU Angular Velocities")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot imu data for verification
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(
        imu_xsens_ros["time"],
        imu_xsens_ros_original["ax"],
        label="ax original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_xsens_ros["time"],
        imu_xsens_ros_original["ay"],
        label="ay original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_xsens_ros["time"],
        imu_xsens_ros_original["az"],
        label="az original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(imu_xsens_ros["time"], imu_xsens_ros["ax"], label="ax")
    plt.plot(imu_xsens_ros["time"], imu_xsens_ros["ay"], label="ay")
    plt.plot(imu_xsens_ros["time"], imu_xsens_ros["az"], label="az")
    plt.title("XSENS IMU Accelerations")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.legend()

    # plot angular velocities
    plt.subplot(2, 1, 2)
    plt.plot(
        imu_xsens_ros["time"],
        imu_xsens_ros_original["omega_x"],
        label="omega_x original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_xsens_ros["time"],
        imu_xsens_ros_original["omega_y"],
        label="omega_y original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        imu_xsens_ros["time"],
        imu_xsens_ros_original["omega_z"],
        label="omega_z original",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(imu_xsens_ros["time"], imu_xsens_ros["omega_x"], label="omega_x")
    plt.plot(imu_xsens_ros["time"], imu_xsens_ros["omega_y"], label="omega_y")
    plt.plot(imu_xsens_ros["time"], imu_xsens_ros["omega_z"], label="omega_z")
    plt.title("XSENS IMU Angular Velocities")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Contsruct graph
    gslam_imu_graph = AUVGraphSLAM()
    gslam_imu_graph.construct_imu_graph(
        imu_data=imu_adis_ros, odom_data=odom, dvl_data=dvl
    )

    # Optimize graph
    result_imu = gslam_imu_graph.optimize_graph(100)
    initial_imu = gslam_imu_graph.initial

    # Plot trajectory compare initial and result
    plot_trajectory_fast_multi(
        initial_imu,
        title="Comparison",
        fignum=1,
        show=False,
        clear=True,
        color="g",
        label="Ground Truth",
    )
    plot_trajectory_fast_multi(
        result_imu,
        title="Comparison",
        fignum=1,
        show=True,
        clear=False,
        color="r",
        label="Graph SLAM (IMU) Result",
    )

    # print all the biases
    last_key_index_imu = result_imu.size() // 3 - 1
    print("IMU Biases over time:")
    for k in range(last_key_index_imu + 1):
        b = result_imu.atConstantBias(B(k))
        print(f"Key {k}: Acc Bias: {b.accelerometer()}, Gyro Bias: {b.gyroscope()}")
    # plot all the Imu bias
    bias_xs, bias_ys, bias_zs = [], [], []
    # Gyro biases
    bias_gx, bias_gy, bias_gz = [], [], []
    for k in range(last_key_index_imu + 1):
        b = result_imu.atConstantBias(B(k))
        bias_xs.append(b.accelerometer()[0])
        bias_ys.append(b.accelerometer()[1])
        bias_zs.append(b.accelerometer()[2])
        bias_gx.append(b.gyroscope()[0])
        bias_gy.append(b.gyroscope()[1])
        bias_gz.append(b.gyroscope()[2])
    plt.figure(3, figsize=(10, 6))
    plt.clf()
    plt.plot(bias_xs, "-r", label="Acc Bias X")
    plt.plot(bias_ys, "-g", label="Acc Bias Y")
    plt.plot(bias_zs, "-b", label="Acc Bias Z")
    plt.plot(bias_gx, "--r", label="Gyro Bias X")
    plt.plot(bias_gy, "--g", label="Gyro Bias Y")
    plt.plot(bias_gz, "--b", label="Gyro Bias Z")
    plt.xlabel("Key Index")
    plt.ylabel("Accelerometer Bias (m/s^2)")
    plt.title("Estimated IMU Accelerometer Bias over Time - IMU Only Graph")
    plt.legend()
    plt.show()

    gslam_full_graph = AUVGraphSLAM()
    gslam_full_graph.construct_full_graph(
        imu_data=imu_adis_ros, odom_data=odom, dvl_data=dvl, depth_data=depth
    )
    result_full = gslam_full_graph.optimize_graph(n_iterations=100)
    initial_full = gslam_full_graph.initial
    # Plot trajectory compare initial and result
    plot_trajectory_fast_multi(
        initial_full,
        title="Comparison Full",
        fignum=2,
        show=False,
        clear=True,
        color="g",
        label="Ground Truth",
    )
    plot_trajectory_fast_multi(
        result_full,
        title="Comparison Full",
        fignum=2,
        show=True,
        clear=False,
        color="b",
        label="Graph SLAM Full Result",
    )


if __name__ == "__main__":
    main()
