import os
import sys
import numpy as np
import pandas as pd

# Configure paths
# DATA_DIR = "./full_dataset"


##################################
# file names
# imu: imu_adis.csv
# dvl: dvl_linkquest.csv
# depth: depth_sensor.csv
# ground truth: odometry.csv (ros msg: odometry)
##################################
class DataLoader:
    """Helper class to handle data reading based on the provided dataloader.py"""

    @staticmethod
    def get_dir(DATA_DIR_PATH):
        global DATA_DIR
        DATA_DIR = DATA_DIR_PATH
        print(f"Data directory set to: {DATA_DIR}")

    @staticmethod
    def read_odom(filename):
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        data = {
            "time": df["%time"].values.astype(np.float64),
            "x": df["field.pose.pose.position.x"].values.astype(np.float64),
            "y": df["field.pose.pose.position.y"].values.astype(np.float64),
            "z": df["field.pose.pose.position.z"].values.astype(np.float64),
            "qx": df["field.pose.pose.orientation.x"].values.astype(np.float64),
            "qy": df["field.pose.pose.orientation.y"].values.astype(np.float64),
            "qz": df["field.pose.pose.orientation.z"].values.astype(np.float64),
            "qw": df["field.pose.pose.orientation.w"].values.astype(np.float64),
        }
        return data

    @staticmethod
    def read_imu(filename):
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        data = {
            "time": df["%time"].values.astype(np.float64),
            "omega_x": df["field.angular_velocity.x"].values.astype(np.float64),
            "omega_y": df["field.angular_velocity.y"].values.astype(np.float64),
            "omega_z": df["field.angular_velocity.z"].values.astype(np.float64),
            "ax": df["field.linear_acceleration.x"].values.astype(np.float64),
            "ay": df["field.linear_acceleration.y"].values.astype(np.float64),
            "az": df["field.linear_acceleration.z"].values.astype(np.float64),
        }
        return data

    @staticmethod
    def read_depth(filename):
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        data = {
            "time": df["%time"].values.astype(np.float64),
            "depth": df["field.depth"].values.astype(np.float64),
        }
        return data

    @staticmethod
    def read_dvl(filename):
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        data = {
            "time": df["%time"].values.astype(np.float64),
            "vx": df["field.velocityEarth0"].values.astype(np.float64),
            "vy": df["field.velocityEarth1"].values.astype(np.float64),
            "vz": df["field.velocityEarth2"].values.astype(np.float64),
        }

        # Note: DVL Z is often inverted or specific to sensor mounting; adjusting based on original code
        return data


# --- Testing Logic ---
def run_tests():
    print("--- Testing DataLoader ---")

    # Test IMU (Note: Using _ros.csv based on your column names)
    print("Reading IMU...")
    # NOTE: Your code uses 'field.' prefixes, so we must use the ROS version of the CSV
    imu = DataLoader.read_imu("imu_adis_ros.csv")
    if imu:
        print(f"   Success! Loaded {len(imu['time'])} IMU measurements.")
        print(
            f"   Accel sample: {imu['ax'][0]:.4f}, {imu['ay'][0]:.4f}, {imu['az'][0]:.4f}"
        )

    # Test Depth
    print("Reading Depth...")
    d_time, d_val = DataLoader.read_depth("depth_sensor.csv")
    if d_time is not None:
        print(f"   Success! Loaded {len(d_val)} depth measurements.")
        print(f"   Max Depth: {np.max(d_val):.2f}m")

    # Test DVL
    print("Reading DVL...")
    dvl_time, dvl_vel = DataLoader.read_dvl("dvl_linkquest.csv")
    if dvl_time is not None:
        print(f"   Success! Loaded {len(dvl_time)} DVL measurements.")
        print(f"   Velocity Shape: {dvl_vel.shape}")

    # Test Odometry
    print("Reading Odometry...")
    odom = DataLoader.read_odom("odometry.csv")
    if odom is not None:
        print(f"   Success! Loaded {len(odom['time'])} odometry measurements.")
        print(
            f"   Position sample: {odom['x'][0]:.2f}, {odom['y'][0]:.2f}, {odom['z'][0]:.2f}"
        )


# if __name__ == "__main__":
#     run_tests()
