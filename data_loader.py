import os
import sys
import numpy as np
import pandas as pd

# Configure paths
DATA_DIR = "./full_dataset"


class DataLoader:
    """Helper class to handle data reading based on the provided dataloader.py"""

    @staticmethod
    def read_iekf_states(filename):
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        return {
            "x": df["p_x"].values,
            "y": df["p_y"].values,
            "z": df["p_z"].values,
            "u": df["v_x"].values,
            "v": df["v_y"].values,
            "r": df["v_z"].values,
            "phi": df["theta_x"].values,
            "theta": df["theta_y"].values,
            "psi": df["theta_z"].values,
        }

    @staticmethod
    def read_state_times(filename):
        path = os.path.join(DATA_DIR, filename)
        return pd.read_csv(path)["time"].values.astype(np.float64)

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
        return df["%time"].values.astype(np.float64), df["field.depth"].values.astype(
            np.float64
        )

    @staticmethod
    def read_dvl(filename):
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        times = df["%time"].values.astype(np.float64)
        vx = df["field.velocityEarth0"].values.astype(np.float64)
        vy = df["field.velocityEarth1"].values.astype(np.float64)
        vz = df["field.velocityEarth2"].values.astype(np.float64)
        # Note: DVL Z is often inverted or specific to sensor mounting; adjusting based on original code
        return times, np.vstack((vx, vy, vz))
