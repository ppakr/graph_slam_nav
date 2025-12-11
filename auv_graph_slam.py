import gtsam
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import DataLoader
from gtsam.symbol_shorthand import B, V, X
from gtsam import NavState, Pose3, Rot3, Point3


class AUVGraphSLAM:
    def __init__(self):
        # pass
        self.DATA_DIR = "./full_dataset"
        DataLoader.get_dir(self.DATA_DIR)

        # Initialize Graph and Parameters
        self.graph = gtsam.NonlinearFactorGraph()

        # IMU Bias configuration (accelerometer and gyroscope)
        acc_bias = np.array([0.067, 0.115, 0.320])
        gyro_bias = np.array([0.067, 0.115, 0.320])
        self.bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

        # IMU Preintegration Parameters (Gravity = 9.81 m/s^2)
        self.params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.bias)
        # Initial Values container
        BIAS_KEY = B(0)  # Bias key at time step 0
        self.initial = gtsam.Values()
        self.initial.insert(BIAS_KEY, self.bias)

        # Noise Models
        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)  # Pose prior
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)  # Velocity prior
        self.depth_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.biasRandomWalk = gtsam.noiseModel.Isotropic.Sigma(
            6, 1e-3
        )  # small random walk
        self.biasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

        self.dt = 1e-6
        self.time_threshold = 1e-4  # Synchronization threshold
        self.node_add = 1.0  # Add a new node to graph every 1.0 second

    # TODO: helper functions (convert data to NavState, floating mean etc.)

    # TODO: main SLAM function (build graph and optimize)
    def optimize(self):
        print("Optimizing...")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
        self.result = optimizer.optimize()
        print("Optimization Complete.")
        print(f"Initial Error: {self.graph.error(self.initial)}")
        print(f"Final Error: {self.graph.error(self.result)}")

    # TODO: plotting function


if __name__ == "__main__":
    auv_slam = AUVGraphSLAM()
