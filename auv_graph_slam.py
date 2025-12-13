import gtsam
import numpy as np
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import B, V, X, D
from gtsam import NavState, Pose3, Rot3, Point3
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List
from functools import partial


# Create Class
class AUVGraphSLAM:
    def __init__(self):

        # initialize biases
        self.accBias = np.array([0.0, 0.0, 0.0])
        self.gyroBias = np.array([0.0, 0.0, 0.0])
        self.bias = gtsam.imuBias.ConstantBias(self.accBias, self.gyroBias)

        self.t_body_sensor = gtsam.Point3(-0.38, 0.0, 0.07)  # translation
        # self.R_body_sensor = gtsam.Rot3.Ypr(np.deg2rad(90), 0, np.deg2rad(180)) # Yaw, Pitch, Roll
        # self.R_body_sensor = gtsam.Rot3.Ypr(np.deg2rad(90), 0, 0) # Yaw, Pitch, Roll
        self.R_body_sensor = gtsam.Rot3.Ypr(0, 0, 0)  # Yaw, Pitch, Roll
        # self.t_body_sensor = gtsam.Point3(0.10, 0.0, -0.42) # rotation
        # self.R_body_sensor = gtsam.Rot3.Ypr(np.deg2rad(-90), 0, np.deg2rad(180)) # Yaw, Pitch, Roll

        self.body_P_sensor = gtsam.Pose3(self.R_body_sensor, self.t_body_sensor)

        # set gravity
        self.pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.pim_params.setBodyPSensor(self.body_P_sensor)

        # Some arbitrary noise sigmas
        gyro_sigma = 8.7e-5
        accel_sigma = 2.0e-4
        I_3x3 = np.eye(3)
        self.pim_params.setGyroscopeCovariance(gyro_sigma**2 * I_3x3)
        self.pim_params.setAccelerometerCovariance(accel_sigma**2 * I_3x3)

        self.pim_params.setIntegrationCovariance(1e-7**2 * I_3x3)

        # preintegrate imu measurements
        self.pim = gtsam.PreintegratedImuMeasurements(self.pim_params, self.bias)

        # Noise models for factors
        self.priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.bias_rw_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-2)

        self.dvl_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.depth_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)

        self.time_threshold = 1e-4  # seconds

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = None

    def get_initial_state_from_data(self, query_time, odom_data, dvl_data):
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
        # reverse z for underwater
        pos_interp[2] = -pos_interp[2]

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

    def depth_error(
        self,
        measurement: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> float:
        key = this.keys()[0]
        estimate = values.atPose3(key)
        depth = estimate.z()
        # error = np.array([measurement - depth + 1.0373129093000006]) # Adjust for sensor offset

        error = measurement - depth - 1.0373129093000006
        # print(f"Depth measurement: {measurement}, Estimated depth: {depth}, Error: {error}")
        if jacobians is not None:
            val = np.zeros((1, 6))
            val[0, 2] = 1
            jacobians[0] = val
        return error

    def velocity_error(
        self,
        measurement: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> float:

        key_pose = this.keys()[0]
        key_vel = this.keys()[1]

        pose = values.atPose3(key_pose)
        vel_est = values.atPoint3(key_vel)  # Estimated velocity in World Frame

        R = pose.rotation().matrix()
        v_meas_body = np.zeros(3)
        v_meas_body = measurement  # Measured velocity in Body Frame

        # Transform measurement to world frame to compare with state
        v_meas_world = R @ v_meas_body

        error = v_meas_world - vel_est

        if jacobians is not None:
            # --- Jacobian w.r.t Pose (X) [3x6] ---
            if jacobians[0] is not None:
                # We need d(Error)/d(Pose).
                # Pose has 6 params: [Rotation (3), Translation (3)]
                # Translation does not affect velocity rotation, so cols 3-5 are 0.
                # Rotation affects R. d(R*v)/d(theta) = -R * [v]_skew

                vx, vy, vz = v_meas_body
                v_skew = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])

                H_pose = np.zeros((3, 6))
                H_pose[:, 0:3] = -R @ v_skew  # Rotation block (3x3)
                # Translation block (3x3) remains 0

                jacobians[0] = H_pose
                # --- Jacobian w.r.t Velocity (V) [3x3] ---
            if jacobians[1] is not None:
                # Error = Constant - V_est
                # d(Error)/d(V_est) = -Identity
                jacobians[1] = -np.eye(3)

        return error

    def construct_imu_graph(self, imu_data, odom_data, dvl_data):
        # add prior initial node from odometry
        init_pose, init_vel = self.get_initial_state_from_data(
            imu_data["time"][0], odom_data, dvl_data
        )

        init_state = NavState(init_pose, init_vel)

        # add initial graph
        self.graph.add(
            gtsam.PriorFactorPose3(X(0), init_state.pose(), self.priorNoise)
        )  # position
        self.graph.add(
            gtsam.PriorFactorVector(V(0), init_state.velocity(), self.velNoise)
        )  # velocity

        # Bias Prior (New)
        # We use a loose prior if we are unsure, or a tight one if calibrated
        bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.graph.add(gtsam.PriorFactorConstantBias(B(0), self.bias, bias_prior_noise))

        # add initial values
        self.initial.insert(X(0), init_state.pose())
        self.initial.insert(V(0), init_state.velocity())
        self.initial.insert(B(0), self.bias)
        last_time = imu_data["time"][0]

        key_index = 0

        for i in range(1, len(imu_data["time"])):
            # get measurement data at index i
            dt = imu_data["time"][i] - imu_data["time"][i - 1]
            # print(f"dt: {dt}")
            measured_acc = np.array(
                [
                    imu_data["ax"][i],
                    imu_data["ay"][i],
                    imu_data["az"][i],
                ]
            )
            # print(f"measured_acc: {measured_acc}")
            measured_omg = np.array(
                [imu_data["omega_x"][i], imu_data["omega_y"][i], imu_data["omega_z"][i]]
            )

            # integrate measurement
            self.pim.integrateMeasurement(measured_acc, measured_omg, dt)

            freq = 3.0  # Hz (DVL freq)
            threshold = 1.0 / freq
            current_time = imu_data["time"][i]
            if current_time - last_time >= threshold:
                # Create IMU factor every threshold seconds
                factor = gtsam.ImuFactor(
                    X(key_index),
                    V(key_index),
                    X(key_index + 1),
                    V(key_index + 1),
                    B(key_index),
                    self.pim,
                )
                self.graph.add(factor)

                self.graph.add(
                    gtsam.BetweenFactorConstantBias(
                        B(key_index),
                        B(key_index + 1),
                        gtsam.imuBias.ConstantBias(),  # Expected change is 0
                        self.bias_rw_noise,
                    )
                )

                # get nav state
                # actual_state = pim.predict(prev_state, bias)
                pose_est, vel_est = self.get_initial_state_from_data(
                    current_time, odom_data, dvl_data
                )

                # set inital bias as previous

                # insert initial estimate
                self.initial.insert(X(key_index + 1), pose_est)
                self.initial.insert(V(key_index + 1), vel_est)
                self.initial.insert(B(key_index + 1), self.bias)
                key_index += 1

                self.pim.resetIntegration()
                # prev_state = actual_state # for next time

                last_time = current_time  # for next time

    def construct_full_graph(self, imu_data, odom_data, dvl_data, depth_data):
        # add prior initial node from odometry
        init_pose, init_vel = self.get_initial_state_from_data(
            imu_data["time"][0], odom_data, dvl_data
        )
        print(f"Initial Pose: {init_pose}")
        init_state = NavState(init_pose, init_vel)

        # add initial graph
        self.graph.add(
            gtsam.PriorFactorPose3(X(0), init_state.pose(), self.priorNoise)
        )  # position
        self.graph.add(
            gtsam.PriorFactorVector(V(0), init_state.velocity(), self.velNoise)
        )  # velocity

        # Bias Prior (New)
        # We use a loose prior if we are unsure, or a tight one if calibrated
        bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.graph.add(gtsam.PriorFactorConstantBias(B(0), self.bias, bias_prior_noise))

        # add initial values
        self.initial.insert(X(0), init_state.pose())
        self.initial.insert(V(0), init_state.velocity())
        self.initial.insert(B(0), self.bias)
        last_time = imu_data["time"][0]

        key_index = 0

        self.dvl_vel_world_list = []

        for i in range(1, len(imu_data["time"])):
            # get measurement data at index i
            dt = imu_data["time"][i] - imu_data["time"][i - 1]
            # print(f"dt: {dt}")
            measured_acc = np.array(
                [
                    imu_data["ax"][i],
                    imu_data["ay"][i],
                    imu_data["az"][i],
                ]
            )
            # print(f"measured_acc: {measured_acc}")
            measured_omg = np.array(
                [imu_data["omega_x"][i], imu_data["omega_y"][i], imu_data["omega_z"][i]]
            )

            # integrate measurement
            self.pim.integrateMeasurement(measured_acc, measured_omg, dt)

            freq = 3.0  # Hz (DVL freq)
            threshold = 1.0 / freq
            current_time = imu_data["time"][i]
            if current_time - last_time >= threshold:
                # Create IMU factor every threshold seconds
                factor = gtsam.ImuFactor(
                    X(key_index),
                    V(key_index),
                    X(key_index + 1),
                    V(key_index + 1),
                    B(key_index),
                    self.pim,
                )
                self.graph.add(factor)

                self.graph.add(
                    gtsam.BetweenFactorConstantBias(
                        B(key_index),
                        B(key_index + 1),
                        gtsam.imuBias.ConstantBias(),  # Expected change is 0
                        self.bias_rw_noise,
                    )
                )

                # Add Depth Factor if available

                depth_idx = (np.abs(depth_data["time"] - current_time)).argmin()
                depth_value = -1.0 * depth_data["depth"][depth_idx]
                depth_time = depth_data["time"][depth_idx]

                if abs(depth_time - current_time) < self.time_threshold:
                    print(
                        f"Adding depth factor at key index {key_index} with depth value {depth_value} at time {depth_time}"
                    )
                    depth_factor = gtsam.CustomFactor(
                        self.depth_noise_model,
                        [X(key_index + 1)],
                        partial(self.depth_error, np.array([depth_value])),
                    )

                    self.graph.add(depth_factor)

                # Add DVL Velocity Factor if available

                dvl_idx = (np.abs(dvl_data["time"] - current_time)).argmin()
                dvl_velocity = np.array(
                    [
                        dvl_data["vx"][dvl_idx],
                        dvl_data["vy"][dvl_idx],
                        dvl_data["vz"][dvl_idx],
                    ]
                )
                dvl_time = dvl_data["time"][dvl_idx]

                if abs(dvl_time - current_time) < self.time_threshold:
                    print(
                        f"Adding DVL factor at key index {key_index + 1} with velocity {dvl_velocity} at time {dvl_time}"
                    )
                    dvl_factor = gtsam.CustomFactor(
                        self.dvl_noise_model,
                        [X(key_index + 1), V(key_index + 1)],
                        partial(self.velocity_error, dvl_velocity),
                    )
                    self.graph.add(dvl_factor)

                # get nav state
                # actual_state = pim.predict(prev_state, bias)
                pose_est, vel_est = self.get_initial_state_from_data(
                    current_time, odom_data, dvl_data
                )

                # insert initial estimate
                self.initial.insert(X(key_index + 1), pose_est)
                self.initial.insert(V(key_index + 1), vel_est)
                self.initial.insert(B(key_index + 1), self.bias)
                key_index += 1

                self.pim.resetIntegration()
                # prev_state = actual_state # for next time

                last_time = current_time  # for next time

    def optimize_graph(self, n_iterations=100):
        # Optimize the graph
        parameters = gtsam.LevenbergMarquardtParams()
        parameters.setVerbosityLM("SUMMARY")
        parameters.setMaxIterations(n_iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, parameters
        )
        self.result = optimizer.optimize()
        print("Optimization complete.")

        return self.result
