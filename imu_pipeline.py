import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, List, Tuple, Optional

try:
    import gtsam
except ImportError:  # Allow file to exist without immediate gtsam dependency
    gtsam = None  # type: ignore


@dataclass
class IMUSample:
    t: float
    accel: np.ndarray  # shape (3,)
    gyro: np.ndarray   # shape (3,)


@dataclass
class IMUCalibration:
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Continuous-time noise density values from datasheet
    accel_noise: float = 2.0e-3  # m/s^2/sqrt(Hz)
    gyro_noise: float = 1.6968e-4  # rad/s/sqrt(Hz)
    accel_bias_rw: float = 3.0e-3  # m/s^3/sqrt(Hz)
    gyro_bias_rw: float = 1.9393e-5  # rad/s^2/sqrt(Hz)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))

    def bias_between_sigmas(self, dt: float) -> np.ndarray:
        """Compute discrete bias noise sigmas for a BetweenFactor over interval dt.

        Converts continuous-time random walk densities to discrete sigmas:
            sigma_discrete = sigma_rw * sqrt(dt)

        Returns:
            6-vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        sqrt_dt = np.sqrt(dt)
        accel_sigma = self.accel_bias_rw * sqrt_dt
        gyro_sigma = self.gyro_bias_rw * sqrt_dt
        return np.array([accel_sigma, accel_sigma, accel_sigma,
                         gyro_sigma, gyro_sigma, gyro_sigma])


class IMUBuffer:
    """Ring buffer for high-rate IMU measurements."""
    def __init__(self, maxlen: int = 4000):
        self.buf: Deque[IMUSample] = deque(maxlen=maxlen)

    def push(self, t: float, accel: np.ndarray, gyro: np.ndarray):
        self.buf.append(IMUSample(t, accel.astype(np.float32), gyro.astype(np.float32)))

    def get_interval(self, t0: float, t1: float) -> List[IMUSample]:
        return [s for s in self.buf if t0 <= s.t <= t1]

    def latest_time(self) -> Optional[float]:
        return self.buf[-1].t if self.buf else None


class IMUPreintegrator:
    """Manages GTSAM preintegration objects between keyframes / frames."""
    def __init__(self, calib: IMUCalibration):
        self.calib = calib
        self._reset_params()
        self.reset(accel_bias=calib.accel_bias, gyro_bias=calib.gyro_bias)

    def _reset_params(self):
        if gtsam is None:
            self.params = None
            return
        # Set up GTSAM preintegration parameters based on the IMU calibration
        # -gravity, because GTSAM does -g.
        p = gtsam.PreintegrationCombinedParams.MakeSharedU(-self.calib.gravity[2])
        p.setAccelerometerCovariance(np.eye(3) * self.calib.accel_noise ** 2)
        p.setGyroscopeCovariance(np.eye(3) * self.calib.gyro_noise ** 2)
        p.setIntegrationCovariance(np.eye(3) * 1e-8)
        p.setBiasAccCovariance(np.eye(3) * self.calib.accel_bias_rw ** 2)
        p.setBiasOmegaCovariance(np.eye(3) * self.calib.gyro_bias_rw ** 2)
        self.params = p

    def reset(self, accel_bias: np.ndarray, gyro_bias: np.ndarray):
        if gtsam is None:
            self.preint = None
            return
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self.preint = gtsam.PreintegratedImuMeasurements(self.params, bias)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def integrate(self, samples: List[IMUSample]):
        if gtsam is None or self.preint is None:
            return
        for i, s in enumerate(samples):
            if self.start_time is None:
                self.start_time = s.t
            if i > 0:
                dt = s.t - samples[i - 1].t
                if dt <= 0 or dt > 1.0:  # basic sanity
                    continue
                self.preint.integrateMeasurement(s.accel, s.gyro, dt)
            self.end_time = s.t

    def get_factor_and_bias(self, state_i, state_j, bias_key):
        if gtsam is None or self.preint is None:
            return None
        return gtsam.ImuFactor(state_i, state_j, bias_key, self.preint)


class IMUStatePropagator:
    """Lightweight kinematic propagation (for initial pose guesses)."""
    def __init__(self, gravity: np.ndarray):
        self.g = gravity

    def propagate(self, pose_wb: np.ndarray, vel_w: np.ndarray, q_wb: np.ndarray,
                  samples: List[IMUSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple Euler integration (placeholder)."""
        if len(samples) < 2:
            return pose_wb, vel_w, q_wb
        p = pose_wb.copy()
        v = vel_w.copy()
        q = q_wb.copy()  # quaternion xyzw
        for i in range(1, len(samples)):
            s_prev = samples[i - 1]
            s = samples[i]
            dt = s.t - s_prev.t
            if dt <= 0 or dt > 0.1:
                continue
            # naive: treat accel as body aligned with world orientation (needs rotation from q)
            a_w = s.accel + self.g
            p += v * dt + 0.5 * a_w * dt**2
            v += a_w * dt
            # gyro integrate (small angle)
            omega = s.gyro * dt
            dq = np.array([omega[0]/2, omega[1]/2, omega[2]/2, 1.0], dtype=np.float32)
            q = q + dq  # not normalized for brevity (replace with proper quat math)
        return p, v, q / np.linalg.norm(q)


# High-level container (to be integrated with optimizer)
class IMUPipeline:
    def __init__(self, calib: IMUCalibration):
        self.buffer = IMUBuffer()
        self.preint = IMUPreintegrator(calib)
        self.propagator = IMUStatePropagator(calib.gravity)
        self.last_keyframe_time: Optional[float] = None

    def add_measurement(self, t: float, accel: np.ndarray, gyro: np.ndarray):
        self.buffer.push(t, accel, gyro)

    def build_between_keyframes(self, t_kf_prev: float, t_kf_curr: float):
        samples = self.buffer.get_interval(t_kf_prev, t_kf_curr)
        self.preint.reset(self.preint.calib.accel_bias, self.preint.calib.gyro_bias)  # type: ignore
        self.preint.integrate(samples)
        return samples


if __name__ == "__main__":
    import os
    import sys
    # Add the parent directory to the path to find data_manager2
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data_manager import DataManager

    # --- 1. Setup: Load data ---
    data_dir = "f:/Code/exercise_10/data/MH_01_easy/mav0"
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please update the 'data_dir' variable in the test script.")
    
    data_manager = DataManager(data_dir)
    data_manager.load_data()
    print(f"Loaded {len(data_manager.cam_df)} cam0 frames and {len(data_manager.imu_df)} IMU measurements.")

    # --- 2. Define an interval for the test ---
    # We'll preintegrate IMU measurements between the first two camera frames.
    cam_times = data_manager.cam_df['timestamp'].values
    if len(cam_times) < 2:
        raise ValueError("Not enough camera frames in the dataset for a test interval.")
    
    for i in range(1, min(5, len(cam_times))):
        t_start = cam_times[0]
        t_end = cam_times[i]
        print(f"\nTesting preintegration between t_start={t_start} and t_end={t_end}")
        gt_pos, gt_quat = data_manager.get_synced_gt_poses(t_start)
        print(gt_pos, gt_quat, "GT at start time")
        # --- 3. Get IMU samples for the interval ---
        imu_samples_list = []
        for imu_row in data_manager.iter_imu_between(t_start, t_end):
            timestamp = imu_row["timestamp"]
            accel = imu_row[["a_x", "a_y", "a_z"]].values.astype(float)
            gyro = imu_row[["w_x", "w_y", "w_z"]].values.astype(float)
            imu_samples_list.append(IMUSample(timestamp, accel, gyro))
        
        
        print(f"Found {len(imu_samples_list)} IMU samples in the interval.")

        # --- 4. Perform Preintegration ---
        if gtsam is None:
            print("\n[ERROR] GTSAM is not installed. Cannot perform preintegration test.")
        else:
            # Initialize pipeline components
            imu_calib = IMUCalibration()
            preintegrator = IMUPreintegrator(imu_calib)

            # Integrate the samples
            preintegrator.integrate(imu_samples_list)

            # --- 5. Print Results ---
            pim = preintegrator.preint
            if pim is not None:
                print("\n--- Preintegration Results ---")
                print(f"Delta Time (s): {pim.deltaTij():.4f}")
                
                delta_p = pim.deltaPij()
                print(f"Delta Position (m): [ {delta_p[0]:.4f}, {delta_p[1]:.4f}, {delta_p[2]:.4f} ]")
                
                delta_v = pim.deltaVij()
                print(f"Delta Velocity (m/s): [ {delta_v[0]:.4f}, {delta_v[1]:.4f}, {delta_v[2]:.4f} ]")

                delta_r_ypr = pim.deltaRij().ypr() # Yaw, Pitch, Roll in radians
                print(f"Delta Rotation (YPR, deg): [ {np.rad2deg(delta_r_ypr[0]):.4f}, {np.rad2deg(delta_r_ypr[1]):.4f}, {np.rad2deg(delta_r_ypr[2]):.4f} ]")
                
                print("\n--- Covariance Matrix (diagonal) ---")
                cov = pim.preintMeasCov()
                print(f"Rotation Cov (rad^2): [ {cov[0,0]:.2e}, {cov[1,1]:.2e}, {cov[2,2]:.2e} ]")
                print(f"Velocity Cov (m^2/s^2): [ {cov[3,3]:.2e}, {cov[4,4]:.2e}, {cov[5,5]:.2e} ]")
                print(f"Position Cov (m^2): [ {cov[6,6]:.2e}, {cov[7,7]:.2e}, {cov[8,8]:.2e} ]")
                
