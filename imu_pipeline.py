import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import gtsam
except ImportError:  # Allow file to exist without immediate gtsam dependency
    gtsam = None  # type: ignore


@dataclass
class IMUSample:
    """One timestamped IMU reading in the body frame."""
    t: float
    accel: np.ndarray  # shape (3,)
    gyro: np.ndarray   # shape (3,)


@dataclass
class IMUCalibration:
    """IMU biases, continuous-time noise densities, and the gravity vector."""
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Continuous-time noise density values from datasheet
    accel_noise: float = 2.0e-3  # m/s^2/sqrt(Hz)
    gyro_noise: float = 1.6968e-4  # rad/s/sqrt(Hz)
    accel_bias_rw: float = 3.0e-3  # m/s^3/sqrt(Hz)
    gyro_bias_rw: float = 1.9393e-5  # rad/s^2/sqrt(Hz)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))

    def bias_between_sigmas(self, dt: float) -> np.ndarray:
        """Discrete bias sigmas for a BetweenFactor spanning dt.

        A random walk integrates variance linearly in time, so the discrete sigma is
        the continuous density scaled by sqrt(dt).

        Args:
            dt: Interval length in seconds.

        Returns:
            6-vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z].
        """
        sqrt_dt = np.sqrt(dt)
        accel_sigma = self.accel_bias_rw * sqrt_dt
        gyro_sigma = self.gyro_bias_rw * sqrt_dt
        return np.array([accel_sigma] * 3 + [gyro_sigma] * 3)


class IMUPreintegrator:
    """Accumulates IMU measurements into a GTSAM preintegration between keyframes."""

    def __init__(self, calib: IMUCalibration):
        self.calib = calib
        self._reset_params()
        self.reset(accel_bias=calib.accel_bias, gyro_bias=calib.gyro_bias)

    def _reset_params(self):
        """Build GTSAM preintegration parameters from the calibration.

        NOTE: the bias covariances set here are only consumed by CombinedImuFactor;
        the plain ImuFactor used by the backend ignores them, so its reported
        uncertainty excludes bias random walk.

        Returns:
            None.
        """
        if gtsam is None:
            self.params = None
            return
        # MakeSharedU takes +g magnitude because GTSAM applies gravity as -g.
        p = gtsam.PreintegrationCombinedParams.MakeSharedU(-self.calib.gravity[2])
        p.setAccelerometerCovariance(np.eye(3) * self.calib.accel_noise ** 2)
        p.setGyroscopeCovariance(np.eye(3) * self.calib.gyro_noise ** 2)
        p.setIntegrationCovariance(np.eye(3) * 1e-8)
        p.setBiasAccCovariance(np.eye(3) * self.calib.accel_bias_rw ** 2)
        p.setBiasOmegaCovariance(np.eye(3) * self.calib.gyro_bias_rw ** 2)
        self.params = p

    def reset(self, accel_bias: np.ndarray, gyro_bias: np.ndarray):
        """Restart preintegration from a new bias estimate.

        Args:
            accel_bias: Accelerometer bias to linearize about.
            gyro_bias: Gyroscope bias to linearize about.

        Returns:
            None.
        """
        if gtsam is None:
            self.preint = None
            return
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self.preint = gtsam.PreintegratedImuMeasurements(self.params, bias)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def integrate(self, samples: List[IMUSample]):
        """Fold samples onto the running preintegration without resetting it.

        Not resetting is what lets one IMU factor span a whole keyframe interval
        across intermediate non-keyframes.

        Args:
            samples: Chronologically ordered IMU samples.

        Returns:
            None.
        """
        if gtsam is None or self.preint is None:
            return
        for s in samples:
            if self.start_time is None:
                self.start_time = s.t
                self.end_time = s.t
                continue
            dt = s.t - self.end_time
            # Reject non-monotonic or implausibly long gaps rather than integrating them.
            if dt <= 0 or dt > 1.0:
                self.end_time = s.t
                continue
            self.preint.integrateMeasurement(s.accel, s.gyro, dt)
            self.end_time = s.t


class IMUPipeline:
    """Owns the preintegrator for the backend's keyframe-to-keyframe intervals."""

    def __init__(self, calib: IMUCalibration):
        self.preint = IMUPreintegrator(calib)
