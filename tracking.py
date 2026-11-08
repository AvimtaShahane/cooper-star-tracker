"""
tracking.py -- Temporal filtering. Single frames are only half the problem.

A real star tracker has two modes:
  LOST IN SPACE  no prior knowledge, solve from scratch (classical.py / model.py)
  TRACKING       propagate the last estimate, correct it with each new frame

This implements the tracking mode as a Multiplicative Extended Kalman Filter
(MEKF), the standard formulation in spacecraft attitude estimation.

Why MULTIPLICATIVE: a quaternion has four numbers but only three degrees of
freedom, so a plain additive Kalman filter on the four components has a
singular covariance and breaks down. The MEKF instead keeps the full attitude
as a reference quaternion and runs the filter on a 3-parameter SMALL-ANGLE
ERROR about it, folding that error back into the reference after each update.
The covariance stays well conditioned and the quaternion stays normalised.

The payoff demo: feed a smooth slew, black out ten frames entirely, and watch
the filter coast through the gap on its angular-rate estimate and reacquire.
That is the moment to put in your video.
"""

from __future__ import annotations

import numpy as np

from render import (angular_error_deg, quat_multiply, quat_normalize,
                    quat_to_matrix)


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def small_angle_quat(theta: np.ndarray) -> np.ndarray:
    """3-vector rotation error -> quaternion. Exact, not the usual approximation."""
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = theta / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


class AttitudeEKF:
    """MEKF over attitude (quaternion) and constant angular-rate bias.

    State: reference quaternion q (4, carried outside the covariance)
           angular rate omega (3, rad/s)
    Covariance: 6x6 over [attitude error (3), rate error (3)]
    """

    def __init__(self, process_attitude=1e-6, process_rate=1e-4,
                 init_attitude_var=1.0, init_rate_var=1e-2):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.P = np.diag([init_attitude_var] * 3 + [init_rate_var] * 3)
        self.Q = np.diag([process_attitude] * 3 + [process_rate] * 3)
        self.initialised = False

    def reset(self, q: np.ndarray, attitude_var: float = 1e-4):
        """Hard-set the attitude, e.g. from a fresh lost-in-space solve."""
        self.q = quat_normalize(np.asarray(q, dtype=np.float64))
        self.omega = np.zeros(3)
        self.P = np.diag([attitude_var] * 3 + [1e-2] * 3)
        self.initialised = True

    # -- prediction --------------------------------------------------------

    def predict(self, dt: float):
        """Propagate forward by dt with no measurement. This is what lets the
        filter coast through dropped frames."""
        if not self.initialised:
            return self.q
        theta = self.omega * dt
        self.q = quat_normalize(quat_multiply(small_angle_quat(theta), self.q))

        # Error-state transition. With the error defined by
        #     q_true = dq(d_theta) (x) q_est,  and  omega_true = omega + d_omega
        # the continuous dynamics are
        #     d_theta_dot = -omega x d_theta + d_omega
        # so the rate coupling is PLUS dt. Getting this sign wrong makes the
        # filter correct the rate in the wrong direction and diverge within a
        # few seconds -- it is the classic MEKF bug.
        F = np.eye(6)
        F[0:3, 0:3] = np.eye(3) - skew(theta)
        F[0:3, 3:6] = np.eye(3) * dt
        self.P = F @ self.P @ F.T + self.Q * dt
        return self.q

    # -- correction --------------------------------------------------------

    def update(self, q_meas: np.ndarray, sigma_deg: float = 0.05):
        """Fold in a measured attitude whose 1-sigma error is sigma_deg.

        sigma_deg should come from the model's uncertainty head or the
        classical solver's residual -- NOT a hand-tuned constant. That is the
        link between the learned and classical halves of this project: the
        network's self-reported confidence becomes the filter's measurement
        covariance.
        """
        q_meas = quat_normalize(np.asarray(q_meas, dtype=np.float64))
        if not self.initialised:
            self.reset(q_meas, (np.deg2rad(sigma_deg)) ** 2)
            return self.q

        # Innovation as a small-angle vector: the rotation from estimate to
        # measurement, expressed in 3 parameters.
        dq = quat_multiply(q_meas, _quat_conj(self.q))
        if dq[0] < 0:
            dq = -dq                      # shortest path, double-cover again
        innovation = 2.0 * dq[1:4]

        R = np.eye(3) * (np.deg2rad(sigma_deg) ** 2)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        correction = K @ innovation

        self.q = quat_normalize(quat_multiply(small_angle_quat(correction[0:3]),
                                              self.q))
        self.omega = self.omega + correction[3:6]

        # Joseph form: stays positive-definite under floating point, where the
        # textbook (I - KH)P form can quietly go indefinite and diverge.
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return self.q

    @property
    def attitude_sigma_deg(self) -> float:
        """Current 1-sigma attitude uncertainty, in degrees."""
        return float(np.degrees(np.sqrt(np.trace(self.P[0:3, 0:3]) / 3.0)))

    @property
    def rate_deg_s(self) -> float:
        return float(np.degrees(np.linalg.norm(self.omega)))


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def run_sequence(measurements, truth, dt=0.1, sigmas=None, dropout=None):
    """Run the filter over a sequence and report per-frame error.

    measurements : list of quaternions or None (None = frame failed to solve)
    truth        : array of true quaternions
    sigmas       : per-frame 1-sigma in degrees (from the uncertainty head)
    dropout      : indices to force-blank, for the coasting demo

    Returns dict with filtered/raw error arrays and the filter's own sigma.
    """
    ekf = AttitudeEKF()
    dropout = set(dropout or [])
    filt_err, raw_err, pred_sigma, coasted = [], [], [], []

    for i, truth_q in enumerate(truth):
        ekf.predict(dt)
        m = None if i in dropout else measurements[i]
        if m is not None:
            s = 0.05 if sigmas is None else float(sigmas[i])
            ekf.update(m, sigma_deg=max(s, 1e-3))
            raw_err.append(angular_error_deg(truth_q, m))
            coasted.append(False)
        else:
            raw_err.append(np.nan)
            coasted.append(True)
        filt_err.append(angular_error_deg(truth_q, ekf.q))
        pred_sigma.append(ekf.attitude_sigma_deg)

    return {"filtered_deg": np.array(filt_err), "raw_deg": np.array(raw_err),
            "sigma_deg": np.array(pred_sigma), "coasted": np.array(coasted),
            "final_rate_deg_s": ekf.rate_deg_s}


# ----------------------------------------------------------------------------
# Self-test: the coasting demo
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    from render import StarCatalog, random_quaternion

    rng = np.random.default_rng(0)
    n, dt, rate = 60, 0.1, 3.0

    # Simulate a smooth slew and noisy per-frame attitude measurements.
    q = random_quaternion(rng)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    step = np.deg2rad(rate) * dt
    dq = np.concatenate([[np.cos(step / 2)], axis * np.sin(step / 2)])

    truth, meas = [], []
    noise_deg = 0.08
    for _ in range(n):
        truth.append(q.copy())
        perturb = small_angle_quat(rng.normal(0, np.deg2rad(noise_deg), 3))
        meas.append(quat_normalize(quat_multiply(perturb, q)))
        q = quat_normalize(quat_multiply(dq, q))
    truth = np.array(truth)

    print(f"simulated slew: {n} frames at {rate} deg/s, "
          f"per-frame measurement noise {noise_deg} deg\n")

    clean = run_sequence(meas, truth, dt)
    print(f"no dropouts:")
    print(f"  raw per-frame error   {np.nanmedian(clean['raw_deg']):.4f} deg")
    print(f"  EKF filtered error    {np.median(clean['filtered_deg'][10:]):.4f} deg")
    print(f"  noise reduction       "
          f"{np.nanmedian(clean['raw_deg'])/np.median(clean['filtered_deg'][10:]):.1f}x")
    print(f"  recovered rate        {clean['final_rate_deg_s']:.3f} deg/s "
          f"(true {rate})")

    gap = list(range(30, 40))
    out = run_sequence(meas, truth, dt, dropout=gap)
    during = out["filtered_deg"][gap]
    print(f"\nblack out frames {gap[0]}-{gap[-1]} (total sensor loss):")
    print(f"  error entering gap    {out['filtered_deg'][gap[0]-1]:.4f} deg")
    print(f"  error at end of gap   {during[-1]:.4f} deg  <-- coasting on "
          f"the rate estimate")
    print(f"  sigma grew            {out['sigma_deg'][gap[0]-1]:.4f} -> "
          f"{out['sigma_deg'][gap[-1]]:.4f} deg  <-- filter knows it is "
          f"getting less sure")
    print(f"  error 3 frames after  {out['filtered_deg'][gap[-1]+3]:.4f} deg  "
          f"<-- reacquired")
