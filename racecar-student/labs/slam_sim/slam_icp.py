import numpy as np
import math
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0 

    def to_array(self):
        return np.array([self.x, self.y, self.theta])


class ICPScanMatcher:
    ICP_MAX_ITERS = 25
    ICP_TOLERANCE = 1e-5
    ICP_MAX_CORR_DIST = 0.5
    ICP_MIN_MATCHES = 10
    ICP_MAX_TRANSLATION = 1.0 
    ICP_MAX_ROTATION = 1.0
    def __init__(self):
        self.last_points = None
        self.last_transform = None
        self.state = Pose(0.0, 0.0, 0.0)

    # ---------- static/pure helpers ----------

    @staticmethod
    def scan_to_points(scan_data, angle_min=-math.pi, angle_increment=None,
                       range_min=0.05, range_max=12): #scna to points, self explantory
        
        num_samples = len(scan_data) #num sample

        if num_samples == 0:
            return np.empty((0, 2)) #empty

        if angle_increment is None:
            angle_increment = (2.0 * math.pi) / num_samples #calc angle inc if not given

        r = np.asarray(scan_data, dtype=float)
        ang = angle_min + np.arange(num_samples) * angle_increment
        ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        return np.column_stack((r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok]))) #cleaner decomp

    @staticmethod
    def get_nearest_neighbors(source, target): #get neighbor
        tree = cKDTree(target) #tree of target
        dists, idx = tree.query(source, k=1) #query, get 
        return dists, idx

    @staticmethod
    def best_fit_transform(A, B): # ls trans, bf
        centroid_A = np.mean(A, axis=0) #centroid of smpa
        centroid_B = np.mean(B, axis=0) #centroid of smp b

        AA = A - centroid_A #delta_position for each in A
        BB = B - centroid_B

        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0: #if
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        return R, t

    @staticmethod
    def _normalize_angle(angle): #wrap angle
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def icp(self, source, target, init_R=None, init_t=None): #one iter, read viir's comments, i just refomrateed the code, but logic is still the same
        R_total = np.eye(2) if init_R is None else init_R.copy()
        t_total = np.zeros(2) if init_t is None else init_t.copy()

        src = (R_total @ source.T).T + t_total

        prev_error = None
        mean_error = None
        iters = 0

        for iters in range(1, self.ICP_MAX_ITERS + 1):
            dists, idx = self.get_nearest_neighbors(src, target)
            inliers = dists <= self.ICP_MAX_CORR_DIST

            if np.count_nonzero(inliers) <= self.ICP_MIN_MATCHES:
                mean_error = None
                break

            src_match = src[inliers]
            tgt_match = target[idx[inliers]]

            R_step, t_step = self.best_fit_transform(src_match, tgt_match)

            src = (R_step @ src.T).T + t_step

            R_total = R_step @ R_total
            t_total = R_step @ t_total + t_step

            mean_error = np.mean(dists[inliers])

            if prev_error is not None and abs(prev_error - mean_error) < self.ICP_TOLERANCE:
                break

            prev_error = mean_error

        return R_total, t_total, mean_error, iters

    def update(self, cur_points):
        cur_points = np.asarray(cur_points, dtype=float)

        if (self.last_points is None
                or len(cur_points) <= self.ICP_MIN_MATCHES
                or len(self.last_points) <= self.ICP_MIN_MATCHES):
            if len(cur_points) > self.ICP_MIN_MATCHES:
                self.last_points = cur_points
            self.last_transform = None
            return self.state #invalid, just return early

        init_R = self.last_transform[0] if self.last_transform is not None else None
        init_t = self.last_transform[1] if self.last_transform is not None else None

        R, t, mean_error, iters = self.icp(
            cur_points,
            self.last_points,
            init_R,
            init_t,
        ) #get rot and transpose from icp

        dtheta = math.atan2(R[1, 0], R[0, 0]) #convert from translation in points to translation in bot pose to absolute pose ()

        if (mean_error is None
                or np.hypot(t[0], t[1]) > self.ICP_MAX_TRANSLATION
                or abs(dtheta) > self.ICP_MAX_ROTATION):
            self.last_points = cur_points
            self.last_transform = None
            print("ICP match failed or implausible; pose not updated.")
            return self.state

        self.last_points = cur_points #update points
        self.last_transform = (R, t) #update transform

        theta = self.state.theta
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        new_x = self.state.x + t[0] * cos_t - t[1] * sin_t
        new_y = self.state.y + t[0] * sin_t + t[1] * cos_t
        new_theta = self._normalize_angle(theta + dtheta)

        self.state = Pose(new_x, new_y, new_theta) #abs pose based on accumulated xy
        return self.state

    def get_pose(self): #getter
        return self.state