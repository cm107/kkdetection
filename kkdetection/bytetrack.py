from typing import List
import numpy as np
import scipy
from cython_bbox import bbox_overlaps
import lap

# local package
from kkdetection.util.image import xyxy_to_xyah, xyah_to_xyxy
from kkdetection.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "KalmanFilter",
    "TrackState",
    "BaseTrack",
    "BYTETracker",
]


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919    
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """
    def __init__(self, std_position: float=1./20, std_velocity: float=1./160):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = std_position
        self._std_weight_velocity = std_velocity

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        assert isinstance(measurement, np.ndarray)
        assert len(measurement.shape) == 1 and measurement.shape[0] == 4
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean     = np.r_[mean_pos, mean_vel]
        std      = [
            2  * self._std_weight_position * measurement[3],
            2  * self._std_weight_position * measurement[3],
            1e-2,
            2  * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean       = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean           = np.dot(self._update_mat, mean)
        covariance     = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr        = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean       = np.dot(mean, self._motion_mat.T)
        left       = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation     = measurement - projected_mean
        new_mean       = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray,
        only_position: bool=False, metric: str='maha'
    ):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


class TrackState(object):
    New     = 0
    Tracked = 1
    Lost    = 2
    Removed = 3


class BaseTrack(object):
    _count   = 0
    track_id = 0
    state    = TrackState.New
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count
    def activate(self, *args):
        raise NotImplementedError
    def predict(self):
        raise NotImplementedError
    def update(self, *args, **kwargs):
        raise NotImplementedError
    def mark_lost(self):
        self.state = TrackState.Lost
    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    def __init__(self, width: int, height: int, kalman_filter: KalmanFilter=KalmanFilter()):
        """
        Params::
            bbox: center x, center y, aspect ratio (width / height), height
        """
        self.width, self.height = width, height
        self.kalman_filter = kalman_filter
        self.mean          = None 
        self.covariance    = None
        self.is_activated  = False
        self.score         = None
        self.tracklet_len  = 0
        self.is_check      = True
        self.end_frame     = 0
        self.start_frame   = 0

    def activate(self, bbox: np.ndarray, score: float, frame_id: int):
        if self.is_check:
            assert isinstance(bbox, np.ndarray)
            assert len(bbox.shape) == 1 and bbox.shape[0] == 4
            assert isinstance(score, float)
            self.is_check = False
        self.track_id              = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(bbox)
        self.tracklet_len          = 0
        self.state                 = TrackState.Tracked
        self.score                 = score
        if frame_id == 1:
            # Only the initial frame will be activated immediately.
            self.is_activated = True
        self.end_frame   = frame_id
        self.start_frame = frame_id

    def predict(self):
        """
        predict x(t + 1) and P(t + 1)
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def update(self, bbox: np.ndarray, score: float, frame_id: int, re_activate: bool=False, new_id: bool=False):
        """
        update x(t + 1) and P(t + 1) with observed bbox
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, bbox
        )
        if re_activate: self.tracklet_len  = 0
        else:           self.tracklet_len += 1
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.end_frame    = frame_id
        self.score        = score
        if new_id:
            self.track_id = self.next_id()
    
    def to_xyxy(self):
        bbox = xyah_to_xyxy(self.mean[:4].copy())
        if bbox[0] < 0: bbox[0] = 0
        if bbox[1] < 0: bbox[1] = 0
        if bbox[2] > self.height: bbox[2] = self.height
        if bbox[3] > self.width:  bbox[3] = self.width
        return bbox
    
    def to_xyah(self):
        return self.mean[:4].copy()

    def __repr__(self):
        return f'OT_{self.track_id}_{self.tracklet_len}({self.start_frame}-{self.end_frame})'


def multi_predict(kalman_filter: KalmanFilter, stracks: List[STrack]):
    if len(stracks) > 0:
        multi_mean       = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance  for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
        for i, (mean, covariance) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean       = mean
            stracks[i].covariance = covariance

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray):
    if bboxes_a.shape[0] == 0 or bboxes_b.shape[0] == 0:
        return np.zeros((bboxes_a.shape[0], bboxes_b.shape[0]))
    return 1 - bbox_overlaps(bboxes_a, bboxes_b)

def calc_tracks_vs_bboxes(tracks: List[STrack], bboxes: np.ndarray, scores: np.ndarray, frame_id: int, threshold: float):
    bboxes = bboxes.copy()
    dists  = ious_matrix(np.ascontiguousarray([track.to_xyxy() for track in tracks]), bboxes)
    matches, i_other_tracks, i_other_bboxes = linear_assignment(dists, thresh=threshold)
    act_tracks, ref_tracks, ref_bboxes, ref_scores, other_tracks, other_bboxes, other_scores = [], [], [], [], [], None, None
    for i_track, i_bbox in matches:
        track = tracks[i_track]
        if track.state == TrackState.Tracked:
            track.update(xyxy_to_xyah(bboxes[i_bbox]), scores[i_bbox], frame_id)
            act_tracks.append(track)
        else:
            ref_tracks.append(track)
            ref_bboxes.append(i_bbox)
            ref_scores.append(i_bbox)
    for i in i_other_tracks: other_tracks.append(tracks[i])
    i_other_bboxes = list(i_other_bboxes)
    other_bboxes   = bboxes[i_other_bboxes]
    other_scores   = scores[i_other_bboxes]
    ref_bboxes     = bboxes[ref_bboxes]
    ref_scores     = bboxes[ref_scores]
    return act_tracks, ref_tracks, ref_bboxes, ref_scores, other_tracks, other_bboxes, other_scores

class BYTETracker(object):
    def __init__(
        self, height: int, width: int, 
        thre_bbox_high: float=0.7, thre_bbox_low: float=0.2, 
        thre_iou_high: float=0.8, thre_iou_low: float=0.5, thre_iou_new: float=0.7, 
        max_time_lost=100
    ):
        """
        Parmas::
            height, width: video size.
            thre_iou_high:
                high score bbox VS pool tracks, IoU threshold
                if thre_iou_high=0.8, IoU < 0.2 is ignored in "lap".
        """
        self.height, self.width = height, width
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks    = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id        = 0
        self.thre_bbox_high  = thre_bbox_high
        self.thre_bbox_low   = thre_bbox_low
        self.thre_iou_high   = thre_iou_high
        self.thre_iou_low    = thre_iou_low
        self.thre_iou_new    = thre_iou_new
        self.max_time_lost   = max_time_lost
        self.kalman_filter   = KalmanFilter()
    
    def init_trackid(self):
        BaseTrack._count = 0

    def update(self, bboxes: np.ndarray, scores: np.ndarray):
        """
        Params::
            bboxes: numpy array (x1, y1, x2, y2)
        """
        self.frame_id += 1
        activated_starcks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []
        inds_high     = scores >= self.thre_bbox_high
        inds_low      = (scores >= self.thre_bbox_low) & (scores < self.thre_bbox_high)
        bboxes_high   = bboxes[inds_high]
        scores_high   = scores[inds_high]
        bboxes_low    = bboxes[inds_low]
        scores_low    = scores[inds_low]
        ''' Add newly detected tracklets to tracked_stracks '''
        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        multi_predict(self.kalman_filter, strack_pool)
        ''' Step 2: Active & Lost tracks VS High Bboxes '''
        act_tracks, ref_tracks, ref_bboxes, ref_scores, other_tracks, new_bboxes, new_scores = calc_tracks_vs_bboxes(
            strack_pool, bboxes_high, scores_high, self.frame_id, self.thre_iou_high
        )
        activated_starcks = activated_starcks + act_tracks
        for i, track in enumerate(ref_tracks):
            track.update(xyxy_to_xyah(ref_bboxes[i]), ref_scores[i], self.frame_id, re_activate=True, new_id=False)
            refind_stracks.append(track)
        ''' Step 3: Other (Active & Lost tracks) VS Low Bboxes '''
        act_tracks, ref_tracks, ref_bboxes, ref_scores, other_tracks, _, _ = calc_tracks_vs_bboxes(
            other_tracks, bboxes_low, scores_low, self.frame_id, self.thre_iou_low
        )
        activated_starcks = activated_starcks + act_tracks
        for i, track in enumerate(ref_tracks):
            track.update(xyxy_to_xyah(ref_bboxes[i]), ref_scores[i], self.frame_id, re_activate=True, new_id=False)
            refind_stracks.append(track)
        for i, track in enumerate(other_tracks):
            track.mark_lost()
            lost_stracks.append(track)
        ''' Step 4: Non Active tracks VS New Bboxes '''
        act_tracks, _, _, _, other_tracks, new_bboxes, new_scores = calc_tracks_vs_bboxes(
            unconfirmed, new_bboxes, new_scores, self.frame_id, self.thre_iou_new
        )
        activated_starcks = activated_starcks + act_tracks
        for i, track in enumerate(other_tracks):
            track.mark_removed()
            removed_stracks.append(track)
        for i, bbox in enumerate(new_bboxes):
            track = STrack(self.height, self.width, kalman_filter=self.kalman_filter)
            track.activate(xyxy_to_xyah(bbox), new_scores[i], self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        return [track for track in self.tracked_stracks if track.is_activated]

    def tracking(self, list_bboxes: List[np.ndarray], list_scores: List[np.ndarray], ret_type: str="xyxy") -> List[dict]:
        assert isinstance(list_bboxes, list)
        assert isinstance(list_scores, list)
        assert len(list_bboxes) == len(list_scores)
        assert isinstance(ret_type, str) and ret_type in ["xyxy", "xyah"]
        self.init_trackid()
        list_ouptuts = []
        for bboxes, scores in zip(list_bboxes, list_scores):
            tracks      = self.update(bboxes, scores)
            dict_output = {track.track_id: getattr(track, f"to_{ret_type}")() for track in tracks}
            list_ouptuts.append(dict_output)
        return list_ouptuts


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack]):
    pdist = ious_matrix(
        np.ascontiguousarray([track.to_xyxy() for track in stracksa]),
        np.ascontiguousarray([track.to_xyxy() for track in stracksb]), 
    )
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].end_frame - stracksa[p].start_frame
        timeq = stracksb[q].end_frame - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
