"""Motion-compensated history for estimated removal pixels, before robots.

This is a conservative temporal filter, not learned video inpainting. Visible
cloth is never filtered. Camera registration failures and scene cuts reset the
history instead of carrying a previous scene into the next demonstration.
"""
import cv2
import numpy as np


class TemporalRemoval:
    def __init__(self):
        self.previous = None
        self.xy = np.stack(np.meshgrid(np.arange(320), np.arange(240)), -1).astype(np.float32)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    @staticmethod
    def registration(previous, raw, exclude):
        old_gray = cv2.cvtColor(previous['raw'], cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        safe = (~previous['exclude']).astype(np.uint8) * 255
        pts = cv2.goodFeaturesToTrack(old_gray, 600, .012, 9, mask=safe)
        if pts is None or len(pts) < 16:
            return None, dict(reason='few_background_features')
        moved, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, moved, None, winSize=(21, 21), maxLevel=3)
        src, dst = pts[:, 0], moved[:, 0]
        x = np.clip(np.rint(dst[:, 0]).astype(int), 0, 639)
        y = np.clip(np.rint(dst[:, 1]).astype(int), 0, 479)
        ok = (status[:, 0] > 0) & (reverse[:, 0] > 0) & (np.linalg.norm(back[:, 0]-src, axis=1) < 1.2)
        ok &= ~exclude[y, x] & (dst[:, 0] >= 0) & (dst[:, 0] < 640) & (dst[:, 1] >= 0) & (dst[:, 1] < 480)
        if ok.sum() < 12:
            return None, dict(reason='few_consistent_tracks', tracks=int(ok.sum()))
        # A low-dimensional consecutive-frame camera model avoids a noisy
        # independent donor homography at each time step.
        affine, inliers = cv2.estimateAffinePartial2D(src[ok], dst[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if affine is None or inliers.sum() < 10:
            return None, dict(reason='camera_fit_failed')
        H = np.eye(3); H[:2] = affine
        predicted = src[ok] @ affine[:, :2].T + affine[:, 2]
        residual = float(np.median(np.linalg.norm(predicted-dst[ok], axis=1)[inliers[:, 0] > 0]))
        warped = cv2.warpPerspective(previous['raw'], H, (640, 480))
        valid = cv2.warpPerspective((~previous['exclude']).astype(np.uint8), H, (640, 480), flags=cv2.INTER_NEAREST) > 0
        observed = valid & ~exclude
        error = float(np.median(np.abs(warped.astype(np.float32)-raw)[observed])) if observed.sum() > 500 else 999.
        scale = float(np.linalg.det(affine[:, :2]))
        info = dict(tracks=int(ok.sum()), inliers=int(inliers.sum()), registration_error_px=residual, observed_rgb_error=error)
        if residual > 1.2 or error > 22 or not .88 < scale < 1.13:
            return None, dict(info, reason='scene_cut_or_bad_registration')
        return H, dict(info, reason='registered')

    def apply(self, raw, clean, result, cloth, human, holes, index):
        # The history plate includes an estimate under the garment, so a moving
        # hand does not select a different source texture every frame.
        exclude = cv2.dilate((cloth | human).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        plate = clean.copy()
        background = human & ~holes & ~cloth
        plate[background] = result[background]
        record = dict(frame_index=index, history_reset=True, background_filtered_pixels=0, cloth_filtered_pixels=0)
        old = self.previous
        baseline = result.copy()
        H = None
        if old is not None and index == old['index'] + 1:
            H, reg = self.registration(old, raw, exclude)
            record.update(reg)
        if H is not None:
            prior = cv2.warpPerspective(old['plate'], H, (640, 480)).astype(np.float32)
            valid = cv2.warpPerspective(np.ones((480, 640), np.uint8), H, (640, 480), flags=cv2.INTER_NEAREST) > 0
            observed = valid & ~exclude
            if observed.sum() > 500:
                # Correct camera exposure from visible surroundings, not from
                # moving skin or the estimated patch itself.
                delta = np.clip(np.median(raw[observed].astype(float)-prior[observed], axis=0), -6, 6)
                prior += delta
            estimated = (cloth | human) & valid
            # A camera affine is approximate near the tabletop/floor depth
            # discontinuity. Reject history that brings a dark floor edge into
            # a brown table estimate, with a soft photometric confidence gate.
            mismatch = np.max(np.abs(prior-plate.astype(np.float32)), axis=2)
            history_weight = .90*np.clip((55-mismatch)/25, 0, 1)
            current_hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
            prior_hsv = cv2.cvtColor(np.clip(prior, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            current_table = (current_hsv[:, :, 0] >= 8) & (current_hsv[:, :, 0] <= 38) & (current_hsv[:, :, 1] > 55)
            prior_table = (prior_hsv[:, :, 0] >= 8) & (prior_hsv[:, :, 0] <= 38) & (prior_hsv[:, :, 1] > 40)
            history_weight[current_table & ~prior_table] = 0
            w = history_weight[estimated, None]
            plate[estimated] = np.rint(np.clip(w*prior[estimated] + (1-w)*plate[estimated], 0, 255)).astype(np.uint8)
            result[background & valid] = plate[background & valid]
            record.update(history_reset=False, background_filtered_pixels=int((background & valid).sum()))
            # Cloth follows nonrigid optical flow, never the background camera
            # transform. Only nearby, bidirectionally reliable visible-cloth
            # tracks may propagate an earlier inferred cloth patch.
            gray = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (320, 240))
            prior_gray = cv2.resize(cv2.cvtColor(old['raw'], cv2.COLOR_BGR2GRAY), (320, 240))
            backflow = self.flow.calc(gray, prior_gray, None)
            forward = self.flow.calc(prior_gray, gray, None)
            maps = self.xy + backflow
            forward_at = cv2.remap(forward, maps[:, :, 0], maps[:, :, 1], cv2.INTER_LINEAR)
            consistent = np.linalg.norm(backflow + forward_at, axis=2) < .8
            old_visible = cv2.resize((old['cloth'] & ~old['human']).astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST)
            old_visible = cv2.remap(old_visible, maps[:, :, 0], maps[:, :, 1], cv2.INTER_NEAREST) > 0
            safe = cv2.resize((cloth & ~human).astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST).astype(bool) & old_visible & consistent
            if safe.sum() > 100 and holes.any():
                # Distance labels extrapolate only 16px from observed cloth;
                # larger unobserved regions deliberately remain unverified.
                dist, labels = cv2.distanceTransformWithLabels((~safe).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                lookup = np.zeros((int(labels.max())+1, 2), np.float32)
                lookup[labels[safe]] = backflow[safe]
                propagated = cv2.resize(lookup[labels], (640, 480)) * 2
                yy, xx = np.mgrid[:480, :640].astype(np.float32)
                mx, my = xx + propagated[:, :, 0], yy + propagated[:, :, 1]
                previous_fill = cv2.remap(old['result'], mx, my, cv2.INTER_LINEAR)
                previous_cloth = cv2.remap((old['cloth'] | old['holes']).astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
                take = holes & previous_cloth & (cv2.resize(dist, (640, 480)) < 8)
                result[take] = np.rint(np.clip(.65*previous_fill[take].astype(float)+.35*result[take], 0, 255)).astype(np.uint8)
                record['cloth_filtered_pixels'] = int(take.sum())
        # A soft outside edge suppresses binary-mask boundary flicker. Actual
        # observed garment pixels remain exact; no robot pixels enter history.
        alpha = cv2.GaussianBlur(human.astype(np.float32), (15, 15), 2.5)
        alpha[human] = 1
        alpha[cloth & ~human] = 0
        alpha[alpha < .01] = 0
        fringe = (alpha > 0) & ~human
        result[fringe] = np.clip(plate[fringe].astype(float)*alpha[fringe, None]+raw[fringe]*(1-alpha[fringe, None]), 0, 255).astype(np.uint8)
        if H is not None:
            old_background = cv2.warpPerspective(old['background'].astype(np.uint8), H, (640, 480), flags=cv2.INTER_NEAREST) > 0
            compare = cv2.erode((background & old_background).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if compare.sum() > 100:
                before = cv2.warpPerspective(old['baseline'], H, (640, 480))
                after = cv2.warpPerspective(old['result'], H, (640, 480))
                record['background_temporal_pixels'] = int(compare.sum())
                record['background_rgb_mae_before'] = float(np.abs(baseline.astype(float)-before)[compare].mean())
                record['background_rgb_mae_after'] = float(np.abs(result.astype(float)-after)[compare].mean())
        record['observed_cloth_unchanged'] = bool(np.array_equal(result[cloth & ~human], raw[cloth & ~human]))
        self.previous = dict(index=index, raw=raw.copy(), exclude=exclude, plate=plate, result=result.copy(), baseline=baseline, background=background, cloth=cloth.copy(), human=human.copy(), holes=holes.copy())
        record['fringe_pixels'] = int(fringe.sum())
        return result, human | fringe, record, H
