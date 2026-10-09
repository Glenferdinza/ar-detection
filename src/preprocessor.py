import cv2
import numpy as np

class ArtefactPreprocessor:
    def __init__(self, beta_passing_grade=25.0, target_size=(640, 640)):
        self.beta_passing_grade = beta_passing_grade
        self.target_size = target_size

    def assess_quality(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return False, {"reason": "empty"}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        contrast = float(gray.std())

        # Tenengrad edge gradient check
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad_score = float(np.mean(sobel_x**2 + sobel_y**2))

        # Check archival monochrome / sepia tone
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))
        is_archival_monochrome = bool(saturation < 28.0)

        # Dynamic tolerance for historical archive photos
        effective_beta_thresh = self.beta_passing_grade * 0.70 if is_archival_monochrome else self.beta_passing_grade
        beta_score = laplacian_var * (contrast / 50.0)

        is_sharp = (beta_score >= effective_beta_thresh) or (is_archival_monochrome and tenengrad_score >= 120.0)
        is_passed = (
            is_sharp and
            20.0 <= mean_brightness <= 248.0 and
            contrast >= (12.0 if is_archival_monochrome else 15.0)
        )

        details = {
            "beta_score": float(beta_score),
            "laplacian_var": laplacian_var,
            "tenengrad_score": float(tenengrad_score),
            "is_archival_monochrome": is_archival_monochrome,
            "mean_brightness": mean_brightness,
            "contrast": contrast,
            "is_passed": bool(is_passed)
        }
        return is_passed, details

    def normalize_color_and_lighting(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        # 1. LAB CLAHE for adaptive local contrast
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))

        # 2. Canonical archival color transfer (neutralize mobile screen cold purple/blue cast)
        lab_f = enhanced_lab.astype(np.float32)
        l_mean, l_std = float(lab_f[:, :, 0].mean()), max(8.0, float(lab_f[:, :, 0].std()))
        a_mean, a_std = float(lab_f[:, :, 1].mean()), max(2.0, float(lab_f[:, :, 1].std()))
        b_mean, b_std = float(lab_f[:, :, 2].mean()), max(2.0, float(lab_f[:, :, 2].std()))

        # Target archival portrait parameters
        target_l_mean, target_l_std = 120.0, 48.0
        target_a_mean, target_a_std = 126.5, 6.0
        target_b_mean, target_b_std = 137.0, 9.0

        lab_f[:, :, 0] = (lab_f[:, :, 0] - l_mean) * (target_l_std / l_std) + target_l_mean
        lab_f[:, :, 1] = (lab_f[:, :, 1] - a_mean) * (target_a_std / a_std) + target_a_mean
        lab_f[:, :, 2] = (lab_f[:, :, 2] - b_mean) * (target_b_std / b_std) + target_b_mean

        lab_norm = np.clip(lab_f, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

    def detect_screen_contour(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None

        h, w = image_bgr.shape[:2]
        total_area = h * w
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_rect = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * 0.04 or area > total_area * 0.92:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / float(ch + 1e-5)
            if 0.35 <= aspect <= 2.8:
                if area > max_area:
                    max_area = area
                    best_rect = (x, y, cw, ch)

        return best_rect

    def refine_inner_portrait(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        h, w = crop_bgr.shape[:2]
        if h < 45 or w < 45:
            return crop_bgr

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Detect vertical bezel (dark bar at top e.g. phone bezel / browser search bar)
        row_means = np.mean(gray, axis=1)
        if len(row_means) >= 20:
            smoothed_row = np.convolve(row_means, np.ones(7)/7, mode='valid')
            diff_row = np.diff(smoothed_row)
            limit_y = int(len(diff_row) * 0.45)
            if limit_y > 5:
                jump_y = int(np.argmax(diff_row[:limit_y])) + 3
                if np.mean(row_means[:jump_y]) < np.mean(row_means[jump_y:min(h, jump_y + 40)]) * 0.92:
                    crop_bgr = crop_bgr[jump_y:, :]
                    gray = gray[jump_y:, :]
                    h = crop_bgr.shape[0]

        # 2. Detect horizontal bezel/hand on the left
        col_means = np.mean(gray, axis=0)
        if len(col_means) >= 20:
            smoothed_col = np.convolve(col_means, np.ones(7)/7, mode='valid')
            diff_col = np.diff(smoothed_col)
            limit_x = int(len(diff_col) * 0.45)
            if limit_x > 5:
                jump_x = int(np.argmax(diff_col[:limit_x])) + 3
                if np.mean(col_means[:jump_x]) < np.mean(col_means[jump_x:min(w, jump_x + 40)]) * 0.92:
                    crop_bgr = crop_bgr[:, jump_x:]

        return crop_bgr

    def detect_orientation_hough(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=30, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            line_arr = np.array(line).reshape(-1)
            if len(line_arr) < 4:
                continue
            x1, y1, x2, y2 = int(line_arr[0]), int(line_arr[1]), int(line_arr[2]), int(line_arr[3])
            if x2 == x1:
                angle = 90.0
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if -45.0 <= angle <= 45.0:
                angles.append(angle)
            elif angle > 45.0:
                angles.append(angle - 90.0)
            elif angle < -45.0:
                angles.append(angle + 90.0)

        if len(angles) == 0:
            return 0.0

        median_angle = float(np.median(angles))
        return median_angle

    def deskew(self, image_bgr, angle):
        if abs(angle) < 0.5:
            return image_bgr

        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image_bgr, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return deskewed

    def standardize(self, image_bgr):
        h, w = image_bgr.shape[:2]
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        meta = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "orig_size": (w, h)
        }
        return padded, meta

    def process(self, image_bgr):
        is_passed, quality_info = self.assess_quality(image_bgr)
        if not is_passed:
            return False, None, quality_info

        normalized = self.normalize_color_and_lighting(image_bgr)
        angle = self.detect_orientation_hough(normalized)
        deskewed = self.deskew(normalized, angle)
        standardized, meta = self.standardize(deskewed)

        quality_info["detected_angle"] = angle
        quality_info["transform_meta"] = meta
        return True, standardized, quality_info
