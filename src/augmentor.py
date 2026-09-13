import cv2
import numpy as np
import random

class MuseumAugmentor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def apply_clahe(self, image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_color_jitter(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Hue shift
        hue_shift = random.uniform(-10.0, 10.0)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        # Saturation shift
        sat_factor = random.uniform(0.65, 1.35)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        # Value (brightness) shift
        val_factor = random.uniform(0.70, 1.30)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)

        jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return jittered

    def apply_random_grayscale(self, image_bgr, p=0.25):
        if random.random() < p:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def apply_gamma(self, image_bgr, gamma=None):
        if gamma is None:
            gamma = random.uniform(0.65, 1.35)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image_bgr, table)

    def add_museum_glare(self, image_bgr):
        h, w = image_bgr.shape[:2]
        center_x = random.randint(int(w * 0.2), int(w * 0.8))
        center_y = random.randint(int(h * 0.2), int(h * 0.8))
        radius = random.randint(int(min(h, w) * 0.12), int(min(h, w) * 0.30))

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (61, 61), 0)

        glare_color = np.full((h, w, 3), 255, dtype=np.uint8)
        alpha = random.uniform(0.18, 0.40)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        blended = (1 - alpha * mask_3ch) * image_bgr + (alpha * mask_3ch) * glare_color
        return np.clip(blended, 0, 255).astype(np.uint8)

    def add_sensor_noise(self, image_bgr):
        row, col, ch = image_bgr.shape
        sigma = random.uniform(6.0, 18.0)
        gauss = np.random.normal(0, sigma, (row, col, ch))
        noisy = image_bgr.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_sharpening(self, image_bgr):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image_bgr, -1, kernel)

    def add_motion_blur(self, image_bgr):
        size = random.choice([3, 5])
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(image_bgr, -1, kernel)

    def apply_cutout(self, image_bgr, box, p=0.30):
        if random.random() >= p:
            return image_bgr
        h, w = image_bgr.shape[:2]
        cut_w = random.randint(int(w * 0.08), int(w * 0.22))
        cut_h = random.randint(int(h * 0.08), int(h * 0.22))
        cx = random.randint(0, w - cut_w)
        cy = random.randint(0, h - cut_h)

        result = image_bgr.copy()
        fill_val = random.choice([0, 114, 255])
        result[cy:cy + cut_h, cx:cx + cut_w] = fill_val
        return result

    def apply_horizontal_flip(self, image_bgr, box):
        flipped = cv2.flip(image_bgr, 1)
        if box is None:
            return flipped, None
        bx, by, bw, bh = box
        new_bx = 1.0 - bx
        return flipped, (new_bx, by, bw, bh)

    def apply_affine_shearing(self, image_bgr, box, shear_factor=0.10):
        h, w = image_bgr.shape[:2]
        M = np.array([
            [1.0, shear_factor, 0],
            [0.0, 1.0, 0]
        ], dtype=np.float32)
        sheared = cv2.warpAffine(image_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if box is None:
            return sheared, None
        bx, by, bw, bh = box
        new_bx = min(0.98, max(0.02, bx + shear_factor * (by - 0.5)))
        return sheared, (new_bx, by, bw, bh)

    def rotate_with_box(self, image_bgr, box, angle):
        h, w = image_bgr.shape[:2]
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(image_bgr, rot_matrix, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

        if box is None:
            return rotated_img, None

        bx, by, bw, bh = box
        x1 = (bx - bw / 2) * w
        y1 = (by - bh / 2) * h
        x2 = (bx + bw / 2) * w
        y2 = (by + bh / 2) * h

        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ])

        new_corners = rot_matrix.dot(corners.T).T
        nx1 = max(0, min(new_w, np.min(new_corners[:, 0])))
        ny1 = max(0, min(new_h, np.min(new_corners[:, 1])))
        nx2 = max(0, min(new_w, np.max(new_corners[:, 0])))
        ny2 = max(0, min(new_h, np.max(new_corners[:, 1])))

        nbw = (nx2 - nx1) / new_w
        nbh = (ny2 - ny1) / new_h
        nbx = (nx1 + nx2) / (2.0 * new_w)
        nby = (ny1 + ny2) / (2.0 * new_h)

        return rotated_img, (nbx, nby, nbw, nbh)

    def generate_handheld_variant(self, hero_img, box=None):
        cw, ch = self.target_size

        # 1. Realistic Indoor Background (wall gradient or neutral room tone)
        base_col = np.array([random.randint(120, 220), random.randint(120, 220), random.randint(120, 220)], dtype=np.float32)
        x_grad = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), cw)
        y_grad = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), ch)
        grad_2d = np.outer(y_grad, x_grad)[:, :, np.newaxis]
        canvas = np.clip(base_col * grad_2d + np.random.normal(0, 4, (ch, cw, 3)), 0, 255).astype(np.uint8)

        # 2. Handheld Phone Dimensions
        pw = random.randint(180, 290)
        aspect = random.uniform(1.85, 2.15)
        ph = int(pw * aspect)
        ph = min(ph, ch - 60)

        # Create phone surface
        phone_surf = np.full((ph, pw, 3), random.randint(18, 40), dtype=np.uint8)
        bezel = random.randint(5, 9)
        sw = pw - 2 * bezel
        sh = ph - 2 * bezel
        screen_bg_val = random.choice([20, 30, 235, 245])
        screen = np.full((sh, sw, 3), screen_bg_val, dtype=np.uint8)

        # Top status bar
        status_h = random.randint(22, 38)
        status_val = int((screen_bg_val - 15) if screen_bg_val > 100 else (screen_bg_val + 15))
        cv2.rectangle(screen, (0, 0), (sw, status_h), (status_val, status_val, status_val), -1)

        # Crop hero based on box if provided
        if box is not None:
            bx, by, bw, bh = box
            ih, iw = hero_img.shape[:2]
            x1 = max(0, int((bx - bw / 2.0) * iw))
            y1 = max(0, int((by - bh / 2.0) * ih))
            x2 = min(iw, int((bx + bw / 2.0) * iw))
            y2 = min(ih, int((by + bh / 2.0) * ih))
            if x2 > x1 + 10 and y2 > y1 + 10:
                cropped_hero = hero_img[y1:y2, x1:x2]
            else:
                cropped_hero = hero_img
        else:
            cropped_hero = hero_img

        hero_fit_w = int(sw * random.uniform(0.65, 0.95))
        h_orig, w_orig = cropped_hero.shape[:2]
        hero_fit_h = int(hero_fit_w * (h_orig / max(1, w_orig)))
        max_h = int(sh * 0.68)
        if hero_fit_h > max_h:
            hero_fit_h = max_h
            hero_fit_w = int(hero_fit_h * (w_orig / max(1, h_orig)))

        resized_hero = cv2.resize(cropped_hero, (hero_fit_w, hero_fit_h), interpolation=cv2.INTER_AREA)

        # Chromatic screen tint (OLED cool purple/blue cast)
        if random.random() > 0.3:
            tint_type = random.choice(["purple", "blue", "cool", "neutral"])
            hero_f = resized_hero.astype(np.float32)
            if tint_type == "purple":
                hero_f[:, :, 0] *= random.uniform(1.05, 1.25)
                hero_f[:, :, 1] *= random.uniform(0.85, 0.95)
                hero_f[:, :, 2] *= random.uniform(1.05, 1.20)
            elif tint_type == "blue":
                hero_f[:, :, 0] *= random.uniform(1.10, 1.30)
                hero_f[:, :, 1] *= random.uniform(0.90, 1.00)
                hero_f[:, :, 2] *= random.uniform(0.80, 0.90)
            resized_hero = np.clip(hero_f, 0, 255).astype(np.uint8)

        hx = (sw - hero_fit_w) // 2
        hy = status_h + random.randint(10, max(11, int(sh * 0.12)))

        screen[hy:hy + hero_fit_h, hx:hx + hero_fit_w] = resized_hero

        # Simulate UI text below photo
        text_y = hy + hero_fit_h + 12
        for _ in range(random.randint(2, 5)):
            if text_y + 8 < sh:
                line_w = random.randint(int(sw * 0.3), int(sw * 0.85))
                line_val = 100 if screen_bg_val > 100 else 160
                cv2.rectangle(screen, (15, text_y), (15 + line_w, text_y + 6), (line_val, line_val, line_val), -1)
                text_y += 12

        # Screen glare reflection
        if random.random() > 0.4:
            glare_mask = np.zeros((sh, sw), dtype=np.float32)
            gx = random.randint(0, sw)
            cv2.line(glare_mask, (gx - 50, 0), (gx + 80, sh), 1.0, thickness=random.randint(25, 50))
            glare_mask = cv2.GaussianBlur(glare_mask, (41, 41), 0)
            alpha = random.uniform(0.12, 0.32)
            screen = np.clip((1.0 - alpha * glare_mask[:, :, np.newaxis]) * screen + (alpha * 255 * glare_mask[:, :, np.newaxis]), 0, 255).astype(np.uint8)

        phone_surf[bezel:bezel + sh, bezel:bezel + sw] = screen

        pb_x1 = bezel + hx
        pb_y1 = bezel + hy
        pb_x2 = pb_x1 + hero_fit_w
        pb_y2 = pb_y1 + hero_fit_h

        px = random.randint(20, max(21, cw - pw - 20))
        py = random.randint(20, max(21, ch - ph - 20))

        canvas[py:py + ph, px:px + pw] = phone_surf

        bx1 = px + pb_x1
        by1 = py + pb_y1
        bx2 = px + pb_x2
        by2 = py + pb_y2

        angle = random.uniform(-14.0, 14.0)
        rot_matrix = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
        rotated_canvas = cv2.warpAffine(canvas, rot_matrix, (cw, ch), borderMode=cv2.BORDER_REFLECT)

        corners = np.array([
            [bx1, by1, 1],
            [bx2, by1, 1],
            [bx2, by2, 1],
            [bx1, by2, 1]
        ])
        new_corners = rot_matrix.dot(corners.T).T
        nx1 = max(0, min(cw, np.min(new_corners[:, 0])))
        ny1 = max(0, min(ch, np.min(new_corners[:, 1])))
        nx2 = max(0, min(cw, np.max(new_corners[:, 0])))
        ny2 = max(0, min(ch, np.max(new_corners[:, 1])))

        nbw = (nx2 - nx1) / cw
        nbh = (ny2 - ny1) / ch
        nbx = (nx1 + nx2) / (2.0 * cw)
        nby = (ny1 + ny2) / (2.0 * ch)

        return rotated_canvas, (nbx, nby, nbw, nbh)

    def generate_variants(self, image_bgr, box=None, num_variants=16):
        variants = []
        # Half classical portrait variants, half handheld phone variants
        num_phone = num_variants // 2
        num_classic = num_variants - num_phone

        for _ in range(num_classic):
            aug_img = image_bgr.copy()
            aug_box = box

            # 1. Geometric: Multi-angle rotation (-20 to +20 deg)
            angle = random.uniform(-20.0, 20.0)
            aug_img, aug_box = self.rotate_with_box(aug_img, aug_box, angle)

            # 2. Geometric: Horizontal flipping (p=0.5) - NO vertical flipping!
            if random.random() > 0.5:
                aug_img, aug_box = self.apply_horizontal_flip(aug_img, aug_box)

            # 3. Geometric: Shearing / oblique perspective
            if random.random() > 0.6:
                shear = random.uniform(-0.12, 0.12)
                aug_img, aug_box = self.apply_affine_shearing(aug_img, aug_box, shear)

            # 4. Pixel: Color Jittering
            if random.random() > 0.3:
                aug_img = self.apply_color_jitter(aug_img)

            # 5. Pixel: Random Grayscale
            aug_img = self.apply_random_grayscale(aug_img, p=0.25)

            # 6. Pixel: Gamma correction
            if random.random() > 0.5:
                aug_img = self.apply_gamma(aug_img)

            # 7. Pixel: CLAHE local contrast
            if random.random() > 0.5:
                aug_img = self.apply_clahe(aug_img)

            # 8. Museum Artefact: Frame glass glare reflection
            if random.random() > 0.6:
                aug_img = self.add_museum_glare(aug_img)

            # 9. Pixel: Camera sensor noise or blur/sharpening
            dice = random.random()
            if dice < 0.3:
                aug_img = self.add_sensor_noise(aug_img)
            elif dice < 0.5:
                aug_img = self.add_motion_blur(aug_img)
            elif dice < 0.7:
                aug_img = self.apply_sharpening(aug_img)

            # 10. Modern Regularizer: Cutout / Random Erasing
            aug_img = self.apply_cutout(aug_img, aug_box, p=0.25)

            variants.append((aug_img, aug_box))

        for _ in range(num_phone):
            p_img, p_box = self.generate_handheld_variant(image_bgr, box=box)
            variants.append((p_img, p_box))

        return variants

