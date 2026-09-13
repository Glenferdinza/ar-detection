import numpy as np

def compute_iou(boxA, boxB):
    xa = max(boxA[0], boxB[0])
    ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2])
    yb = min(boxA[3], boxB[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    boxA_area = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter_area / float(boxA_area + boxB_area - inter_area)

class ProductionARTracker:
    def __init__(self, ema_alpha=0.65, max_lost_frames=8, hysteresis_conf=0.25):
        self.ema_alpha = ema_alpha
        self.max_lost_frames = max_lost_frames
        self.hysteresis_conf = hysteresis_conf
        
        self.is_locked = False
        self.locked_cls = None
        self.locked_box = None
        self.locked_conf = 0.0
        self.lost_frame_count = 0
        self.total_locked_frames = 0

    def update(self, detected_boxes):
        if not detected_boxes:
            if self.is_locked:
                self.lost_frame_count += 1
                if self.lost_frame_count <= self.max_lost_frames:
                    # Coasting: preserve locked AR exhibit during momentary blur / hand shake
                    return [{
                        'cls': self.locked_cls,
                        'conf': self.locked_conf * 0.95,
                        'xyxy': [int(v) for v in self.locked_box],
                        'is_coasting': True
                    }]
                else:
                    self._reset()
            return []

        # Find best match with current lock or pick highest confidence candidate
        if self.is_locked:
            best_match = None
            best_iou = 0.0
            for b in detected_boxes:
                if b['cls'] == self.locked_cls:
                    iou = compute_iou(self.locked_box, b['xyxy'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = b

            if best_match is not None and (best_iou >= 0.20 or best_match['conf'] >= self.hysteresis_conf):
                # Smooth bounding box with Exponential Moving Average
                cur_box = np.array(best_match['xyxy'], dtype=np.float32)
                self.locked_box = (self.ema_alpha * cur_box) + ((1.0 - self.ema_alpha) * np.array(self.locked_box, dtype=np.float32))
                self.locked_conf = (self.ema_alpha * best_match['conf']) + ((1.0 - self.ema_alpha) * self.locked_conf)
                self.lost_frame_count = 0
                self.total_locked_frames += 1

                return [{
                    'cls': self.locked_cls,
                    'conf': float(self.locked_conf),
                    'xyxy': [int(round(v)) for v in self.locked_box],
                    'is_coasting': False
                }]
            else:
                self.lost_frame_count += 1
                if self.lost_frame_count <= self.max_lost_frames:
                    return [{
                        'cls': self.locked_cls,
                        'conf': self.locked_conf * 0.90,
                        'xyxy': [int(v) for v in self.locked_box],
                        'is_coasting': True
                    }]
                else:
                    self._reset()

        # Acquisition: lock onto highest confidence candidate
        top_cand = sorted(detected_boxes, key=lambda b: b['conf'], reverse=True)[0]
        self.is_locked = True
        self.locked_cls = top_cand['cls']
        self.locked_box = [float(v) for v in top_cand['xyxy']]
        self.locked_conf = top_cand['conf']
        self.lost_frame_count = 0
        self.total_locked_frames = 1

        return [{
            'cls': self.locked_cls,
            'conf': float(self.locked_conf),
            'xyxy': [int(round(v)) for v in self.locked_box],
            'is_coasting': False
        }]

    def _reset(self):
        self.is_locked = False
        self.locked_cls = None
        self.locked_box = None
        self.locked_conf = 0.0
        self.lost_frame_count = 0
        self.total_locked_frames = 0
