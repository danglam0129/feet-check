import cv2


class ClassifierRenderer:
    @staticmethod
    def draw_skeleton(img, landmarks, connections, w, h, vis_threshold=0.6):
        for a, b in connections:
            la = landmarks[a]
            lb = landmarks[b]
            va = getattr(la, "visibility", getattr(la, "presence", 1.0))
            vb = getattr(lb, "visibility", getattr(lb, "presence", 1.0))
            if va < vis_threshold or vb < vis_threshold:
                continue
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            cv2.line(img, (ax, ay), (bx, by), (0, 200, 255), 2)

    @staticmethod
    def draw_floor_line(img, y, color=(255, 0, 0), thickness=2):
        h, w = img.shape[:2]
        cv2.line(img, (0, int(y)), (w, int(y)), color, thickness)

    @staticmethod
    def draw_label(img, text):
        cv2.putText(
            img,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
