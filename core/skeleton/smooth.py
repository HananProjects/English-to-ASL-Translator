from typing import Dict, Tuple

Vec2 = Tuple[float, float]


class EMASmoother:
    """
    Exponential Moving Average smoother for animation joints.
    Maintains per-joint state.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.prev: Dict[str, Vec2] = {}

    def update(self, joints: Dict[str, Vec2]) -> Dict[str, Vec2]:
        smoothed = {}

        for name, (x, y) in joints.items():
            if name in self.prev:
                px, py = self.prev[name]
                sx = self.alpha * x + (1 - self.alpha) * px
                sy = self.alpha * y + (1 - self.alpha) * py
            else:
                # first observation → no smoothing yet
                sx, sy = x, y

            smoothed[name] = (sx, sy)
            self.prev[name] = (sx, sy)

        # Remove joints that disappeared (e.g., hand left frame)
        missing = set(self.prev.keys()) - set(joints.keys())
        for k in missing:
            del self.prev[k]

        return smoothed
