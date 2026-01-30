"""
Very simple 2D skeleton poses.
Coordinates are normalized (0.0–1.0) relative to widget size.
"""

POSES = {
    "REST": {
        "head": (0.5, 0.2),
        "shoulder": (0.5, 0.35),
        "elbow_left": (0.45, 0.5),
        "hand_left": (0.42, 0.65),
        "elbow_right": (0.55, 0.5),
        "hand_right": (0.58, 0.65),
        "torso": (0.5, 0.6),
    },

    "YOU": {
        "head": (0.5, 0.2),
        "shoulder": (0.5, 0.35),
        "elbow_right": (0.6, 0.45),
        "hand_right": (0.7, 0.4),
        "elbow_left": (0.45, 0.5),
        "hand_left": (0.42, 0.65),
        "torso": (0.5, 0.6),
    },

    "GO": {
        "head": (0.5, 0.2),
        "shoulder": (0.5, 0.35),
        "elbow_right": (0.65, 0.4),
        "hand_right": (0.75, 0.35),
        "elbow_left": (0.45, 0.5),
        "hand_left": (0.42, 0.65),
        "torso": (0.5, 0.6),
    },

    "WHERE": {
        "head": (0.5, 0.2),
        "shoulder": (0.5, 0.35),
        "elbow_right": (0.6, 0.45),
        "hand_right": (0.6, 0.3),
        "elbow_left": (0.4, 0.45),
        "hand_left": (0.4, 0.3),
        "torso": (0.5, 0.6),
    }
}