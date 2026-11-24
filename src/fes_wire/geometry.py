import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def z_rotation(degrees: float) -> np.ndarray:
    return Rotation.from_euler("z", degrees, degrees=True).as_matrix()


def fit(source: np.ndarray, target: np.ndarray) -> tuple:
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Alignment requires matching arrays of at least three atoms")
    p, q = source.mean(0), target.mean(0)
    if np.linalg.matrix_rank(source - p) < 2:
        raise ValueError("Collinear alignment is ambiguous")
    u, _, vt = np.linalg.svd((source - p).T @ (target - q))
    rotation = (u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt).T
    translation = q - rotation @ p
    rmsd = np.sqrt(np.mean(np.sum((source @ rotation.T + translation - target) ** 2, axis=1)))
    return rotation, translation, float(rmsd)


def contacts(a: np.ndarray, b: np.ndarray, cutoff: float) -> tuple:
    if not len(a) or not len(b):
        return 0, None
    tree = cKDTree(b)
    minimum = float(tree.query(a)[0].min())
    count = sum(len(x) for x in tree.query_ball_point(a, cutoff))
    return count, minimum


def unique_twists(values: list, order: int) -> list:
    return sorted({round(float(v) % (360 / order), 8) for v in values})
