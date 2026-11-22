import numpy as np

from turkey_fog.metrics import scores


def test_scores():
    y = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])   # 2 hits, 1 miss, 1 false alarm
    pod, far, csi = scores(y, pred)
    assert pod == 2 / 3
    assert far == 1 / 3
    assert csi == 2 / 4


def test_scores_no_alarms():
    pod, far, csi = scores(np.array([1, 0]), np.array([0, 0]))
    assert pod == 0 and far == 0 and csi == 0
