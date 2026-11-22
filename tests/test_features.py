import numpy as np

from turkey_fog.features import window_features

CHANNELS = ["IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]


def test_window_features_btd():
    win = np.full((8, 5, 5), 270.0)
    win[CHANNELS.index("IR_039"), 2, 2] = 266.0   # centre pixel: IR10.8 − IR3.9 = 4 K
    f = window_features(win, CHANNELS)
    assert f["btd_center"] == 4.0
    assert f["btd_max"] == 4.0
    assert f["IR_108_center"] == 270.0
    assert len(f) == 12
