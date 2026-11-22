import pandas as pd

from turkey_fog.metar import mark_events, mark_fog, pick_spread


def test_mark_fog():
    df = pd.DataFrame({"vsby": [0.12, 0.12, 5.0, 6.0, 0.3],
                       "wxcodes": ["FZFG", "BCFG", "FG", None, "BR FG"]})
    result = mark_fog(df)
    # FZFG at low visibility = fog; BCFG (patches) is not; FG with high visibility is not
    assert result["is_fog"].tolist() == [True, False, False, False, True]


def test_mark_events_splits_on_gap():
    fog = pd.DataFrame({"valid": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30",
                                                 "2020-01-01 03:00", "2020-01-01 03:30"])})
    assert mark_events(fog, "1h")["event"].tolist() == [0, 0, 1, 1]


def test_pick_spread_keeps_first_and_last():
    g = pd.DataFrame({"x": range(10)})
    picked = pick_spread(g, 3)["x"].tolist()
    assert len(picked) == 3 and picked[0] == 0 and picked[-1] == 9
