import sys
sys.path.append("src")
import pandas as pd
from metar import mark_fog

def test_mark_fog():
    df = pd.DataFrame({"vsby": [0.12, 0.12, 5.0, 6.0],
                       "wxcodes": ["FZFG", "BCFG", "FG", None]})
    result = mark_fog(df)
    assert result["is_fog"].tolist() == [True, False, False, False]