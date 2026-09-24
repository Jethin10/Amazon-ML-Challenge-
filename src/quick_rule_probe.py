"""Compare fast high-precision rules before making an urgent leaderboard file."""

from __future__ import annotations

import numpy as np
import pandas as pd

from train_v1 import score_at_threshold
from v1 import WORK, connect


def main() -> None:
    cols = ["s1_id", "label", "name_exact", "address_exact", "name_jaro",
            "name_sorted_jaro", "address_jaro", "address_sorted_jaro",
            "name_token_overlap", "address_token_overlap", "number_conflict"]
    frame = pd.read_parquet(WORK / "train_valid_features.parquet", columns=cols)
    con = connect()
    ids = [r[0] for r in con.execute("""
        SELECT entity_id FROM train_s1
        WHERE hash(entity_id) % 1000 >=42 AND hash(entity_id) % 1000 <50
    """).fetchall()]
    counts = dict(con.execute("""
        SELECT g.s1_id, count(*) FROM train_truth g JOIN train_s1 s
          ON g.s1_id=s.entity_id
        WHERE hash(s.entity_id) % 1000 >=42 AND hash(s.entity_id) % 1000 <50
        GROUP BY g.s1_id
    """).fetchall())
    con.close()
    position = {value: i for i, value in enumerate(ids)}
    group = frame.s1_id.map(position).to_numpy(dtype=np.int32)
    truth = np.array([counts.get(value, 0) for value in ids])
    label = frame.label.to_numpy(dtype=np.int8)
    ne = frame.name_exact == 1
    ae = frame.address_exact == 1
    aj = frame.address_jaro
    asj = frame.address_sorted_jaro
    nj = frame.name_jaro
    nsj = frame.name_sorted_jaro
    nt = frame.name_token_overlap
    at = frame.address_token_overlap
    nc = frame.number_conflict == 1
    rules = {
        "exact_name_address": ne & ((aj >= .72) | (at >= .5)) | ae & (nj >= .80),
        "exact_name_strict": ne & (asj >= .82) | ae & (nsj >= .90),
        "exact_or_fuzzy": (ne & ((aj >= .7) | (at >= .5))) |
            (ae & (nj >= .8)) |
            ((nj >= .93) & (at >= .6) & ~nc) |
            ((nsj >= .9) & (asj >= .9) & (nt >= .5) & ~nc),
        "fuzzy_strict": (ne & (asj >= .85)) |
            (ae & (nsj >= .85)) |
            ((nsj >= .93) & (asj >= .90) & ~nc),
    }
    for name, selected in rules.items():
        result = score_at_threshold(0.5, selected.to_numpy(dtype=np.float32),
                                    label, group, truth)
        print(name, "macro", round(result[0], 6),
              "singleton", round(result[1], 6),
              "selected", int(selected.sum()), flush=True)


if __name__ == "__main__":
    main()
