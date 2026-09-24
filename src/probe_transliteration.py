"""Estimate whether script romanization can retrieve currently missed links."""

from __future__ import annotations

import re
from collections import Counter

from anyascii import anyascii

from v1 import connect


PUNCT = re.compile("[^a-z0-9]+")
LEGAL = re.compile(r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|llp)\b")


def normalized(value: str) -> str:
    return PUNCT.sub(" ", anyascii(value or "").lower()).strip()


def name_key(value: str) -> str:
    return PUNCT.sub("", LEGAL.sub(" ", normalized(value)))


def main() -> None:
    con = connect()
    rows = con.execute("""
        SELECT s.country, s.business_name, t.business_name,
               s.name5, t.name5,
               s.business_address, t.business_address
        FROM train_truth g
        JOIN work_s1 s ON g.s1_id=s.entity_id
        JOIN train_targets t ON g.target_id=t.entity_id
        LEFT JOIN candidates c ON g.s1_id=c.s1_id AND g.target_id=c.target_id
        WHERE c.target_id IS NULL
    """).fetchall()
    con.close()
    counts = Counter()
    for country, sn, tn, old_s5, old_t5, sa, ta in rows:
        counts["total"] += 1
        counts[f"{country}_total"] += 1
        s_key = name_key(sn)
        t_key = name_key(tn)
        if any(ord(c) > 127 for c in (sn or "") + (tn or "")):
            counts["nonascii_name_either"] += 1
        if len(s_key)>=5 and s_key[:5] == t_key[:5]:
            counts["same_translit_name5"] += 1
            if old_s5 != old_t5:
                counts["new_translit_name5"] += 1
                counts[f"{country}_new_translit_name5"] += 1
        if s_key and s_key==t_key:
            counts["exact_translit_name"] += 1
        s_addr = normalized(sa)
        t_addr = normalized(ta)
        if len(s_addr)>=8 and s_addr[:8]==t_addr[:8]:
            counts["same_translit_address8"] += 1
        if country == "india":
            if len(s_key)>=5 and s_key[:5] == t_key[:5]:
                counts["india_same_translit_name5"] += 1
            if s_key and s_key==t_key:
                counts["india_exact_translit_name"] += 1
    for key, count in counts.most_common():
        print(key, count)


if __name__ == "__main__":
    main()
