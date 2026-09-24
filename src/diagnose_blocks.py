"""Measure candidate coverage and block fanout on the deterministic pilot set."""

from v1 import connect


ROUTES = {
    "name5": ("name5",),
    "name4_city": ("name4", "city5"),
    "name4_number": ("name4", "addr_num"),
    "city_number": ("city5", "addr_num"),
    "address8": ("addr8",),
}


def main() -> None:
    con = connect()
    con.execute("DROP TABLE IF EXISTS pilot_links")
    con.execute(
        """
        CREATE TABLE pilot_links AS
        SELECT g.s1_id, g.target_id,
               s.name_key AS sn, t.name_key AS tn,
               s.addr_key AS sa, t.addr_key AS ta,
               s.name5 AS sn5, t.name5 AS tn5,
               s.name4 AS sn4, t.name4 AS tn4,
               s.city5 AS sc, t.city5 AS tc,
               s.addr_num AS sx, t.addr_num AS tx,
               s.addr8 AS sa8, t.addr8 AS ta8,
               s.country AS country
        FROM train_truth g
        JOIN work_s1 s ON g.s1_id = s.entity_id
        JOIN train_targets t ON g.target_id = t.entity_id
        """
    )
    total = con.execute("SELECT count(*) FROM pilot_links").fetchone()[0]
    print("Pilot positives:", total, flush=True)
    for label, fields in ROUTES.items():
        key_cols = ", ".join(fields)
        con.execute(f"DROP TABLE IF EXISTS stats_{label}")
        con.execute(
            f"CREATE TABLE stats_{label} AS "
            f"SELECT country, {key_cols}, count(*) AS n "
            f"FROM train_targets GROUP BY country, {key_cols}"
        )
        left = {"name5": "sn5", "name4": "sn4", "city5": "sc", "addr_num": "sx", "addr8": "sa8"}
        right = {"name5": "tn5", "name4": "tn4", "city5": "tc", "addr_num": "tx", "addr8": "ta8"}
        eq = " AND ".join(f"p.{left[f]} = p.{right[f]}" for f in fields)
        on = " AND ".join(f"p.{right[f]} = b.{f}" for f in fields)
        q = (
            "SELECT count(*) FILTER (WHERE " + eq + ") AS raw, "
            + ", ".join(
                f"count(*) FILTER (WHERE {eq} AND b.n <= {cap}) AS c{cap}"
                for cap in (50, 100, 200, 500, 1000)
            )
            + f" FROM pilot_links p LEFT JOIN stats_{label} b ON p.country=b.country AND {on}"
        )
        print(label, con.execute(q).fetchone(), flush=True)
    print("exact name", con.execute("SELECT count(*) FROM pilot_links WHERE sn=tn AND sn<>''").fetchone()[0])
    print("exact address", con.execute("SELECT count(*) FROM pilot_links WHERE sa=ta AND sa<>''").fetchone()[0])
    joins = """
        FROM pilot_links p
        LEFT JOIN stats_name5 n5 ON p.country=n5.country AND p.tn5=n5.name5
        LEFT JOIN stats_name4_city nc ON p.country=nc.country AND p.tn4=nc.name4 AND p.tc=nc.city5
        LEFT JOIN stats_name4_number nn ON p.country=nn.country AND p.tn4=nn.name4 AND p.tx=nn.addr_num
        LEFT JOIN stats_city_number cn ON p.country=cn.country AND p.tc=cn.city5 AND p.tx=cn.addr_num
        LEFT JOIN stats_address8 a8 ON p.country=a8.country AND p.ta8=a8.addr8
    """
    for cap in (50, 100, 200, 500):
        rule = (
            "(p.sn=p.tn AND p.sn<>'') OR (p.sa=p.ta AND p.sa<>'') OR "
            f"(p.sn5=p.tn5 AND length(p.sn5)=5 AND n5.n<={cap}) OR "
            f"(p.sn4=p.tn4 AND p.sc=p.tc AND length(p.sc)>=3 AND nc.n<={cap}) OR "
            f"(p.sn4=p.tn4 AND p.sx=p.tx AND p.sx<>'' AND nn.n<={cap}) OR "
            f"(p.sc=p.tc AND p.sx=p.tx AND p.sx<>'' AND cn.n<={cap}) OR "
            f"(p.sa8=p.ta8 AND length(p.sa8)=8 AND a8.n<={cap})"
        )
        covered = con.execute(f"SELECT count(*) FILTER (WHERE {rule}) {joins}").fetchone()[0]
        print("union cap", cap, covered, f"{covered/total:.4f}", flush=True)
        for label, fields in ROUTES.items():
            on = " AND ".join(f"s.{f}=b.{f}" for f in fields)
            pairs = con.execute(
                f"SELECT coalesce(sum(b.n),0) FROM work_s1 s JOIN stats_{label} b "
                f"ON s.country=b.country AND {on} WHERE b.n<={cap}"
            ).fetchone()[0]
            print("  ", label, "estimated pairs", pairs, flush=True)
    con.close()


if __name__ == "__main__":
    main()
