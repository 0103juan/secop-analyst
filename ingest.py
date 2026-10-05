"""Loads contracts from SECOP II (datos.gov.co) into Postgres, with an embedding of what each contract is for.

    python ingest.py                                   # Cauca, signed since 2025
    python ingest.py --where "departamento='Huila'"    # another slice

It is incremental: each slice remembers the newest `ultima_actualizacion` it has seen and the next run only
asks upstream for rows from that day on. A row that comes again is updated, and embedded again only if its
text changed.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import psycopg
from pgvector.psycopg import register_vector

import db

DATASET = "https://www.datos.gov.co/resource/jbjy-vk9h.json"  # "SECOP II - Contratos Electrónicos"
DEFAULT = os.environ.get("INGEST_WHERE", "departamento='Cauca' AND fecha_de_firma >= '2025-01-01'")
FIELDS = ("id_contrato", "nombre_entidad", "nit_entidad", "departamento", "ciudad", "tipo_de_contrato",
          "modalidad_de_contratacion", "estado_contrato", "objeto_del_contrato", "proveedor_adjudicado",
          "fecha_de_firma", "valor_del_contrato", "ultima_actualizacion", "urlproceso")
PAGE = 2000

UPSERT = """
INSERT INTO contracts (id, entity, entity_nit, department, city, contract_type, modality, status, object, supplier,
                       signed_on, value_cop, updated_at, url, embedding)
VALUES (%(id_contrato)s, %(nombre_entidad)s, %(nit_entidad)s, %(departamento)s, %(ciudad)s, %(tipo_de_contrato)s,
        %(modalidad_de_contratacion)s, %(estado_contrato)s, %(objeto_del_contrato)s, %(proveedor_adjudicado)s,
        %(fecha_de_firma)s, %(valor_del_contrato)s, %(ultima_actualizacion)s, %(url)s, %(embedding)s)
ON CONFLICT (id) DO UPDATE SET
    entity = EXCLUDED.entity, entity_nit = EXCLUDED.entity_nit, department = EXCLUDED.department, city = EXCLUDED.city,
    contract_type = EXCLUDED.contract_type, modality = EXCLUDED.modality, status = EXCLUDED.status,
    object = EXCLUDED.object, supplier = EXCLUDED.supplier, signed_on = EXCLUDED.signed_on,
    value_cop = EXCLUDED.value_cop, updated_at = EXCLUDED.updated_at, url = EXCLUDED.url,
    embedding = coalesce(EXCLUDED.embedding, contracts.embedding)
"""


def fetch(where: str, offset: int, get=urllib.request.urlopen) -> list[dict]:
    query = urllib.parse.urlencode({"$select": ",".join(FIELDS), "$where": where, "$limit": PAGE, "$offset": offset,
                                    "$order": "ultima_actualizacion, id_contrato"})
    for attempt in range(4):  # the open-data portal times out now and then
        try:
            with get(f"{DATASET}?{query}", timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 5)


def number(value):
    """Upstream sends some numbers with text around them; a value that is not a number is stored as unknown."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def clean(row: dict) -> dict | None:
    if not row.get("id_contrato") or not (row.get("objeto_del_contrato") or "").strip():
        return None
    row = {field: row.get(field) for field in FIELDS}
    row["nit_entidad"], row["valor_del_contrato"] = number(row["nit_entidad"]), number(row["valor_del_contrato"])
    row["url"] = (row.pop("urlproceso") or {}).get("url")
    return row


def load(conn, rows: list[dict], embed) -> int:
    """Upserts a page. Only the rows that are new or whose text changed are embedded, which is the slow part."""
    rows = list({row["id_contrato"]: row for row in filter(None, map(clean, rows))}.values())
    known = dict(conn.execute("SELECT id, object FROM contracts WHERE id = ANY(%s) AND embedding IS NOT NULL",
                              ([row["id_contrato"] for row in rows],)).fetchall())
    changed = [row for row in rows if known.get(row["id_contrato"]) != row["objeto_del_contrato"]]
    vectors = embed([row["objeto_del_contrato"] for row in changed]) if changed else []
    for row in rows:
        row["embedding"] = None
    for row, vector in zip(changed, vectors):
        row["embedding"] = vector
    with conn.cursor() as cursor:
        cursor.executemany(UPSERT, rows)
    return len(changed)


def sync(conn, where: str, embed, full: bool = False, get=urllib.request.urlopen) -> tuple[int, int]:
    since = None if full else conn.execute("SELECT newest FROM sync_state WHERE slice = %s", (where,)).fetchone()
    # ultima_actualizacion has no time of day: asking from that day on repeats a few rows and misses none.
    scope = f"({where}) AND ultima_actualizacion >= '{since[0]:%Y-%m-%d}'" if since and since[0] else where
    seen = embedded = offset = 0
    while page := fetch(scope, offset, get):
        embedded += load(conn, page, embed)
        seen, offset = seen + len(page), offset + PAGE
        newest = max((row.get("ultima_actualizacion") or "" for row in page), default="")
        if newest:
            conn.execute("INSERT INTO sync_state (slice, newest) VALUES (%s, %s) ON CONFLICT (slice) DO UPDATE "
                         "SET newest = greatest(sync_state.newest, EXCLUDED.newest)", (where, newest))
        conn.commit()  # a page at a time: an interrupted run keeps what it loaded and resumes from its cursor
        print(f"  {seen} rows read, {embedded} embedded", flush=True)
        if len(page) < PAGE:
            break
    return seen, embedded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--where", default=DEFAULT, help="a SoQL filter over the upstream dataset")
    parser.add_argument("--full", action="store_true", help="ignore the cursor and read the whole slice again")
    args = parser.parse_args()
    start = time.perf_counter()
    with psycopg.connect(db.DSN) as conn:
        register_vector(conn)
        seen, embedded = sync(conn, args.where, db.fastembed(), args.full)
    print(f"{seen} rows read, {embedded} embedded, {time.perf_counter() - start:.0f} s")
