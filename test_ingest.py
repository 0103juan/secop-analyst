import io
import json
import urllib.parse

import psycopg
import pytest
from pgvector.psycopg import register_vector

import ingest
from conftest import DSN, embed

pytestmark = pytest.mark.anyio


def portal(rows):
    """Stands in for datos.gov.co: answers a page from `rows` and keeps the filters it was asked with."""
    asked = []

    def get(url, timeout):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        asked.append(query["$where"])
        return io.BytesIO(json.dumps(rows[int(query["$offset"]):int(query["$offset"]) + ingest.PAGE]).encode())
    return get, asked


def row(key, text, updated="2025-05-01T00:00:00.000", **more):
    return {"id_contrato": key, "nombre_entidad": "ALCALDIA", "nit_entidad": "891580006", "objeto_del_contrato": text,
            "valor_del_contrato": "1000", "fecha_de_firma": "2025-04-01T00:00:00.000", "ultima_actualizacion": updated,
            "urlproceso": {"url": "https://example.org/" + key}, **more}


async def test_a_second_run_asks_only_for_what_changed_and_embeds_only_new_text(pool):
    counted = []

    def counting(texts):
        counted.append(len(texts))
        return embed(texts)

    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        conn.execute("TRUNCATE sync_state")
        get, asked = portal([row("N1", "Obra de acueducto"), row("N2", "Servicios de vigilancia"), row("", "sin id"),
                             row("N3", "  "), row("N4", "Valor raro", valor_del_contrato="No definido")])
        assert ingest.sync(conn, "departamento='Cauca'", counting, get=get) == (5, 3)

        later = [row("N1", "Obra de acueducto", "2025-06-01T00:00:00.000", estado_contrato="Cerrado"),
                 row("N2", "Servicios de vigilancia y aseo", "2025-06-01T00:00:00.000")]
        get, asked = portal(later)
        assert ingest.sync(conn, "departamento='Cauca'", counting, get=get) == (2, 1)

        assert asked == ["(departamento='Cauca') AND ultima_actualizacion >= '2025-05-01'"] and counted == [3, 1]
        rows = conn.execute("SELECT id, status, value_cop, url, embedding IS NOT NULL FROM contracts WHERE id LIKE 'N%' ORDER BY id").fetchall()
        assert rows == [("N1", "Cerrado", 1000, "https://example.org/N1", True), ("N2", None, 1000, "https://example.org/N2", True),
                        ("N4", None, None, "https://example.org/N4", True)]
