"""c_select: PQC cost simulation.

Wraps ML-KEM-768 (key exchange) and ML-DSA-65 (signatures) from the
`pqcrypto` package (PQClean bindings, NIST-standardized parameter sets) and
measures the real computational / communication overhead each client would
pay to secure one round of federated learning with post-quantum crypto.

This is the "Security Layer" described in the project write-up: it does not
invent new PQC, it just makes the standardized primitives measurable so the
"Selection Layer" (selection.py / strategy.py) can use the numbers.
"""

import time

from pqcrypto.kem.ml_kem_768 import (
    generate_keypair as kem_keypair,
    encrypt as kem_encrypt,
    decrypt as kem_decrypt,
)
from pqcrypto.sign.ml_dsa_65 import (
    generate_keypair as dsa_keypair,
    sign as dsa_sign,
    verify as dsa_verify,
)


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def measure_kem_cost() -> dict:
    """Time+size one ML-KEM-768 key-establishment exchange."""
    t0 = time.perf_counter()
    pk, sk = kem_keypair()
    t1 = time.perf_counter()
    ciphertext, shared_secret = kem_encrypt(pk)
    t2 = time.perf_counter()
    _ = kem_decrypt(sk, ciphertext)
    t3 = time.perf_counter()

    return {
        "kem_keygen_ms": _ms(t0, t1),
        "kem_encaps_ms": _ms(t1, t2),
        "kem_decaps_ms": _ms(t2, t3),
        "kem_pk_bytes": len(pk),
        "kem_ct_bytes": len(ciphertext),
    }


def measure_sign_cost(payload_bytes: int) -> dict:
    """Time+size one ML-DSA-65 sign/verify over a `payload_bytes`-sized message.

    `payload_bytes` should approximate the size of the model update the
    client is signing (e.g. len(pickled state_dict)).
    """
    message = b"\x00" * max(payload_bytes, 1)

    pk, sk = dsa_keypair()
    t0 = time.perf_counter()
    signature = dsa_sign(sk, message)
    t1 = time.perf_counter()
    ok = dsa_verify(pk, message, signature)
    t2 = time.perf_counter()

    return {
        "dsa_sign_ms": _ms(t0, t1),
        "dsa_verify_ms": _ms(t1, t2),
        "dsa_pk_bytes": len(pk),
        "dsa_sig_bytes": len(signature),
        "dsa_ok": ok,
    }


def simulate_client_pqc_overhead(model_update_bytes: int) -> dict:
    """Full per-round PQC cost a client pays: key exchange + signing its update.

    Returns a flat dict of metrics ready to be attached to a Flower
    MetricRecord and reported back to the server.
    """
    kem = measure_kem_cost()
    dsa = measure_sign_cost(model_update_bytes)

    total_time_ms = (
        kem["kem_keygen_ms"]
        + kem["kem_encaps_ms"]
        + kem["kem_decaps_ms"]
        + dsa["dsa_sign_ms"]
        + dsa["dsa_verify_ms"]
    )
    total_pqc_bytes = (
        kem["kem_pk_bytes"] + kem["kem_ct_bytes"] + dsa["dsa_pk_bytes"] + dsa["dsa_sig_bytes"]
    )

    return {
        **kem,
        **{k: v for k, v in dsa.items() if k != "dsa_ok"},
        "pqc_total_time_ms": total_time_ms,
        "pqc_total_bytes": total_pqc_bytes,
    }