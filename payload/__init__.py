"""Recoverable-payload utilities (codec + passphrase-based crypto).

Used by the ``/api/embed`` and ``/api/extract`` paths to frame a self-describing
header around an (optionally encrypted) message before it is handed to a
generator's ``embed_payload``. See :mod:`payload.codec` and :mod:`payload.crypto`.
"""