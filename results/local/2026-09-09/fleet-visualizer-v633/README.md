# Verified fleet v633 display evidence

The installed `gtoc12-fleet-v633` dataset displays the verified 24-ship fleet.
All 23 preceding ship JSON records are preserved exactly. The recovered ship
uses 20 saved event positions with 19 explicit unsampled gaps; its final mass
is the 501.4432517332 kg after cargo unloading. The first display export and
its correction are retained. No new propagation or optimization ran.

The Compute panel charges both checker attempts: 47.144631559 seconds of summed
worker time, with 23.189034797 seconds for the successful final pair. Archive
recovery, historical optimization and display export are outside that timing.

`evidence.zip` preserves all 35 sealed preparation, validation, schema, HTTP,
installation and ownership records, plus their exact index. The original source
and current installed files are referenced by recorded hashes. Archived
validation scripts require the repository inputs; the archive-only auditor
below checks saved bytes without running those scripts or any numerical work.

Run `python audit_archive.py` from this directory for portable byte verification.

Index SHA-256: `769fdc01be4c118d61b06e97294996a135e25681df17c931ea44b9fbc4af23ea`.
Archive SHA-256: `c63d9c5ab2ee15e1cbdbb7a328e05c306d89731a81d98b94fbdafc51d4cb71c5`.

[Verified Result and scientific checks](../fleet-addition-v633/README.md).
[Display and copy/paste loading instructions](../../../../docs/GTOC12_FLEET_ADDITION.md).
