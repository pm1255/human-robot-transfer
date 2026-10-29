# Server runtime snapshots

These are the actual server code copies retained separately from the formatted local source. `v2-runtime/v2` came from human2robot-v2-20260923/code/v2. `v3-runtime/v3` came from human2robot-wan-v4-20260924/code/v3. The v4 server copy is already in human2robot/v4.

The v2 annotation script differs in file hash from the local maintained version; do not confuse the current source hash with an experiment's runtime hash. Dependencies, weights, datasets, and job logs are excluded.

`annotation-20261029/` preserves 116 source/config/document files from the later
annotation server, including scripts absent from the original public release.
Its manifest records each original byte size and SHA-256. Use the maintained
top-level modules for new work; this snapshot documents historical execution.
