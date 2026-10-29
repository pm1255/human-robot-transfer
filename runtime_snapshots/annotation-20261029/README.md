# Annotation runtime snapshot · 2026-10-29

This is a byte-preserving copy of 116 project files from the annotation server.
`MANIFEST.json` lists their relative paths, sizes, and SHA-256 values. It includes
21 files that were absent from the maintained local checkout at inspection time.
An empty historical `tools/build_annotation_views.py` is intentionally retained
as an empty file; it is not a working entry point.

The maintained implementation is in the repository's top-level `human2robot/`,
`tools/`, and `local_annotation_lab/`. Some maintained files include later fixes
or portable path arguments and therefore differ from this snapshot. The old
fast HaMeR renderer is historical; use `human2robot.hand_mesh.infer` for dense
surfaces. Original machine/data paths remain in this archive for provenance.

No model weights, datasets, prediction videos, Python environments, or login
credentials are included. This snapshot does not itself make an experiment
portable or establish the accuracy of its predictions.
