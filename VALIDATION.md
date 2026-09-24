# Validation — 2026-09-24

## Repository and gallery

- Repository is public; source push and visibility were verified on 2026-09-24.
- README contains 48 distinct animated previews: 24 annotation, 8 hand-to-arm, 4 body-to-G1, 12 robot-to-human variants from 6 source scenes.
- The linked public website retains 66 result videos / 53 distinct source clip IDs: 36 annotation, 8 arm, 4 humanoid, 18 generated variants.
- Each new GIF has a public full-video link, dataset attribution and source-preview SHA-256 in `docs/gallery/provenance.json`. Media are outside the code MIT license.
- README relative links, all 48 GIF paths and Python syntax were checked. Panoramic three-panel videos use one full-width table row; annotation previews use three columns and robot-to-human A/B previews use two.

## Startup validation

- All 33 existing Human2Robot tests passed.
- Downloaded all three official MediaPipe models using the new `models` command.
- Ran the new arm command on a 0.5-second EPIC clip: 5 frames processed through hand detection, trajectory processing, Aloha IK, mesh export and 1920×360 H.264 rendering.
- Ran the new humanoid command on a 0.5-second HMDB golf clip: 5 frames through hand/body/object annotation, provenance gating, annotation overlay and G1 rendering. Existing Aloha and compatible G1 29-DOF assets were used; downloads of the full third-party asset packages were not rerun.
- Fresh `bash scripts/setup.sh render` installation completed in an isolated virtual environment (Python 3.12, MediaPipe 0.10.32). All 33 tests and all three commands (`annotate`, `arm`, `humanoid`) passed again on five-frame clips in this clean environment.
- The first smoke runs used Python 3.12 / MediaPipe 0.10.21. A sandboxed macOS run failed to create an OpenGL context; the same command succeeded with normal access to the local graphics context. See QUICKSTART for the headless macOS limitation.
- No new model training or video generation quality improvement is claimed. The fresh VACE CUDA environment and custom-video GPU inference were not executed; the documented launcher wraps the previously used native inference path.

## Earlier release checks

- No checkpoints or original datasets are bundled.
- Existing Git history was scanned for common credential patterns with no matches. This was a bounded pattern scan, not a security certification.
