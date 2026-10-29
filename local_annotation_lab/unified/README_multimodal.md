# Eight-episode multimodal pilot

Extends the existing unified site; do not rerun `build_pilot.py` over the new release without merging its manifest.

1. Input episode directories contain immutable `cam_0.mp4`, `source.json`, `segments.json`. Model guides are in `human2robot/object_annotation/unified_multimodal.py` and explicitly refer to full source frames.
2. SAM3 uses the server's official SAM3.1 predictor with project-local session/output adapters. The output adapter retains real tracker masks on frames without a detector cache. Regression: `python -m unittest tests.test_sam3_compat`.
3. Run preflight, then sam, render, depth stages. Preflight requires visible pot detection in at least 27/30 frames; visual review remains mandatory. Depth is independent of SAM and may run separately.
4. `audit_multimodal.py` checks timestamps, mask shape, area and exact box extent plus motion/area review flags. It cannot establish accuracy. `package_multimodal_outputs.py` validates complete movies, creates mask ZIPs and preserves large raw arrays on server.
5. Transfer `deliverables.tar.gz`, verify `transfer_manifest.json` hashes. Fill `visual_review.json` from actual observed images. `publish_multimodal.py` requires every sample review, full decode and matching source hashes before updating the site manifest. Do not claim independent acceptance.
6. Browser verify all samples, metadata and seeks, and keep `browser_validation.json`. Current 8 × 5 new movies cover 2,481 physical source frames. Earlier rejected outputs are separate, never release inputs.

Array conventions: masks are packed bits with `[height,width]` shape; instance index is the row in that frame's packed array; bbox `[x1,y1,x2,y2]` exclusive pixel bounds. Flow is previous→current in source pixels/frame, first frame invalid, no camera subtraction. Depth is uncalibrated monocular model output with predicted camera K. The manifest's old `rigid_object=false` from the cloth depth module is normalized in the delivered depth manifest to `rigid_constraint_applied=false`.

Temporal identity, exact gripper boundaries, task object identity and contact are all unverified. Q3/Q4 valid labels require instance review; Q5 requires common metric geometry; Q6/Q8 cannot be inferred from nearest 2D pixels; Q9 needs scene and collision annotation.
