"""Conservative provenance gate: ego wearer's whole body is not observed.
Raw model predictions remain available for error auditing. This gate does not
prove third-person correctness; those predictions still require visual review.
"""


def gate(annotation):
    meta = annotation["meta"]
    if "body_use_gate" in meta:
        return annotation
    if meta["view"] == "egocentric":
        meta["body_use_gate"] = (
            "rejected: egocentric wearer whole body not independently observed"
        )
        meta["raw_full_body_frames"] = meta["full_body_frames"]
        meta["full_body_frames"] = 0
        for f in annotation["frames"]:
            f["body_raw_prediction"] = f.pop("body", [])
            f["body_local_xyz_raw_prediction"] = f.pop("body_local_xyz", [])
            f["raw_full_body_visible"] = f["full_body_visible"]
            f.update(body=[], body_local_xyz=[], full_body_visible=False)
    else:
        meta["body_use_gate"] = (
            "third-person model candidate; not manually verified per frame"
        )
    return annotation
