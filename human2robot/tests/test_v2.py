import numpy as np
from v2.annotate import HandTracker


def hand(x, side):
    return {"xy": [[x, 0.5]] * 21, "side": side}


def test_two_hands_survive_detection_order_swap():
    t = HandTracker()
    a = t.update([hand(0.2, "Left"), hand(0.8, "Right")], 0)
    b = t.update([hand(0.79, "Right"), hand(0.21, "Left")], 0.1)
    assert (
        len(b) == 2
        and b[0]["track_id"] == a[1]["track_id"]
        and b[1]["track_id"] == a[0]["track_id"]
    )


def test_reappearance_after_gap_does_not_bridge_identity():
    t = HandTracker()
    a = t.update([hand(0.2, "Left")], 0)
    t.update([], 0.4)
    b = t.update([hand(0.2, "Left")], 0.5)
    assert b[0]["track_id"] != a[0]["track_id"]


def test_large_jump_gets_new_identity():
    t = HandTracker()
    a = t.update([hand(0.2, "Left")], 0)
    b = t.update([hand(0.9, "Left")], 0.1)
    assert a[0]["track_id"] != b[0]["track_id"]


def test_ego_false_body_is_retained_but_not_admitted():
    from v2.review_gate import gate

    d = {
        "meta": {"view": "egocentric", "full_body_frames": 1},
        "frames": [
            {
                "body": [[0.5, 0.5, 1, 1]],
                "body_local_xyz": [[0, 0, 0]],
                "full_body_visible": True,
            }
        ],
    }
    r = gate(d)
    assert r["meta"]["full_body_frames"] == 0
    assert r["frames"][0]["body"] == [] and r["frames"][0]["body_raw_prediction"]
    assert not r["frames"][0]["full_body_visible"]
