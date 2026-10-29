#!/usr/bin/env python3
"""Run the per-recording annotation jobs on the 4090 devspace.

This runner consumes the *new* RGB stream produced by HumanEgo's raw-Aria
preprocessing.  It never reads the old ``assets/videos/multi`` display clips.
Every stage writes its own video/JSON pair under ``annotation_v2/assets`` so
the browser can show one annotation per video and stream it from the cluster.

The runner is deliberately fail-closed:

* missing model/checkpoint -> the stage is ``BLOCKED_*``;
* a proxy or precomputed HumanEgo preview is never promoted to ``EXACT_RUN``;
* the final manifest records the exact command and input used for every clip.

SAM3 and HaMeR are optional per stage.  Set their paths in the environment and
the script will run all available stages across every preprocessed recording.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], env: dict[str, str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, " ".join(cmd)


def find_rgb(recording: Path) -> Path | None:
    vis = recording / "preprocess" / "vis"
    candidates = [
        vis / "aria_cam_rgb.mp4",
        vis / "aria_cam_rgb_h264.mp4",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Keep this fallback narrow: only an RGB-named preview from the freshly
    # generated preprocess directory, never an arbitrary old display clip.
    for p in sorted(vis.glob("*rgb*.mp4")):
        if p.is_file():
            return p
    return None


def task_name(recording: Path) -> str:
    # .../<task>/aria/mps_<task>_<id>_vrs
    return recording.parent.parent.name


def recording_id(recording: Path) -> str:
    return recording.name.removeprefix("mps_").removesuffix("_vrs")


def persist_progress(rows: list[dict], out_root: Path, page_assets_root: Path, media_root: Path) -> None:
    """Persist every completed row so the streaming page updates incrementally."""
    manifest = {
        "schema": "humanego.exact_annotation_jobs.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_policy": "raw HumanEgo Aria recordings -> freshly extracted RGB; old display previews are excluded",
        "output_policy": "each layer has its own video/JSON; browser streams cluster files by HTTP Range",
        "rows": rows,
        "counts": {
            "recordings": len(rows),
            "exact_layer_rows": sum(any(v.get("status") == "EXACT_RUN" for v in x.get("layers", {}).values()) for x in rows),
            "blocked_rows": sum(x.get("status", "").startswith("BLOCKED") for x in rows),
        },
    }
    out = out_root / "assets" / "exact_annotation_jobs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_path = page_assets_root / "assets" / "multi_manifest.json"
    if not manifest_path.is_file():
        manifest_path = out_root / "assets" / "multi_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        page_manifest = json.loads(manifest_path.read_text())
        by_id = {r["source_clip_id"]: r for r in rows}

        def rel_media(value: str | None) -> str | None:
            if not value:
                return None
            p = Path(value).expanduser().resolve()
            try:
                return str(p.relative_to(media_root))
            except ValueError:
                return str(p)

        for page_row in page_manifest.get("clips", []):
            job = by_id.get(page_row.get("source_clip_id"))
            if not job or not job.get("input_video"):
                continue
            page_row.update({
                "video": rel_media(job["input_video"]),
                "raw_source_video": rel_media(job["input_video"]),
                "raw_source_status": "ready",
                "display_video_kind": "raw_rgb_cluster_stream",
                "reannotation_required": False,
                "reannotation_status": "v2_exact_layers",
                "annotation_revision": "v2_exact_layers",
            })
            for key, layer_name in (("hand", "hand"), ("hamer", "hamer"), ("sam3", "sam3"), ("affordance", "affordance")):
                layer = job.get("layers", {}).get(layer_name, {})
                if layer.get("status") == "EXACT_RUN":
                    # Hand annotations are split: skeleton and HaMeR 3-D mesh
                    # must never be put in one video.  Keep hand_video as the
                    # skeleton alias for older pages; hand_shape_video is an
                    # alias of the independent HaMeR artifact only.
                    if key == "hand":
                        page_row["hand_video"] = rel_media(layer.get("skeleton_video") or layer.get("video"))
                        page_row["hand_skeleton_video"] = page_row["hand_video"]
                        page_row.pop("hand_shape_video", None)
                        page_row.pop("hand_shape_json", None)
                    elif key == "hamer":
                        page_row["hamer_video"] = rel_media(layer.get("video"))
                        page_row["hamer_json"] = rel_media(layer.get("json"))
                        page_row["hand_shape_video"] = page_row["hamer_video"]
                        page_row["hand_shape_json"] = page_row["hamer_json"]
                    else:
                        page_row[f"{key}_video"] = rel_media(layer.get("video"))
                    page_row[f"{key}_json"] = rel_media(layer.get("json"))
                else:
                    page_row.pop(f"{key}_video", None)
                    page_row.pop(f"{key}_json", None)
                    if key == "hand":
                        page_row.pop("hand_skeleton_video", None)
                        page_row.pop("hand_shape_video", None)
            page_row["annotation_final"] = all(
                job.get("layers", {}).get(name, {}).get("status") == "EXACT_RUN"
                for name in ("hand", "hamer", "sam3", "affordance")
            )
        manifest_path.write_text(json.dumps(page_manifest, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=os.environ.get("RAW_ROOT", ""))
    ap.add_argument("--output-root", default=os.environ.get("OUT_ROOT", ""))
    ap.add_argument("--lab-root", default=Path(__file__).resolve().parent)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    lab_root = Path(args.lab_root).expanduser().resolve()
    if not raw_root.is_dir() or not out_root.is_dir():
        raise SystemExit("--raw-root and --output-root must already exist")

    hand_script = lab_root.parent / "remote_scripts" / "run_mediapipe_hand.py"
    hamer_script = lab_root.parent / "remote_scripts" / "run_hamer_mesh.py"
    sam3_script = lab_root / "run_sam3_video.py"
    affordance_script = lab_root / "run_sam3_affordance.py"
    hand_model = os.environ.get("HAND_LANDMARKER_MODEL", "")
    hamer_checkpoint = os.environ.get("HAMER_CHECKPOINT", "")
    sam3_checkpoint = os.environ.get("SAM3_CHECKPOINT", "")
    sam3_root = os.environ.get("SAM3_ROOT", "")
    run_hamer = os.environ.get("RUN_HAMER", "1") != "0"
    run_sam3 = os.environ.get("RUN_SAM3", "1") != "0"
    media_root = Path(os.environ.get("MEDIA_ROOT", str(out_root.parent))).expanduser().resolve()
    page_assets_root = Path(os.environ.get("PAGE_ASSETS_ROOT", str(out_root))).expanduser().resolve()

    # Semantic labels are the object-prompt source for SAM3.  The object
    # segmenter should receive the noun that Qwen verified, rather than a
    # generic task name such as "bread or croissant" whenever possible.
    language_nouns: dict[str, str] = {}
    seed_manifest_path = page_assets_root / "assets" / "multi_manifest.json"
    if seed_manifest_path.is_file():
        try:
            seed_manifest = json.loads(seed_manifest_path.read_text())
            language_nouns = {}
            for x in seed_manifest.get("clips", []):
                sid = str(x.get("source_clip_id"))
                noun = str(x.get("language_noun") or "").strip()
                # Prefer the verified Qwen object when the manifest stores it
                # only in the language JSON artifact.
                if not noun and x.get("language_json"):
                    try:
                        ref = str(x["language_json"])
                        p = Path(ref) if Path(ref).is_absolute() else page_assets_root / ref
                        payload = json.loads(p.read_text())
                        if payload.get("qwen_pass"):
                            noun = str((payload.get("annotation") or {}).get("object") or "").strip()
                    except Exception:
                        noun = ""
                if noun:
                    language_nouns[sid] = noun
        except Exception:
            language_nouns = {}

    rows: list[dict] = []
    recordings = sorted(p.parent for p in raw_root.rglob("sample.vrs"))
    if args.limit:
        recordings = recordings[: args.limit]
    for recording in recordings:
        task = task_name(recording)
        rid = recording_id(recording)
        stem = f"{task}_{rid}"
        clip_out = out_root / "assets" / "videos" / "exact" / stem
        clip_out.mkdir(parents=True, exist_ok=True)
        log_path = clip_out / "run.log"
        rgb = find_rgb(recording)
        row: dict = {
            "source_clip_id": rid,
            "task": task,
            "recording": recording.name,
            "raw_recording": str(recording),
            "input_video": str(rgb) if rgb else None,
            "status": "BLOCKED_INPUT",
            "layers": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if rgb is None:
            row["reason"] = "HumanEgo preprocess/vis/aria_cam_rgb.mp4 missing; raw recording exists but RGB extraction has not completed."
            rows.append(row)
            persist_progress(rows, out_root, page_assets_root, media_root)
            continue
        row["status"] = "RUNNING"
        base_env = os.environ.copy()
        base_env.update({"HUMANEGO_INPUT": str(rgb), "PYTHONUNBUFFERED": "1"})

        # Hand skeleton and discrete hand shape are emitted as two separate
        # videos by the hand runner.  The old mixed preview is never promoted
        # to an exact layer.
        if hand_model and hand_script.is_file():
            hand_out = clip_out / "hand"
            env = {**base_env, "HAND_LANDMARKER_MODEL": hand_model, "HUMANEGO_HAND_OUT": str(hand_out)}
            hand_json = hand_out / f"{rgb.stem}_hand.json"
            hand_skeleton_video = hand_out / f"{rgb.stem}_hand_skeleton.mp4"
            hand_shape_video = hand_out / f"{rgb.stem}_hand_shape.mp4"
            if hand_json.is_file() and hand_skeleton_video.is_file() and hand_shape_video.is_file():
                row["layers"]["hand"] = {"status": "EXACT_RUN", "json": str(hand_json), "video": str(hand_skeleton_video), "skeleton_video": str(hand_skeleton_video), "shape_video": str(hand_shape_video), "command": "reused existing exact output"}
            else:
                rc, command = run(["python", str(hand_script)], env, log_path)
                row["layers"]["hand"] = {"status": "EXACT_RUN" if rc == 0 and hand_json.is_file() and hand_skeleton_video.is_file() and hand_shape_video.is_file() else "FAILED", "json": str(hand_json), "video": str(hand_skeleton_video), "skeleton_video": str(hand_skeleton_video), "shape_video": str(hand_shape_video), "command": command}
        else:
            row["layers"]["hand"] = {"status": "BLOCKED_MODEL", "reason": "HAND_LANDMARKER_MODEL is not set"}

        # HaMeR consumes the hand JSON and produces an independent 3D-mesh
        # video.  It is not allowed to fall back to 2D skeletons.
        hamer_out = clip_out / "hamer"
        hamer_json = hamer_out / f"{rgb.stem}_hamer_mesh.json"
        hamer_video = hamer_out / f"{rgb.stem}_hamer_mesh.mp4"
        if hamer_json.is_file() and hamer_video.is_file():
            row["layers"]["hamer"] = {"status": "EXACT_RUN", "json": str(hamer_json), "video": str(hamer_video), "command": "reused existing exact output"}
        elif run_hamer and hamer_checkpoint and hamer_script.is_file():
            hand_out = clip_out / "hand"
            hand_json = hand_out / f"{rgb.stem}_hand.json"
            if hand_json.is_file():
                env = {**base_env, "HAMER_CHECKPOINT": hamer_checkpoint, "HUMANEGO_HAND_JSON": str(hand_json), "HAMER_OUT_DIR": str(hamer_out)}
                rc, command = run(["python", str(hamer_script)], env, log_path)
                row["layers"]["hamer"] = {"status": "EXACT_RUN" if rc == 0 and hamer_json.is_file() else "FAILED", "json": str(hamer_json), "video": str(hamer_video), "command": command}
            else:
                row["layers"]["hamer"] = {"status": "BLOCKED_DEPENDENCY", "reason": "hand JSON missing"}
        else:
            row["layers"]["hamer"] = {"status": "BLOCKED_MODEL", "reason": "HAMER_CHECKPOINT is not set"}

        # Official SAM3 mask and propagation, followed by separate affordance
        # and two-contact candidate videos.
        sam3_out = clip_out / "sam3"
        sam3_json = sam3_out / f"{rgb.stem}__sam3.json"
        sam3_video = sam3_out / f"{rgb.stem}__sam3.mp4"
        aff_json = sam3_out / "affordance" / f"{rgb.stem}__sam3_affordance.json"
        aff_video = sam3_out / "affordance" / f"{rgb.stem}__sam3_affordance.mp4"
        noun = language_nouns.get(rid)
        if noun and noun.lower().startswith("the "):
            noun = noun[4:]
        if noun and noun.lower() in {"unknown", "unknown object"}:
            noun = ""
        if sam3_json.is_file() and sam3_video.is_file():
            row["layers"]["sam3"] = {"status": "EXACT_RUN", "json": str(sam3_json), "video": str(sam3_video), "command": "reused existing exact output"}
            if aff_json.is_file() and aff_video.is_file():
                row["layers"]["affordance"] = {"status": "EXACT_RUN", "json": str(aff_json), "video": str(aff_video), "command": "reused existing exact output"}
            else:
                row["layers"]["affordance"] = {"status": "BLOCKED_DEPENDENCY", "reason": "SAM3 affordance output missing"}
        elif not noun:
            row["layers"]["sam3"] = {"status": "BLOCKED_DEPENDENCY", "reason": "Qwen 语义尚未通过或没有确认的目标名；不使用任务级猜测 prompt"}
            row["layers"]["affordance"] = {"status": "BLOCKED_DEPENDENCY", "reason": "等待 Qwen 确认目标名和 SAM3 mask"}
        elif run_sam3 and sam3_checkpoint and sam3_script.is_file():
            prompt = os.environ.get("SAM3_PROMPT") or f"the {noun}"
            env = {**base_env, "SAM3_CHECKPOINT": sam3_checkpoint, "SAM3_ROOT": sam3_root, "SAM3_OUT_DIR": str(sam3_out), "SAM3_PROMPT": os.environ.get("SAM3_PROMPT", prompt)}
            rc, command = run(["python", str(sam3_script), "--input", str(rgb), "--prompt", env["SAM3_PROMPT"], "--checkpoint", sam3_checkpoint, "--sam3-root", sam3_root, "--output-dir", str(sam3_out)], env, log_path)
            sam3_layer = {"status": "EXACT_RUN" if rc == 0 and sam3_json.is_file() else "FAILED", "json": str(sam3_json), "video": str(sam3_video), "command": command}
            row["layers"]["sam3"] = sam3_layer
            if sam3_json.is_file():
                rc2, command2 = run(["python", str(affordance_script), "--sam3-json", str(sam3_json), "--input", str(rgb), "--output-dir", str(sam3_out / "affordance")], env, log_path)
                row["layers"]["affordance"] = {"status": "EXACT_RUN" if rc2 == 0 and aff_json.is_file() else "FAILED", "json": str(aff_json), "video": str(aff_video), "command": command2}
            else:
                row["layers"]["affordance"] = {"status": "BLOCKED_DEPENDENCY", "reason": "SAM3 JSON missing"}
        else:
            row["layers"]["sam3"] = {"status": "BLOCKED_MODEL", "reason": "SAM3_CHECKPOINT is not set"}
            row["layers"]["affordance"] = {"status": "BLOCKED_DEPENDENCY", "reason": "SAM3 mask missing"}

        row["status"] = "EXACT_LAYERS_READY" if any(x.get("status") == "EXACT_RUN" for x in row["layers"].values()) else "BLOCKED_MODEL"
        rows.append(row)
        persist_progress(rows, out_root, page_assets_root, media_root)

    manifest = {
        "schema": "humanego.exact_annotation_jobs.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_policy": "raw HumanEgo Aria recordings -> freshly extracted RGB; old display previews are excluded",
        "output_policy": "each layer has its own video/JSON; browser streams cluster files by HTTP Range",
        "rows": rows,
        "counts": {
            "recordings": len(rows),
            "exact_layer_rows": sum(any(v.get("status") == "EXACT_RUN" for v in x["layers"].values()) for x in rows),
            "blocked_rows": sum(x["status"].startswith("BLOCKED") for x in rows),
        },
    }
    out = out_root / "assets" / "exact_annotation_jobs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    # If the cluster checkout contains the lab manifest, update only the
    # affected source rows.  Paths are made relative to the media root so the
    # same page can stream them with /media/<path>.
    media_root = Path(os.environ.get("MEDIA_ROOT", str(out_root.parent))).expanduser().resolve()
    page_assets_root = Path(os.environ.get("PAGE_ASSETS_ROOT", str(out_root))).expanduser().resolve()
    manifest_path = page_assets_root / "assets" / "multi_manifest.json"
    # A cluster runner can keep the static page in ``lab/`` while generated
    # videos live in ``annotation_v2/``.  Prefer the page's manifest so the
    # browser sees the same rows that were just reannotated.
    if not manifest_path.is_file():
        manifest_path = out_root / "assets" / "multi_manifest.json"
    if manifest_path.is_file():
        try:
            page_manifest = json.loads(manifest_path.read_text())
            by_id = {r["source_clip_id"]: r for r in rows}

            def rel_media(value: str | None) -> str | None:
                if not value:
                    return None
                p = Path(value).expanduser().resolve()
                try:
                    return str(p.relative_to(media_root))
                except ValueError:
                    return str(p)

            for page_row in page_manifest.get("clips", []):
                job = by_id.get(page_row.get("source_clip_id"))
                if not job or not job.get("input_video"):
                    continue
                page_row.update({
                    "video": rel_media(job["input_video"]),
                    "raw_source_video": rel_media(job["input_video"]),
                    "raw_source_status": "ready",
                    "display_video_kind": "raw_rgb_cluster_stream",
                    "reannotation_required": False,
                    "reannotation_status": "v2_exact_layers",
                    "annotation_revision": "v2_exact_layers",
                })
                for key, layer_name in (("hand", "hand"), ("hamer", "hamer"), ("sam3", "sam3"), ("affordance", "affordance")):
                    layer = job["layers"].get(layer_name, {})
                    if layer.get("status") != "EXACT_RUN":
                        continue
                    page_row[f"{key}_video"] = rel_media(layer.get("video"))
                    page_row[f"{key}_json"] = rel_media(layer.get("json"))
                page_row["annotation_final"] = all(
                    job["layers"].get(name, {}).get("status") == "EXACT_RUN"
                    for name in ("hand", "hamer", "sam3", "affordance")
                )
            manifest_path.write_text(json.dumps(page_manifest, ensure_ascii=False, indent=2) + "\n")
            manifest["page_manifest_updated"] = str(manifest_path)
            out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        except Exception as exc:
            manifest["page_manifest_update_error"] = str(exc)
            out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    status_paths = [out_root / "assets" / "runtime_status.json"]
    page_status = page_assets_root / "assets" / "runtime_status.json"
    if page_status not in status_paths:
        status_paths.append(page_status)
    for status_path in status_paths:
      if status_path.is_file():
        try:
            status = json.loads(status_path.read_text())
            exact_rows = manifest["counts"]["exact_layer_rows"]
            status["generated_at"] = datetime.now(timezone.utc).isoformat()
            status["status"] = "EXACT_LAYERS_PARTIAL" if exact_rows else "BLOCKED_INPUT_AND_MODELS"
            status.setdefault("annotation", {})["exact_job_rows"] = exact_rows
            status.setdefault("annotation", {})["exact_job_manifest"] = "assets/exact_annotation_jobs.json"
            status.setdefault("retargeting", {})["exact_robot_videos"] = 0
            status["retargeting"]["retargeting_note"] = "Annotation layers do not imply ForeHOI/GHOST/EgoMatrix robot reproduction."
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        except Exception:
            pass
    print(json.dumps({"manifest": str(out), **manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
