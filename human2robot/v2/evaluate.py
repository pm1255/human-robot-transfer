"""Native RoboTwin single-task rollout, matched seeds, no proxy success metric."""

import argparse, json, os, sys, traceback, time, subprocess, hashlib
from pathlib import Path
import numpy as np
import torch
import cv2, yaml
from v2.policy import Policy


def config(root, out):
    with open(root / "task_config/demo_clean.yml") as f:
        args = yaml.safe_load(f)
    emb = yaml.safe_load((root / "task_config/_embodiment_config.yml").read_text())[
        "aloha-agilex"
    ]["file_path"]
    robot = yaml.safe_load((root / emb / "config.yml").read_text())
    camera = yaml.safe_load((root / "task_config/_camera_config.yml").read_text())[
        args["camera"]["head_camera_type"]
    ]
    args.update(
        task_name="adjust_bottle",
        task_config="demo_clean",
        ckpt_setting="human2robot_v2",
        policy_name="h2r_v2",
        left_robot_file=emb,
        right_robot_file=emb,
        dual_arm_embodied=True,
        left_embodiment_config=robot,
        right_embodiment_config=robot,
        head_camera_h=camera["h"],
        head_camera_w=camera["w"],
        eval_mode=True,
        render_freq=0,
        save_data=False,
        collect_data=False,
        eval_video_log=False,
        save_path=str(out / "scratch"),
    )
    return args


def frame_obs(obs):
    images = np.stack(
        [
            cv2.resize(
                obs["observation"][c]["rgb"], (128, 128), interpolation=cv2.INTER_AREA
            )
            for c in ["head_camera", "left_camera", "right_camera"]
        ]
    )
    state = np.asarray(obs["joint_action"]["vector"], np.float32)
    assert state.shape == (14,) and np.isfinite(state).all()
    return images, state


def run(a):
    root = Path(a.robotwin).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "description/utils"))
    from envs.adjust_bottle import adjust_bottle

    args = config(root, out)
    torch.set_num_threads(4)
    env = adjust_bottle()
    env.test_num = 0
    env.suc = 0
    if a.checkpoint:
        ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = Policy(ckpt["horizon"]).cuda().eval()
        model.load_state_dict(ckpt["model"], strict=True)
        st = {k: v.cuda() for k, v in ckpt["stats"].items()}
    records = []
    seeds = (
        json.loads(Path(a.seeds).read_text())
        if a.seeds
        else list(range(100000, 100000 + a.episodes))
    )
    for seed in seeds:
        started = time.time()
        record = {
            "seed": seed,
            "mode": "policy" if a.checkpoint else "expert",
            "valid": False,
            "success": False,
        }
        actions = []
        raw_actions = []
        states = []
        physical_before = []
        physical_after = []
        bottle_points = []
        ff = None
        active_arm = None
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            if not a.checkpoint:
                env.play_once()
                record.update(
                    valid=bool(env.plan_success),
                    success=bool(env.check_success()),
                    steps=getattr(env, "take_action_cnt", 0),
                )
                obs = env.get_obs()
                cv2.imwrite(
                    str(out / f"expert_{seed}.jpg"),
                    cv2.cvtColor(
                        obs["observation"]["head_camera"]["rgb"], cv2.COLOR_RGB2BGR
                    ),
                )
            else:
                record["valid"] = True
                env.set_instruction(
                    instruction="Adjust the bottle and lift it to the target position."
                )
                env.step_lim = a.max_steps or env.step_lim
                initial = frame_obs(env.get_obs())[1]
                record["initial_state"] = initial.tolist()
                record["initial_bottle_point"] = np.asarray(
                    env.bottle.get_functional_point(0)
                ).tolist()
                record["gripper_threshold"] = a.gripper_threshold
                record["step_limit"] = env.step_lim
                while env.take_action_cnt < env.step_lim:
                    obs = env.get_obs()
                    images, state = frame_obs(obs)
                    with torch.no_grad():
                        pred = model(
                            torch.as_tensor(images[None], device="cuda"),
                            (
                                torch.as_tensor(state[None], device="cuda")
                                - st["state_mean"]
                            )
                            / st["state_std"],
                        )[0]
                        pred = (
                            pred * st.get("target_std", st.get("delta_std"))
                            + st.get("target_mean", st.get("delta_mean"))
                        ).cpu().numpy() + (
                            state
                            if ckpt.get("action_representation", "delta") == "delta"
                            else 0
                        )
                    assert np.isfinite(pred).all()
                    if a.single_active_arm and active_arm is None:
                        scores = [
                            float(
                                np.linalg.norm(
                                    pred[:, off : off + 6] - initial[off : off + 6],
                                    axis=1,
                                ).sum()
                            )
                            for off in [0, 7]
                        ]
                        active_arm = int(np.argmax(scores))
                        record["active_arm_inferred"] = active_arm
                        record["active_arm_scores"] = scores
                        record["inactive_arm_constraint"] = (
                            "hold initial commanded pose; task-specific, selected from model prediction only"
                        )
                    for raw_command in pred[: a.execute]:
                        raw_actions.append(raw_command.copy())
                        command = raw_command.copy()
                        if active_arm is not None:
                            off = 7 * (1 - active_arm)
                            command[off : off + 7] = initial[off : off + 7]
                        command[[6, 13]] = (
                            command[[6, 13]] >= a.gripper_threshold
                        ).astype(np.float32)
                        if env.take_action_cnt >= env.step_lim or env.eval_success:
                            break
                        obs = env.get_obs()
                        rgb = obs["observation"]["head_camera"]["rgb"]
                        if ff is None:
                            h, w = rgb.shape[:2]
                            ff = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-v",
                                    "error",
                                    "-y",
                                    "-f",
                                    "rawvideo",
                                    "-pixel_format",
                                    "rgb24",
                                    "-video_size",
                                    f"{w}x{h}",
                                    "-framerate",
                                    "15",
                                    "-i",
                                    "-",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-threads",
                                    "1",
                                    "-crf",
                                    "25",
                                    "-pix_fmt",
                                    "yuv420p",
                                    str(out / f"rollout_{seed}.mp4"),
                                ],
                                stdin=subprocess.PIPE,
                            )
                        ff.stdin.write(np.ascontiguousarray(rgb).tobytes())
                        states.append(frame_obs(obs)[1])
                        bottle_points.append(
                            np.asarray(env.bottle.get_functional_point(0))[:3]
                        )
                        physical_before.append(
                            env.robot.get_left_arm_real_jointState()
                            + env.robot.get_right_arm_real_jointState()
                        )
                        actions.append(command.copy())
                        env.take_action(command, action_type="qpos")
                        physical_after.append(
                            env.robot.get_left_arm_real_jointState()
                            + env.robot.get_right_arm_real_jointState()
                        )
                    if env.take_action_cnt % 40 == 0:
                        (out / "progress.json").write_text(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "step": env.take_action_cnt,
                                    "elapsed": time.time() - started,
                                }
                            )
                        )
                    if env.eval_success:
                        break
                point = np.asarray(env.bottle.get_functional_point(0))
                dx = (
                    max(0, float(point[0]) + 0.15)
                    if env.qpose_tag == 0
                    else max(0, 0.15 - float(point[0]))
                )
                dz = max(0, 0.9 - float(point[2]))
                record["goal_region_distance_m"] = float(np.hypot(dx, dz))
                record["final_bottle_point"] = point.tolist()
                record.update(
                    valid=True,
                    success=bool(env.eval_success),
                    steps=env.take_action_cnt,
                    native_check_success=bool(env.check_success()),
                )
                np.savez_compressed(
                    out / f"actions_{seed}.npz",
                    state=states,
                    action=actions,
                    raw_action=raw_actions[: len(actions)],
                    physical_before=physical_before,
                    physical_after=physical_after,
                    bottle_point=bottle_points,
                )
                if actions:
                    jj = [i for i in range(14) if i not in [6, 13]]
                    aa = np.array(actions)
                    ss = np.array(states)
                    dev = np.abs(aa[:, jj] - ss[:, jj])
                    record["command_state_joint_mae_rad"] = float(dev.mean())
                    record["command_state_joint_p95_rad"] = float(
                        np.quantile(dev, 0.95)
                    )
                    record["command_physical_after_joint_mae_rad"] = float(
                        np.abs(aa[:, jj] - np.array(physical_after)[:, jj]).mean()
                    )
                    record["observation_state_semantics"] = (
                        "native drive targets, not measured qpos"
                    )
                    record["command_joint_range_rad"] = (
                        aa[:, jj].max(0) - aa[:, jj].min(0)
                    ).tolist()
                    record["joint_third_difference_per_command"] = (
                        float(np.abs(np.diff(aa[:, jj], n=3, axis=0)).mean())
                        if len(aa) > 3
                        else None
                    )
                    ff.stdin.write(
                        np.ascontiguousarray(
                            env.get_obs()["observation"]["head_camera"]["rgb"]
                        ).tobytes()
                    )
        except Exception as e:
            record["error"] = repr(e)
            traceback.print_exc()
        finally:
            if ff is not None:
                ff.stdin.close()
                ff.wait(timeout=30)
            try:
                env.close_env(clear_cache=True)
            except Exception:
                traceback.print_exc()
        record["seconds"] = time.time() - started
        records.append(record)
        (out / "results.json").write_text(json.dumps(records, indent=2))
        print(record, flush=True)
    valid = [r for r in records if r["valid"]]
    (out / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": (
                    hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
                    if a.checkpoint
                    else None
                ),
                "native_task_sha256": hashlib.sha256(
                    (root / "envs/adjust_bottle.py").read_bytes()
                ).hexdigest(),
                "records": records,
                "valid": len(valid),
                "invalid": len(records) - len(valid),
                "successes": sum(r["success"] for r in valid),
                "success_rate": (
                    sum(r["success"] for r in valid) / len(valid) if valid else None
                ),
                "criterion": "unmodified RoboTwin adjust_bottle.check_success",
                "warning": "inference command count is not fixed wall-clock 15Hz; native TOPP interpolation",
            },
            indent=2,
        )
    )
    if not a.checkpoint:
        (out / "valid_seeds.json").write_text(
            json.dumps([r["seed"] for r in records if r["valid"] and r["success"]])
        )
    if not valid:
        raise RuntimeError(
            "No valid episodes; simulation/inference failed, not a 0% policy score"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robotwin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--seeds")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--execute", type=int, default=4)
    p.add_argument("--gripper-threshold", type=float, default=0.5)
    p.add_argument("--single-active-arm", action="store_true")
    run(p.parse_args())
