#!/usr/bin/env python3
"""
Milestone 4 validation plots + summary from the controller logs (validation/runs/*.csv).

    python3 plot_validation.py                 # all runs in ./runs  -> ./figures + ./summary.md
    python3 plot_validation.py path/to/runs    # another log folder
    python3 plot_validation.py --latest 5      # only per-run figures for the 5 newest runs

What is checked (every number comes from the ROS2 + MuJoCo run):
  * joint tracking:  q_ref (trajectory) vs q (measured in the simulator)              [mrad]
  * task space:      TCP of the reference, FK(q_ref), vs TCP sensed in MuJoCo (/ee_pose)  [mm]
  * straightness:    distance of the SENSED TCP from the commanded straight line       [mm]
  * kinematics:      FK(q measured) with the DH equations vs TCP sensed in MuJoCo      [mm]
  * torques:         total motor torque and its gravity-compensation part              [N*m]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = range(1, 7)


# ============================================================================
# Loading
# ============================================================================
class Run:
    def __init__(self, csv_path):
        self.path = csv_path
        self.name = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path) as handle:
            header = handle.readline().strip().split(",")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        self.columns = {name: data[:, k] for k, name in enumerate(header)}
        meta_path = os.path.splitext(csv_path)[0] + ".json"
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.t = self.columns["t"]

    def get(self, prefix):
        return np.column_stack([self.columns[f"{prefix}{i}"] for i in JOINTS])

    def xyz(self, prefix):
        return np.column_stack([self.columns[f"{prefix}_{a}"] for a in "xyz"])

    @property
    def label(self):
        return self.meta.get("label", self.name)

    @property
    def experiment(self):
        return self.meta.get("experiment", "")

    @property
    def kind(self):
        return self.meta.get("kind", "joint")

    @property
    def moving(self):
        """Samples during the trajectory itself (the log also has a short settling tail)."""
        return self.t <= self.meta.get("duration", self.t[-1]) + 1e-9

    def metrics(self):
        error = self.get("q_ref") - self.get("q")
        tcp_ref, tcp_sim, tcp_fk = self.xyz("tcp_ref"), self.xyz("tcp_sim"), self.xyz("tcp_fk")
        valid = np.all(np.isfinite(tcp_sim), axis=1)
        tcp_error = np.linalg.norm(tcp_ref - tcp_sim, axis=1)[valid]
        kin_error = np.linalg.norm(tcp_fk - tcp_sim, axis=1)[valid]
        result = {
            "max_joint_error_mrad": 1000 * np.abs(error).max(),
            "rms_joint_error_mrad": 1000 * np.sqrt((error[self.moving] ** 2).mean()),
            "final_joint_error_mrad": 1000 * np.abs(error[-1]).max(),
            "max_tcp_error_mm": 1000 * tcp_error.max() if tcp_error.size else np.nan,
            "max_fk_vs_sim_mm": 1000 * kin_error.max() if kin_error.size else np.nan,
            "peak_torque": np.abs(self.get("tau")).max(axis=0),
        }
        if self.kind == "task" and valid.any():
            result["max_line_deviation_mm"] = 1000 * line_deviation(tcp_ref, tcp_sim, valid).max()
        return result


def line_deviation(tcp_ref, tcp_sim, valid):
    """Distance of each sensed TCP point from the commanded straight line (start -> end)."""
    start, end = tcp_ref[0], tcp_ref[-1]
    direction = end - start
    length = np.linalg.norm(direction)
    points = tcp_sim[valid]
    if length < 1e-9:
        return np.linalg.norm(points - start, axis=1)
    direction /= length
    offsets = points - start
    return np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)


def load_runs(folder):
    runs = [Run(p) for p in sorted(glob.glob(os.path.join(folder, "*.csv")))]
    return [r for r in runs if len(r.t) > 2]


# ============================================================================
# Per-run figure
# ============================================================================
def plot_run(run, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t = run.t
    q_ref, q = np.degrees(run.get("q_ref")), np.degrees(run.get("q"))
    colors = plt.cm.tab10(np.arange(6))

    ax = axes[0, 0]
    for j in range(6):
        ax.plot(t, q_ref[:, j], "--", color=colors[j], lw=1.2)
        ax.plot(t, q[:, j], "-", color=colors[j], lw=1.0, label=f"q{j + 1}")
    ax.set(title="Joint angles: reference (dashed) vs measured (solid)", xlabel="t [s]", ylabel="deg")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    error = 1000 * (run.get("q_ref") - run.get("q"))
    for j in range(6):
        ax.plot(t, error[:, j], color=colors[j], label=f"e{j + 1}")
    ax.axvline(run.meta.get("duration", t[-1]), color="k", ls=":", lw=0.8)
    ax.set(title="Joint tracking error q_ref - q", xlabel="t [s]", ylabel="mrad")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1, 0]
    tau, gravity = run.get("tau"), run.get("tau_gravity")
    for j in range(6):
        ax.plot(t, tau[:, j], color=colors[j], label=f"tau{j + 1}")
        ax.plot(t, gravity[:, j], ":", color=colors[j], lw=1.0)
    ax.set(title="Motor torques (solid) and gravity-compensation part G(q) (dotted)",
           xlabel="t [s]", ylabel="N*m")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1, 1]
    tcp_ref, tcp_sim, tcp_fk = run.xyz("tcp_ref"), run.xyz("tcp_sim"), run.xyz("tcp_fk")
    valid = np.all(np.isfinite(tcp_sim), axis=1)
    ax.plot(t[valid], 1000 * np.linalg.norm(tcp_ref - tcp_sim, axis=1)[valid],
            label="|TCP reference - TCP sensed in MuJoCo|")
    ax.plot(t[valid], 1000 * np.linalg.norm(tcp_fk - tcp_sim, axis=1)[valid],
            label="|DH FK(q measured) - TCP sensed| (kinematics)")
    if run.kind == "task":
        ax.plot(t[valid], 1000 * line_deviation(tcp_ref, tcp_sim, valid),
                label="distance of sensed TCP from the straight line")
    ax.set(title="Task-space validation", xlabel="t [s]", ylabel="mm")
    ax.legend(fontsize=8)

    m = run.metrics()
    fig.suptitle(f"{run.label}   [{run.experiment or run.meta.get('step', '')}]\n"
                 f"max joint error {m['max_joint_error_mrad']:.3f} mrad, "
                 f"max TCP error {m['max_tcp_error_mm']:.3f} mm, "
                 f"gravity comp {run.meta.get('gravity_compensation')}, "
                 f"feed-forward {run.meta.get('feedforward')}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run.name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ============================================================================
# Comparison figures
# ============================================================================
def first(runs, experiment):
    return next((r for r in runs if r.experiment == experiment), None)


def plot_profiles(runs, out_dir):
    """Cubic vs quintic joint-space profile: reference velocity, acceleration, torque."""
    cubic, quintic = first(runs, "profile-cubic"), first(runs, "profile-quintic")
    if not (cubic and quintic):
        return None
    joint = int(np.argmax(np.abs(cubic.get("q_ref")[-1] - cubic.get("q_ref")[0])))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for run, name in ((cubic, "cubic"), (quintic, "quintic")):
        axes[0].plot(run.t, np.degrees(run.get("q_ref")[:, joint]), label=name)
        axes[1].plot(run.t, run.get("qd_ref")[:, joint], label=name)
        axes[2].plot(run.t, run.get("qdd_ref")[:, joint], label=name)
        axes[3].plot(run.t, run.get("tau")[:, joint], label=name)
    titles = [f"q{joint + 1} reference [deg]", f"q{joint + 1}' reference [rad/s]",
              f"q{joint + 1}'' reference [rad/s^2]", f"motor torque {joint + 1} [N*m]"]
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel="t [s]")
        ax.legend()
    fig.suptitle("Joint-space trajectories: cubic (acceleration jumps at start/end) vs quintic "
                 "(smooth acceleration)")
    fig.tight_layout()
    path = os.path.join(out_dir, "compare_cubic_vs_quintic.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_control_laws(runs, out_dir):
    names = [("control-pid-only", "PID only"), ("control-pid+gravity", "PID + gravity compensation"),
             ("control-pid+gravity+feedforward", "PID + gravity comp. + inertia feed-forward")]
    found = [(first(runs, key), label) for key, label in names if first(runs, key)]
    if len(found) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for run, label in found:
        error = 1000 * np.linalg.norm(run.get("q_ref") - run.get("q"), axis=1)
        axes[0].plot(run.t, error, label=label)
        tcp_error = 1000 * np.linalg.norm(run.xyz("tcp_ref") - run.xyz("tcp_sim"), axis=1)
        axes[1].plot(run.t, tcp_error, label=label)
    axes[0].set(title="Joint tracking error norm |q_ref - q|", xlabel="t [s]", ylabel="mrad")
    axes[1].set(title="TCP error |FK(q_ref) - TCP sensed in MuJoCo|", xlabel="t [s]", ylabel="mm")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend()
    fig.suptitle("Motor-level control laws on the same quintic joint-space move")
    fig.tight_layout()
    path = os.path.join(out_dir, "compare_control_laws.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_paths(runs, out_dir):
    joint_run, task_run = first(runs, "path-joint-space"), first(runs, "path-task-space")
    if not (joint_run and task_run):
        return None
    fig, (ax_top, ax) = plt.subplots(1, 2, figsize=(15, 5))
    for run, name, color in ((joint_run, "joint-space quintic", "tab:blue"),
                             (task_run, "task-space straight line", "tab:orange")):
        sim, ref = run.xyz("tcp_sim"), run.xyz("tcp_ref")
        valid = np.all(np.isfinite(sim), axis=1)
        ax_top.plot(ref[:, 0], ref[:, 1], "--", color=color, lw=2.5, alpha=0.4, label=f"{name}: reference")
        ax_top.plot(sim[valid, 0], sim[valid, 1], "-", color=color, lw=1.2, label=f"{name}: sensed in MuJoCo")
        ax.plot(run.t[valid], 1000 * line_deviation(ref, sim, valid), color=color, label=name)
    start, end = task_run.xyz("tcp_ref")[0], task_run.xyz("tcp_ref")[-1]
    ax_top.plot(*start[:2], "ko")
    ax_top.annotate("start", start[:2], textcoords="offset points", xytext=(6, 6))
    ax_top.plot(*end[:2], "ks")
    ax_top.annotate("end", end[:2], textcoords="offset points", xytext=(6, 6))
    ax_top.set(title=f"TCP path, top view (both at z = {start[2]:.2f} m)", xlabel="x [m]", ylabel="y [m]")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend(fontsize=8)
    ax.set(title="Distance of the sensed TCP from the straight line start -> end", xlabel="t [s]", ylabel="mm")
    ax.legend()
    fig.suptitle("Same start and end pose: joint-space interpolation curves the TCP path, "
                 "task-space interpolation keeps it straight")
    fig.tight_layout()
    path = os.path.join(out_dir, "compare_joint_vs_task_space.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_line_scaling(runs, out_dir):
    cubic, quintic = first(runs, "line-cubic"), first(runs, "line-quintic")
    if not (cubic and quintic):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for run, name in ((cubic, "cubic s(t)"), (quintic, "quintic s(t)")):
        ref, sim = run.xyz("tcp_ref"), run.xyz("tcp_sim")
        valid = np.all(np.isfinite(sim), axis=1)
        dt = np.gradient(run.t)
        speed_ref = np.linalg.norm(np.gradient(ref, axis=0), axis=1) / dt
        speed_sim = np.linalg.norm(np.gradient(np.where(valid[:, None], sim, ref), axis=0), axis=1) / dt
        axes[0].plot(run.t, 1000 * (ref[:, 2] - ref[0, 2]), label=f"{name} reference")
        axes[1].plot(run.t, speed_ref, label=f"{name} reference")
        axes[1].plot(run.t, speed_sim, ":", label=f"{name} sensed")
        axes[2].plot(run.t[valid], 1000 * line_deviation(ref, sim, valid), label=name)
    axes[0].set(title="TCP height change [mm]", xlabel="t [s]")
    axes[1].set(title="TCP speed [m/s]", xlabel="t [s]")
    axes[2].set(title="Sensed distance from the straight line [mm]", xlabel="t [s]")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Task-space straight line (12 cm down): cubic vs quintic time scaling")
    fig.tight_layout()
    path = os.path.join(out_dir, "compare_line_time_scaling.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_task(runs, experiment, title, out_dir):
    """Whole application run: TCP path in 3D and the tracking error of every segment."""
    task_runs = [r for r in runs if r.experiment == experiment]
    if not task_runs:
        return None
    fig = plt.figure(figsize=(16, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax = fig.add_subplot(1, 2, 2)
    offset = 0.0
    for run in task_runs:
        sim = run.xyz("tcp_sim")
        valid = np.all(np.isfinite(sim), axis=1)
        color = "tab:blue" if run.kind == "joint" else "tab:orange"
        ax3d.plot(*sim[valid].T, color=color, lw=1.2)
        error = 1000 * np.abs(run.get("q_ref") - run.get("q")).max(axis=1)
        ax.plot(run.t + offset, error, color=color, lw=0.9)
        offset += run.t[-1]
    ax3d.plot([], [], color="tab:blue", label="joint-space quintic segments")
    ax3d.plot([], [], color="tab:orange", label="task-space straight-line segments")
    ax3d.set(title="TCP path sensed in MuJoCo", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax3d.legend(fontsize=8)
    ax.set(title="Largest joint tracking error of each segment (time of all motions joined)",
           xlabel="motion time [s]", ylabel="mrad")
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(out_dir, f"task_{experiment}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ============================================================================
# Summary
# ============================================================================
def write_summary(runs, figures, path):
    lines = ["# Milestone 4 validation summary (ROS2 + MuJoCo)", "",
             "Generated by `validation/plot_validation.py` from the controller logs in `validation/runs/`.",
             "Joint errors: reference q_ref vs angle measured in the simulator. TCP error: FK(q_ref) vs the "
             "TCP site sensed in MuJoCo (`/ee_pose`). Line deviation: sensed TCP distance from the commanded "
             "straight line. FK vs sim: DH forward kinematics of the measured angles vs the sensed TCP "
             "(includes up to 1 sim step of message offset).", "",
             "| # | experiment / step | trajectory | control | max joint err [mrad] | RMS [mrad] | "
             "max TCP err [mm] | line dev. [mm] | FK vs sim [mm] | peak torque [N*m] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, run in enumerate(runs, 1):
        m = run.metrics()
        control = "PID" + ("+G" if run.meta.get("gravity_compensation") else "") + \
                  ("+FF" if run.meta.get("feedforward") else "")
        where = run.experiment + (": " + run.meta["step"] if "step" in run.meta else "")
        lines.append(
            f"| {k} | {where} | {run.label} | {control} | {m['max_joint_error_mrad']:.3f} | "
            f"{m['rms_joint_error_mrad']:.3f} | {m['max_tcp_error_mm']:.3f} | "
            f"{m.get('max_line_deviation_mm', float('nan')):.3f} | {m['max_fk_vs_sim_mm']:.3f} | "
            f"{np.round(m['peak_torque'], 1).tolist()} |")
    lines += ["", "## Figures", ""]
    lines += [f"![{os.path.basename(f)}](figures/{os.path.basename(f)})" for f in figures if f]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", default=os.path.join(HERE, "runs"))
    parser.add_argument("--out", default=os.path.join(HERE, "figures"))
    parser.add_argument("--latest", type=int, default=0,
                        help="per-run figures only for the N newest runs (0 = all)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.runs)
    if not runs:
        print(f"no runs in {args.runs}")
        return
    figures = [plot_profiles(runs, args.out), plot_control_laws(runs, args.out),
               plot_paths(runs, args.out), plot_line_scaling(runs, args.out),
               plot_task(runs, "sorting", "Industrial application: sorting 4 sample vials into "
                         "colour-coded bins", args.out),
               plot_task(runs, "lift", "Lifting demo: Erlenmeyer flask (150 g)", args.out)]
    selected = runs[-args.latest:] if args.latest else runs
    per_run_dir = os.path.join(args.out, "runs")
    os.makedirs(per_run_dir, exist_ok=True)
    for run in selected:
        plot_run(run, per_run_dir)
    summary = os.path.join(os.path.dirname(os.path.abspath(args.out)), "summary.md")
    write_summary(runs, figures, summary)
    print(f"{len(runs)} runs -> {args.out} ({len(selected)} per-run figures), summary: {summary}")


if __name__ == "__main__":
    main()
