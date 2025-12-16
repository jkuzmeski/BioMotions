"""Plot COP/GRF from biomechanics_data.npz.

This reads the outputs written by protomotions/inference_biomechanics.py and plots:
- GRF (Fx,Fy,Fz) for left/right foot
- COP (x,y,z) for left/right foot
- optional foot-contact shading derived from summed vertical load fz_*

Example:
  D:/Isaac/BioMotions/Biomotions/Scripts/python.exe scripts/plot_biomechanics_cop_grf.py \
    --npz results/.../biomechanics/<experiment>/biomechanics_data.npz \
    --env 0 --out-dir output/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it into your venv, e.g.\n"
            "  pip install matplotlib\n"
            f"Original error: {exc}"
        )


def _load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out


def _get_dt_seconds(data: dict) -> float:
    if "dt" in data and np.isscalar(data["dt"]):
        return float(data["dt"])
    if "fps" in data and np.isscalar(data["fps"]):
        fps = float(data["fps"])
        if fps > 0:
            return 1.0 / fps
    # Fallback: 200 Hz (common in this repo’s IsaacLab configs)
    return 1.0 / 200.0


def _as_float_array(x) -> np.ndarray:
    arr = np.asarray(x)
    # numpy may store objects for some metadata fields; reject those
    if arr.dtype == object:
        raise ValueError("Encountered object array where numeric tensor expected")
    return arr.astype(np.float64, copy=False)


def _maybe_body_names(data: dict) -> list[str] | None:
    if "body_names" not in data:
        return None
    raw = data["body_names"]
    # torch-saved -> npz can become object arrays
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        return [str(x) for x in raw.tolist()]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    # scalar string or unknown
    return None


def _plot_xyz_timeseries(
    plt,
    t: np.ndarray,
    xyz: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    contact_mask: np.ndarray | None = None,
    contact_label: str | None = None,
):
    # xyz: [T,3]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, xyz[:, 0], label="x")
    ax.plot(t, xyz[:, 1], label="y")
    ax.plot(t, xyz[:, 2], label="z")

    if contact_mask is not None:
        # Shade regions where contact_mask is True
        # Convert to contiguous spans
        in_span = False
        span_start = 0
        for i in range(len(contact_mask)):
            if contact_mask[i] and not in_span:
                in_span = True
                span_start = i
            elif not contact_mask[i] and in_span:
                in_span = False
                ax.axvspan(t[span_start], t[i - 1], alpha=0.12, color="gray")
        if in_span:
            ax.axvspan(t[span_start], t[-1], alpha=0.12, color="gray")
        if contact_label:
            ax.text(
                0.995,
                0.02,
                contact_label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                alpha=0.7,
            )

    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot COP/GRF time series from biomechanics_data.npz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to biomechanics_data.npz",
    )
    parser.add_argument(
        "--env",
        type=int,
        default=0,
        help="Which environment index to plot (0-based)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/biomechanics_plots",
        help="Directory to write PNGs into",
    )
    parser.add_argument(
        "--fz-threshold",
        type=float,
        default=20.0,
        help="Vertical load threshold (N) to consider the foot in contact",
    )

    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = _load_npz(npz_path)
    dt = _get_dt_seconds(data)

    # Expect shapes:
    # grf_left/right: [T, E, 3]
    # cop_left/right: [T, E, 3]
    # fz_left/right:  [T, E]
    required_any = [
        ("grf_left", "cop_left", "fz_left"),
        ("grf_right", "cop_right", "fz_right"),
    ]
    has_any = any(all(k in data for k in group) for group in required_any)
    if not has_any:
        body_names = _maybe_body_names(data)
        hint = (
            "\nHint: ensure you ran inference with the contact-pad robot variant so COP/GRF is computed, e.g. "
            "robot=smpl_lower_body_170cm_contact_pads, and that contact sensors were enabled."
        )
        if body_names:
            hint += f"\nThis file includes body_names={body_names}"
        raise KeyError(
            "NPZ does not contain derived COP/GRF keys (grf_*/cop_*/fz_*)." + hint
        )

    # Determine length T from any present array
    sample_key = "grf_left" if "grf_left" in data else "grf_right"
    sample = _as_float_array(data[sample_key])
    if sample.ndim != 3:
        raise ValueError(f"Expected {sample_key} to be [T,E,3], got shape={sample.shape}")

    T, E, _ = sample.shape
    if args.env < 0 or args.env >= E:
        raise ValueError(f"--env out of range: {args.env} not in [0, {E-1}]")

    t = np.arange(T, dtype=np.float64) * dt

    plt = _require_matplotlib()

    def _plot_foot(side: str):
        grf_k = f"grf_{side}"
        cop_k = f"cop_{side}"
        fz_k = f"fz_{side}"
        if grf_k not in data or cop_k not in data or fz_k not in data:
            return

        grf = _as_float_array(data[grf_k])[:, args.env, :]
        cop = _as_float_array(data[cop_k])[:, args.env, :]
        fz = _as_float_array(data[fz_k])[:, args.env]
        contact_mask = fz > float(args.fz_threshold)

        _plot_xyz_timeseries(
            plt,
            t,
            grf,
            title=f"{side.upper()} foot GRF (env {args.env})",
            ylabel="force (N)",
            out_path=out_dir / f"grf_{side}_env{args.env}.png",
            contact_mask=contact_mask,
            contact_label=f"contact: fz>{args.fz_threshold:g}N",
        )

        _plot_xyz_timeseries(
            plt,
            t,
            cop,
            title=f"{side.upper()} foot COP (env {args.env})",
            ylabel="position (m)",
            out_path=out_dir / f"cop_{side}_env{args.env}.png",
            contact_mask=contact_mask,
            contact_label=f"contact: fz>{args.fz_threshold:g}N",
        )

    _plot_foot("left")
    _plot_foot("right")

    print(f"Wrote plots to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
