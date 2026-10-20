"""One-off: dimensioned top-down comparison of the ADR-0007 Stage C test parts.

Left  = original Raspberry Pi 4 case (bench-002).
Right = new generic PCB tray (Stage C) -- same interpreter, same pi4_case pattern,
        different board facts (parametric pi4_mount).

Top-down (looking down +Z onto the tray floor), both drawn to the SAME scale so the
size difference is honest. Not loaded from the geometry; values mirror the two
LayoutSpec fixtures exactly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

WALL = 2.0
CLEAR = 2.0


class Tray:
    def __init__(
        self,
        title,
        board_l,
        board_w,
        board_t,
        dx,
        dy,
        od,
        screw_dia,
        screw_name,
        stand_h,
        ports,
        envelope,
        accent,
    ):
        self.title = title
        self.board_l, self.board_w, self.board_t = board_l, board_w, board_t
        self.dx, self.dy = dx, dy
        self.od, self.screw_dia, self.screw_name = od, screw_dia, screw_name
        self.stand_h, self.ports, self.envelope, self.accent = stand_h, ports, envelope, accent
        self.inner_l, self.inner_w = board_l + 2 * CLEAR, board_w + 2 * CLEAR
        self.outer_l, self.outer_w = self.inner_l + 2 * WALL, self.inner_w + 2 * WALL
        self.case_h = WALL + stand_h + 10.0  # WALL_ABOVE_STANDOFF = 10


PI4 = Tray(
    "ORIGINAL — Raspberry Pi 4 case  (bench-002)",
    85.0,
    56.0,
    1.5,
    58.0,
    49.0,
    7.0,
    2.7,
    "M2.5",
    4.0,
    "2 banks: 50mm (long) + 44mm (short)",
    (95, 70, 30),
    "#1f77b4",
)
GEN = Tray(
    "NEW — Generic PCB tray  (Stage C)",
    70.0,
    50.0,
    1.6,
    60.0,
    40.0,
    8.0,
    3.2,
    "M3",
    4.0,
    "none",
    (85, 65, 25),
    "#d62728",
)


def hdim(ax, x0, x1, y, label, color="#333"):
    ax.annotate("", (x0, y), (x1, y), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.1})
    ax.text((x0 + x1) / 2, y + 1.2, label, ha="center", va="bottom", fontsize=8.5, color=color)


def vdim(ax, y0, y1, x, label, color="#333"):
    ax.annotate("", (x, y0), (x, y1), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.1})
    ax.text(
        x - 1.5,
        (y0 + y1) / 2,
        label,
        ha="right",
        va="center",
        fontsize=8.5,
        color=color,
        rotation=90,
    )


def draw(ax, t: Tray):
    ol, ow, il, iw = t.outer_l, t.outer_w, t.inner_l, t.inner_w
    # Case outer + inner cavity.
    ax.add_patch(
        mpatches.Rectangle(
            (-ol / 2, -ow / 2), ol, ow, fill=True, facecolor="#f0f0f0", edgecolor="black", lw=1.6
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (-il / 2, -iw / 2), il, iw, fill=True, facecolor="white", edgecolor="#888", lw=1.0
        )
    )
    # Board footprint (dashed).
    ax.add_patch(
        mpatches.Rectangle(
            (-t.board_l / 2, -t.board_w / 2),
            t.board_l,
            t.board_w,
            fill=False,
            edgecolor=t.accent,
            lw=1.2,
            ls="--",
        )
    )
    # Standoffs + screw holes at the 4 hole-rect corners.
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * t.dx / 2, sy * t.dy / 2
            ax.add_patch(
                mpatches.Circle(
                    (cx, cy), t.od / 2, facecolor=t.accent, alpha=0.30, edgecolor=t.accent, lw=1.0
                )
            )
            ax.add_patch(
                mpatches.Circle(
                    (cx, cy), t.screw_dia / 2, facecolor="white", edgecolor="black", lw=0.8
                )
            )
    # Pi 4 ports: gaps in the +Y (long) and -X (short) walls.
    if "none" not in t.ports:
        ax.add_patch(
            mpatches.Rectangle(
                (-50 / 2, ow / 2 - WALL), 50, WALL, facecolor="#ff7f0e", edgecolor="none"
            )
        )
        ax.add_patch(
            mpatches.Rectangle((-ol / 2, -44 / 2), WALL, 44, facecolor="#ff7f0e", edgecolor="none")
        )
        ax.text(0, ow / 2 + 1, "port 50mm", ha="center", va="bottom", fontsize=7.5, color="#ff7f0e")
        ax.text(
            -ol / 2 - 1,
            0,
            "port 44mm",
            ha="right",
            va="center",
            fontsize=7.5,
            color="#ff7f0e",
            rotation=90,
        )

    # Dimensions: hole rectangle (accent), board (dashed accent), case outer (black).
    hdim(ax, -t.dx / 2, t.dx / 2, 0, f"hole dx {t.dx:.0f}", t.accent)
    vdim(ax, -t.dy / 2, t.dy / 2, t.dx / 2 - 1, f"dy {t.dy:.0f}", t.accent)
    hdim(
        ax, -t.board_l / 2, t.board_l / 2, -t.board_w / 2 - 5, f"board L {t.board_l:.0f}", t.accent
    )
    vdim(
        ax, -t.board_w / 2, t.board_w / 2, -t.board_l / 2 - 4, f"board W {t.board_w:.0f}", t.accent
    )
    hdim(ax, -ol / 2, ol / 2, ow / 2 + 7, f"case {ol:.0f}", "black")
    vdim(ax, -ow / 2, ow / 2, ol / 2 + 7, f"case {ow:.0f}", "black")

    info = (
        f"board: {t.board_l:.0f} x {t.board_w:.0f} x {t.board_t:.1f} mm\n"
        f"hole rect: {t.dx:.0f} x {t.dy:.0f} mm\n"
        f"screw: {t.screw_name}  (clearance dia {t.screw_dia:.1f}, r {t.screw_dia / 2:.2f})\n"
        f"standoff OD {t.od:.0f}, height {t.stand_h:.0f}  | wall {WALL:.0f}, clearance {CLEAR:.0f}\n"
        f"case (outer): {t.outer_l:.0f} x {t.outer_w:.0f} x {t.case_h:.0f} mm\n"
        f"envelope: {t.envelope[0]} x {t.envelope[1]} x {t.envelope[2]} mm\n"
        f"ports: {t.ports}"
    )
    ax.text(
        0,
        -ow / 2 - 16,
        info,
        ha="center",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "fc": "#fbfbfb", "ec": t.accent},
    )

    ax.set_title(t.title, fontsize=10.5, color=t.accent, pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(-62, 62)
    ax.set_ylim(-58, 48)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(14, 8))
draw(axes[0], PI4)
draw(axes[1], GEN)
fig.suptitle(
    "ADR-0007 Stage C — one tray interpreter, two boards (parametric pi4_mount)\n"
    "top-down view, both to the same scale",
    fontsize=12.5,
    y=0.99,
)
fig.text(
    0.5,
    0.015,
    "Same interpret_pi4 + pi4_case pattern. Only the board facts differ; the mount "
    "rectangle + screw size flow to the checker via the interface declaration params.",
    ha="center",
    fontsize=9,
    style="italic",
    color="#555",
)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))

out = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "results"
    / "_cli"
    / "stage_c_compare.png"
)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(out)
