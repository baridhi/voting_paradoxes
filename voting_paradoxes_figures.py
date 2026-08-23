"""
voting_paradoxes_figures.py
============================
Reproduces Figures 1-5 of Gupta, Malakar & Sinha (2018), "Voting Paradoxes
in Four Candidate Elections", by computing every labeled point exactly the
same way `voting_paradoxes.py` does for the tables (same centroid/shrink
helpers, same exact-Fraction vertex coordinates), then rendering them with
matplotlib.

Two coordinate systems are used, matching the paper:

- 3-candidate points are barycentric triples (alpha_A, alpha_B, alpha_C)
  with sum 1, plotted inside a 2D equilateral "Saari" triangle (Figure 1).
- 4-candidate points are barycentric quadruples (alpha_A, alpha_B, alpha_C,
  alpha_D) with sum 1, plotted inside a 3D tetrahedron (Figures 2-5).

Nothing here re-derives new mathematics -- every point is built from the
same vertex definitions already validated against the paper's tables in
`voting_paradoxes.py` (see that module's `geometric_bounds_3` /
`geometric_bounds_4` for the underlying derivations). This module is purely
about turning those already-verified points into pictures.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from voting_paradoxes import centroid, shrink

F = Fraction
Point3 = Tuple[float, float, float]
Point4 = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Coordinate conversion: barycentric -> plotting coordinates
# ---------------------------------------------------------------------------

# Figure 1 (3-candidate): standard equilateral-triangle embedding, laid out
# to match the paper's own layout (A bottom-left, B bottom-right, C top).
TRIANGLE_2D_VERTICES = {
    "A": (0.0, 0.0),
    "B": (1.0, 0.0),
    "C": (0.5, 3 ** 0.5 / 2),
}


def bary3_to_2d(point: Sequence[float]) -> Tuple[float, float]:
    """(alpha_A, alpha_B, alpha_C) -> 2D Cartesian point inside the triangle
    with vertices A, B, C as laid out in `TRIANGLE_2D_VERTICES`."""
    a, b, c = (float(x) for x in point)
    ax, ay = TRIANGLE_2D_VERTICES["A"]
    bx, by = TRIANGLE_2D_VERTICES["B"]
    cx, cy = TRIANGLE_2D_VERTICES["C"]
    return (a * ax + b * bx + c * cx, a * ay + b * by + c * cy)


# Figures 2-5 (4-candidate): embed a regular tetrahedron in 3D. The mapping
# from the 4 candidate-labels to the 4 canonical 3D vertices is chosen
# per-figure below purely to match each figure's described orientation in
# the paper (e.g. Figure 2 has A "on top"; Figure 4 has D "on top") -- this
# is a plotting choice only, it does not change any of the underlying
# barycentric mathematics.
import math

# A regular tetrahedron with one vertex "up" (apex) and the other three
# forming an equilateral triangle "base" below -- much easier to read in a
# 2D projection than a cube-corner embedding, and matches how the paper's
# own Figures 2-5 are drawn (one vertex singled out at the top).
_REGULAR_TETRA_CANONICAL = [
    (0.0, 0.0, 1.0),                                   # apex
    (math.sqrt(8) / 3, 0.0, -1 / 3),                   # base vertex 1
    (-math.sqrt(2) / 3, math.sqrt(6) / 3, -1 / 3),     # base vertex 2
    (-math.sqrt(2) / 3, -math.sqrt(6) / 3, -1 / 3),    # base vertex 3
]


def make_tetra_embedding(vertex_order: Sequence[str]) -> Dict[str, Point3]:
    """Assign each of the 4 labels in `vertex_order` to one of the 4
    canonical regular-tetrahedron corners, in order (first label -> first
    corner, etc.)."""
    assert len(vertex_order) == 4
    return dict(zip(vertex_order, _REGULAR_TETRA_CANONICAL))


def bary4_to_3d(point: Sequence[float], embedding: Dict[str, Point3], label_order: Sequence[str] = ("A", "B", "C", "D")) -> Point3:
    """(alpha_A, alpha_B, alpha_C, alpha_D) -> 3D Cartesian point, using the
    given label->corner `embedding` (see `make_tetra_embedding`)."""
    weights = [float(x) for x in point]
    total = (0.0, 0.0, 0.0)
    for w, lab in zip(weights, label_order):
        vx, vy, vz = embedding[lab]
        total = (total[0] + w * vx, total[1] + w * vy, total[2] + w * vz)
    return total


# ---------------------------------------------------------------------------
# FIGURE 1: The Saari triangle for 3 candidates
# ---------------------------------------------------------------------------

def figure1_points() -> Dict[str, Tuple[float, float, float]]:
    """All labeled points in Figure 1, using exactly the same construction
    as `voting_paradoxes.geometric_bounds_3` (verified against Table 4)."""
    A = (F(1), F(0), F(0))
    B = (F(0), F(1), F(0))
    C = (F(0), F(0), F(1))
    D = (F(1, 2), F(1, 2), F(0))          # midpoint AB
    E = (F(1, 2), F(0), F(1, 2))          # midpoint AC
    Fp = (F(0), F(1, 2), F(1, 2))         # midpoint BC
    H = (F(1, 3), F(1, 3), F(1, 3))       # centroid of ABC
    G = centroid([D, E])                  # midpoint of D,E
    P = centroid([G, D, H])               # centroid of GDH (== centroid of IJK)
    I = tuple((P[i] + G[i]) / 2 for i in range(3))
    J = tuple((P[i] + H[i]) / 2 for i in range(3))
    K = tuple((P[i] + D[i]) / 2 for i in range(3))
    return {
        "A": A, "B": B, "C": C, "D": D, "E": E, "F": Fp, "H": H,
        "G": G, "P": P, "I": I, "J": J, "K": K,
    }


def plot_figure1(annotate_fractions: bool = True, figsize=(8, 7)):
    """Reproduce Figure 1: the Saari triangle, its three medians (meeting at
    centroid H), the sub-triangle G-D-H (the region with no Strong Condorcet
    Winner and alpha_A > alpha_B > alpha_C), and the innermost triangle I-J-K
    (the OWNCM / Condorcet-Paradox region underlying Table 4)."""
    pts = figure1_points()
    pts2d = {k: bary3_to_2d(v) for k, v in pts.items()}

    fig, ax = plt.subplots(figsize=figsize)

    def line(p1, p2, **kw):
        (x1, y1), (x2, y2) = pts2d[p1], pts2d[p2]
        ax.plot([x1, x2], [y1, y2], **kw)

    # Outer triangle
    for p1, p2 in [("A", "B"), ("B", "C"), ("C", "A")]:
        line(p1, p2, color="tab:blue", lw=1.5, zorder=1)

    # Three medians (each meets the opposite vertex; all cross at H)
    line("A", "F", color="tab:blue", lw=0.8, ls="--", zorder=1)
    line("B", "E", color="tab:blue", lw=0.8, ls="--", zorder=1)
    line("C", "D", color="tab:blue", lw=0.8, ls="--", zorder=1)

    # Sub-triangle G-D-H (no-SCW, alpha_A>alpha_B>alpha_C region)
    for p1, p2 in [("G", "D"), ("D", "H"), ("H", "G")]:
        line(p1, p2, color="tab:green", lw=1.6, zorder=2)

    # Innermost triangle I-J-K (OWNCM region -> Table 4)
    for p1, p2 in [("I", "J"), ("J", "K"), ("K", "I")]:
        line(p1, p2, color="tab:red", lw=1.8, zorder=3)

    for lab, (x, y) in pts2d.items():
        ax.scatter([x], [y], color="black", s=18, zorder=4)
        offset = {"A": (-0.05, -0.04), "B": (0.02, -0.04), "C": (0, 0.03)}.get(lab, (0.012, 0.012))
        ax.annotate(lab, (x, y), xytext=(x + offset[0], y + offset[1]), fontsize=11, fontweight="bold")
        if annotate_fractions and lab in ("H", "G", "P", "I", "J", "K"):
            a, b, c = pts[lab]
            txt = f"({float(a):.3f}, {float(b):.3f}, {float(c):.3f})"
            ax.annotate(txt, (x, y), xytext=(x + offset[0], y + offset[1] - 0.035), fontsize=7.5, color="dimgray")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Figure 1 — Saari triangle: OWNCM region (red) inside\n"
                  "the no-Strong-Condorcet-Winner region G-D-H (green)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FIGURE 2: The tetrahedron for 4 candidates (full construction, case a)
# ---------------------------------------------------------------------------

def figure2_points() -> Dict[str, Tuple[float, float, float, float]]:
    """All labeled points in Figure 2: the full tetrahedron ABCD, its edge
    midpoints, and the points X, Y (on face BCD, alpha_A=0) and X', Y'
    (on the alpha_A=0.5 cutting plane) plus the overall centroid O."""
    A = (F(1), F(0), F(0), F(0))
    B = (F(0), F(1), F(0), F(0))
    C = (F(0), F(0), F(1), F(0))
    D = (F(0), F(0), F(0), F(1))
    AB = centroid([A, B]); AC = centroid([A, C]); AD = centroid([A, D])
    BC = centroid([B, C]); BD = centroid([B, D]); CD = centroid([C, D])
    X = centroid([BC, BD])           # midpoint of BC,BD (on face BCD)
    Y = centroid([B, C, D])          # centroid of face BCD
    Xp = centroid([AB, AC])          # =B'C', renamed below for clarity  (on alpha_A=0.5 plane)
    Yp = centroid([AB, AC, AD])      # centroid of the alpha_A=0.5 plane triangle
    O = centroid([A, B, C, D])       # centroid of the whole tetrahedron
    return {
        "A": A, "B": B, "C": C, "D": D,
        "AB": AB, "AC": AC, "AD": AD, "BC": BC, "BD": BD, "CD": CD,
        "X": X, "Y": Y, "X'": Xp, "Y'": Yp, "O": O,
    }


def plot_figure2(figsize=(8, 8), elev=15, azim=-60):
    """Reproduce Figure 2: the outer tetrahedron ABCD with A "on top", its
    edge midpoints, and the constructed points X, Y (face BCD) and X', Y'
    (the alpha_A=0.5 plane) used to build the Case (a)/(b) sub-regions."""
    pts = figure2_points()
    embedding = make_tetra_embedding(["A", "B", "C", "D"])  # A on top for this figure
    pts3d = {k: bary4_to_3d(v, embedding) for k, v in pts.items()}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def line(p1, p2, **kw):
        (x1, y1, z1), (x2, y2, z2) = pts3d[p1], pts3d[p2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], **kw)

    # Outer tetrahedron edges
    for p1, p2 in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
        line(p1, p2, color="tab:blue", lw=1.4)

    # Face BCD internal construction (X, Y)
    for p1, p2 in [("B", "X"), ("C", "X"), ("D", "X"), ("Y", "B"), ("Y", "C"), ("Y", "D")]:
        line(p1, p2, color="tab:green", lw=0.8, ls="--")

    # alpha_A=0.5 plane (AB-AC-AD triangle) and its own construction (X', Y')
    for p1, p2 in [("AB", "AC"), ("AC", "AD"), ("AD", "AB")]:
        line(p1, p2, color="tab:orange", lw=1.2)
    for p1, p2 in [("AB", "X'"), ("AC", "X'"), ("AD", "Y'")]:
        line(p1, p2, color="tab:orange", lw=0.7, ls="--")

    for lab, (x, y, z) in pts3d.items():
        ax.scatter([x], [y], [z], color="black", s=22)
        ax.text(x, y, z, f"  {lab}", fontsize=9, fontweight="bold" if len(lab) == 1 else "normal")

    ax.set_title("Figure 2 — Tetrahedron ABCD (A on top) with midpoints,\n"
                  "face-BCD construction (X, Y) and the alpha_A=0.5 plane (X', Y')")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FIGURE 3: Case (a) — outer X'Y'-B'C'-O and inner shrunk X''Y''-B''C''-O'
# ---------------------------------------------------------------------------

def figure3_points() -> Tuple[Dict[str, Point4], Dict[str, Point4], Point4]:
    """Outer tetrahedron X'-Y'-B'C'-O, inner shrunk tetrahedron
    X''-Y''-B''C''-O', and their shared centroid G1. Reuses exactly the
    same construction as `voting_paradoxes.geometric_bounds_4('a')`, which
    is independently verified against the paper's Table 5."""
    Xp = (F(1, 2), F(1, 4), F(1, 8), F(1, 8))
    Yp = (F(1, 2), F(1, 6), F(1, 6), F(1, 6))
    BpCp = (F(1, 2), F(1, 4), F(1, 4), F(0))
    O = (F(1, 4), F(1, 4), F(1, 4), F(1, 4))
    outer = {"X'": Xp, "Y'": Yp, "B'C'": BpCp, "O": O}

    G1 = centroid(list(outer.values()))
    ratio = (372 / 1296) ** (1 / 3)
    inner_raw = {k: shrink(tuple(float(x) for x in v), tuple(float(x) for x in G1), ratio)
                 for k, v in outer.items()}
    inner = {f"{k}\u2032\u2032".replace("''", "''"): v for k, v in inner_raw.items()}
    # relabel: X'->X'', Y'->Y'', B'C'->B''C'', O->O'
    inner = {
        "X''": inner_raw["X'"], "Y''": inner_raw["Y'"],
        "B''C''": inner_raw["B'C'"], "O'": inner_raw["O"],
    }
    return outer, inner, G1


def plot_figure3(figsize=(8, 8), elev=15, azim=-60):
    """Reproduce Figure 3: the outer tetrahedron X'-Y'-B'C'-O (Case a) with
    the smaller, similar, centroid-sharing tetrahedron X''-Y''-B''C''-O'
    (the OWNCM region behind Table 5) drawn nested inside it."""
    outer, inner, G1 = figure3_points()
    embedding = make_tetra_embedding(["A", "B", "C", "D"])

    all_pts = {**outer, **inner, "G1": G1}
    pts3d = {k: bary4_to_3d(v, embedding) for k, v in all_pts.items()}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def line(p1, p2, **kw):
        (x1, y1, z1), (x2, y2, z2) = pts3d[p1], pts3d[p2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], **kw)

    outer_labels = ["X'", "Y'", "B'C'", "O"]
    inner_labels = ["X''", "Y''", "B''C''", "O'"]
    for i in range(4):
        for j in range(i + 1, 4):
            line(outer_labels[i], outer_labels[j], color="tab:blue", lw=1.6)
            line(inner_labels[i], inner_labels[j], color="tab:red", lw=1.8)

    for lab, (x, y, z) in pts3d.items():
        color = "tab:red" if lab in inner_labels else ("tab:purple" if lab == "G1" else "black")
        ax.scatter([x], [y], [z], color=color, s=24)
        ax.text(x, y, z, f"  {lab}", fontsize=9, fontweight="bold", color=color)

    ax.set_title("Figure 3 — Case (a): outer tetrahedron X'Y'-B'C'-O (blue)\n"
                 "and inner OWNCM tetrahedron X''Y''-B''C''-O' (red), shared centroid G1")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FIGURE 4: Tetrahedron for 4 candidates, oriented for Case (b) (D on top)
# ---------------------------------------------------------------------------

def figure4_points() -> Dict[str, Point4]:
    """All labeled points in Figure 4: tetrahedron ABCD (redrawn with D on
    top), and the points Q (midpoint AB), R (centroid ABC), S (midpoint of
    AC and Q -- read from the paper as midpoint of Q and AC), and P (overall
    centroid)."""
    A = (F(1), F(0), F(0), F(0))
    B = (F(0), F(1), F(0), F(0))
    C = (F(0), F(0), F(1), F(0))
    D = (F(0), F(0), F(0), F(1))
    AC = centroid([A, C])
    BC = centroid([B, C])
    Q = centroid([A, B])
    R = centroid([A, B, C])
    S = centroid([AC, Q])
    P = centroid([A, B, C, D])
    return {"A": A, "B": B, "C": C, "D": D, "AC": AC, "BC": BC, "Q": Q, "R": R, "S": S, "P": P}


def plot_figure4(figsize=(8, 8), elev=15, azim=-60):
    """Reproduce Figure 4: the tetrahedron ABCD redrawn with D "on top" and
    A, B, C on the base, plus the points Q, R, S, P used to build the
    Case (b) OWNCM region (Figure 5 / Table 6)."""
    pts = figure4_points()
    embedding = make_tetra_embedding(["D", "A", "B", "C"])  # D on top for this figure
    pts3d = {k: bary4_to_3d(v, embedding) for k, v in pts.items()}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def line(p1, p2, **kw):
        (x1, y1, z1), (x2, y2, z2) = pts3d[p1], pts3d[p2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], **kw)

    for p1, p2 in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]:
        line(p1, p2, color="tab:blue", lw=1.4)
    for p1, p2 in [("A", "Q"), ("B", "Q"), ("A", "R"), ("B", "R"), ("C", "R"), ("Q", "S"), ("AC", "S")]:
        line(p1, p2, color="tab:green", lw=0.8, ls="--")
    line("D", "P", color="tab:purple", lw=0.9, ls="--")

    for lab, (x, y, z) in pts3d.items():
        ax.scatter([x], [y], [z], color="black", s=22)
        ax.text(x, y, z, f"  {lab}", fontsize=9, fontweight="bold" if len(lab) == 1 else "normal")

    ax.set_title("Figure 4 — Tetrahedron ABCD (D on top): construction of\n"
                 "Q, R, S, P for the Case (b) region alpha_A+alpha_D < alpha_B+alpha_C")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FIGURE 5: Case (b) — outer PQRS and inner shrunk P'Q'R'S'
# ---------------------------------------------------------------------------

def figure5_points() -> Tuple[Dict[str, Point4], Dict[str, Point4], Point4]:
    """Outer tetrahedron P-Q-R-S, inner shrunk tetrahedron P'-Q'-R'-S', and
    their shared centroid G2. Reuses exactly the same construction as
    `voting_paradoxes.geometric_bounds_4('b')`, independently verified
    against the paper's Table 6."""
    P = (F(1, 4), F(1, 4), F(1, 4), F(1, 4))
    Q = (F(1, 2), F(1, 2), F(0), F(0))
    R = (F(1, 3), F(1, 3), F(1, 3), F(0))
    S = (F(1, 2), F(1, 4), F(1, 4), F(0))
    outer = {"P": P, "Q": Q, "R": R, "S": S}

    G2 = centroid(list(outer.values()))
    ratio = 2 / 3  # (384/1296)^(1/3) simplifies exactly to 2/3
    inner_raw = {k: shrink(tuple(float(x) for x in v), tuple(float(x) for x in G2), ratio)
                 for k, v in outer.items()}
    inner = {"P'": inner_raw["P"], "Q'": inner_raw["Q"], "R'": inner_raw["R"], "S'": inner_raw["S"]}
    return outer, inner, G2


def plot_figure5(figsize=(8, 8), elev=15, azim=-60):
    """Reproduce Figure 5: the outer tetrahedron P-Q-R-S (Case b) with the
    smaller, similar, centroid-sharing tetrahedron P'-Q'-R'-S' (the OWNCM
    region behind Table 6) drawn nested inside it."""
    outer, inner, G2 = figure5_points()
    embedding = make_tetra_embedding(["D", "A", "B", "C"])

    all_pts = {**outer, **inner, "G2": G2}
    pts3d = {k: bary4_to_3d(v, embedding) for k, v in all_pts.items()}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def line(p1, p2, **kw):
        (x1, y1, z1), (x2, y2, z2) = pts3d[p1], pts3d[p2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], **kw)

    outer_labels = ["P", "Q", "R", "S"]
    inner_labels = ["P'", "Q'", "R'", "S'"]
    for i in range(4):
        for j in range(i + 1, 4):
            line(outer_labels[i], outer_labels[j], color="tab:blue", lw=1.6)
            line(inner_labels[i], inner_labels[j], color="tab:red", lw=1.8)

    for lab, (x, y, z) in pts3d.items():
        color = "tab:red" if lab in inner_labels else ("tab:purple" if lab == "G2" else "black")
        ax.scatter([x], [y], [z], color=color, s=24)
        ax.text(x, y, z, f"  {lab}", fontsize=9, fontweight="bold", color=color)

    ax.set_title("Figure 5 — Case (b): outer tetrahedron P-Q-R-S (blue)\n"
                 "and inner OWNCM tetrahedron P'-Q'-R'-S' (red), shared centroid G2")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout()
    return fig
