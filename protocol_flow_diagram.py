#!/usr/bin/env python3
"""Protocol-flow diagram for the camera-ready paper, rendered with manim.

Replaces the hand-drawn fig9_protocol_flow export with a script-generated
figure whose labels match the corrected fast-path mechanism: Phase 1 carries
a conditional BLS commitment, the fast path triggers on >= 2N/3 matching
commitments, and the fast-path certificate is the same object Phase 2
produces.

Render a still (writes media/images/.../fig_protocol_flow.png):

    manim -s --format png -r 1660,1280 --output_file fig_protocol_flow \
        protocol_flow_diagram.py ProtocolFlow
"""

from manim import (
    Scene, VGroup, Text, RoundedRectangle, DashedVMobject, Arrow,
    DashedLine, WHITE, DOWN, LEFT, RIGHT, UP, config,
)

config.background_color = WHITE

FONT = "DejaVu Sans"

# Palette (kept close to the original drawing)
P1_FILL, P1_STROKE, P1_TITLE, P1_BODY = "#EFECFB", "#6C5CE7", "#3D3580", "#5B4FC4"
P2_FILL, P2_STROKE, P2_TITLE, P2_BODY = "#FBECEF", "#B0486A", "#8E2F4F", "#B0486A"
OK_FILL, OK_STROKE, OK_TITLE, OK_BODY = "#E7F6EF", "#1E8E63", "#116B49", "#1E8E63"
DARK, GRAY = "#2B2B2B", "#6B6B6B"
DEC = "#4A3FB5"


def box(title, lines, fill, stroke, title_color, body_color, width, height):
    rect = RoundedRectangle(corner_radius=0.08, width=width, height=height,
                            fill_color=fill, fill_opacity=1.0,
                            stroke_color=stroke, stroke_width=2.2)
    rows = [Text(title, font=FONT, weight="BOLD", font_size=17,
                 color=title_color)]
    for ln in lines:
        rows.append(Text(ln, font=FONT, font_size=13.5, color=body_color))
    content = VGroup(*rows).arrange(DOWN, buff=0.07)
    if content.width > width - 0.24:
        content.scale_to_fit_width(width - 0.24)
    if content.height > height - 0.18:
        content.scale_to_fit_height(height - 0.18)
    content.move_to(rect.get_center())
    return VGroup(rect, content)


def lab(text, color, max_w=None, size=12.5, bold=False, bg=True):
    t = Text(text, font=FONT, font_size=size, color=color,
             weight="BOLD" if bold else "NORMAL")
    if max_w is not None and t.width > max_w:
        t.scale_to_fit_width(max_w)
    if bg:
        t.add_background_rectangle(color=WHITE, opacity=1.0, buff=0.03)
    return t


class ProtocolFlow(Scene):
    def construct(self):
        # ---- column headers -------------------------------------------
        for x, name, sub in [(-3.7, "Validators", "N nodes, up to N/3 byzantine"),
                             (0.3, "Aggregator", "rotating leader role"),
                             (3.85, "State", "finality outcome")]:
            h = Text(name, font=FONT, weight="BOLD", font_size=21, color=DARK)
            s = Text(sub, font=FONT, font_size=13, color=GRAY)
            self.add(VGroup(h, s).arrange(DOWN, buff=0.07).move_to([x, 3.55, 0]))

        # ---- phase 1 container ------------------------------------------
        p1_rect = RoundedRectangle(corner_radius=0.12, width=10.2, height=3.6,
                                   stroke_color=P1_STROKE, stroke_width=1.5,
                                   fill_opacity=0.0).move_to([0, 1.25, 0])
        self.add(DashedVMobject(p1_rect, num_dashes=95))
        p1_t = Text("Phase 1: distance filtering + speculative commitments",
                    font=FONT, weight="BOLD", font_size=16, color=P1_TITLE)
        p1_t.move_to(p1_rect.get_corner(UP + LEFT), aligned_edge=UP + LEFT)
        p1_t.shift(RIGHT * 0.28 + DOWN * 0.14)
        self.add(p1_t)

        # ---- phase 1 boxes ----------------------------------------------
        v1 = box("Send digest + bloom", ["64 B digest, 25 B bloom filter",
                                         "+ 96 B BLS commitment",
                                         "if state matches proposal"],
                 P1_FILL, P1_STROKE, P1_TITLE, P1_BODY, 2.9, 1.35)
        v1.move_to([-3.7, 1.85, 0])

        a1 = box("Cluster on distance", ["‖D_i − D_ref‖ ≤ τ",
                                         "excludes byzantine",
                                         "flags stragglers"],
                 P1_FILL, P1_STROKE, P1_TITLE, P1_BODY, 2.6, 1.35)
        a1.move_to([0.3, 1.85, 0])

        v2 = box("Receive missing txs", ["aligned validators",
                                         "sit idle this step"],
                 P1_FILL, P1_STROKE, P1_TITLE, P1_BODY, 2.5, 0.95)
        v2.move_to([-3.7, 0.35, 0])

        a2 = box("decision", ["≥ 2N/3 matching", "commitments collected?"],
                 WHITE, DEC, DARK, DEC, 2.3, 0.95)
        a2.move_to([0.3, 0.25, 0])

        s1 = box("Finalize in one round trip",
                 ["aggregate Phase 1 commitments:",
                  "96 B aggregate + N/8 B bitmap"],
                 OK_FILL, OK_STROKE, OK_TITLE, OK_BODY, 2.9, 1.0)
        s1.move_to([3.85, 0.25, 0])

        # arrows, phase 1
        ar1 = Arrow(v1[0].get_right(), a1[0].get_left(), buff=0.05,
                    stroke_width=2.2, max_tip_length_to_length_ratio=0.12,
                    color=DEC)
        self.add(ar1)
        self.add(lab("N parallel sends", DEC, max_w=1.15)
                 .move_to([-1.65, 2.12, 0]))

        back = DashedLine([-1.0, 1.45, 0], [-2.3, 1.15, 0],
                          stroke_width=1.6, color=DEC)
        back.add_tip(tip_length=0.11, tip_width=0.11)
        self.add(back)
        self.add(lab("bloom diff, only to stragglers", DEC, max_w=1.9)
                 .move_to([-1.6, 0.95, 0]))

        yes = Arrow(a2[0].get_right(), s1[0].get_left(), buff=0.05,
                    stroke_width=2.4, max_tip_length_to_length_ratio=0.14,
                    color=OK_STROKE)
        self.add(yes)
        self.add(lab("yes, fast path", OK_STROKE, max_w=0.85, bold=True)
                 .move_to([1.95, 0.55, 0]))

        down = Arrow(a2[0].get_bottom(), [a2[0].get_bottom()[0], -0.85, 0],
                     buff=0.05, stroke_width=2.2,
                     max_tip_length_to_length_ratio=0.18, color=DEC)
        self.add(down)
        self.add(lab("no, proceed to Phase 2", DEC, max_w=1.7)
                 .move_to([1.5, -0.72, 0]))

        self.add(v1, a1, v2, a2, s1)

        # ---- phase 2 container ------------------------------------------
        p2_rect = RoundedRectangle(corner_radius=0.12, width=10.2, height=3.05,
                                   stroke_color=P2_STROKE, stroke_width=1.5,
                                   fill_opacity=0.0).move_to([0, -2.45, 0])
        self.add(DashedVMobject(p2_rect, num_dashes=95))
        p2_t = Text("Phase 2: hash commitment (fallback: full collection)",
                    font=FONT, weight="BOLD", font_size=16, color=P2_TITLE)
        p2_t.move_to(p2_rect.get_corner(UP + LEFT), aligned_edge=UP + LEFT)
        p2_t.shift(RIGHT * 0.28 + DOWN * 0.14)
        self.add(p2_t)

        # ---- phase 2 boxes ----------------------------------------------
        v3 = box("Sign block hash", ["96 B BLS sig", "on SHA-256(B)"],
                 P2_FILL, P2_STROKE, P2_TITLE, P2_BODY, 2.5, 0.95)
        v3.move_to([-3.7, -1.95, 0])

        a3 = box("Aggregate signatures", ["96 B BLS aggregate",
                                          "N/8 B signer bitmap",
                                          "verify ≥ 2N/3 signed"],
                 P2_FILL, P2_STROKE, P2_TITLE, P2_BODY, 2.6, 1.3)
        a3.move_to([0.3, -2.0, 0])

        s2 = box("Block finalized", ["≥ 2N/3 signatures verified",
                                     "same certificate as fast path"],
                 OK_FILL, OK_STROKE, OK_TITLE, OK_BODY, 2.9, 1.0)
        s2.move_to([3.85, -1.95, 0])

        v4 = box("Verify and accept", ["check aggregate sig", "apply block"],
                 P2_FILL, P2_STROKE, P2_TITLE, P2_BODY, 2.5, 0.95)
        v4.move_to([-3.7, -3.35, 0])

        ar3 = Arrow(v3[0].get_right(), a3[0].get_left(), buff=0.05,
                    stroke_width=2.2, max_tip_length_to_length_ratio=0.12,
                    color=P2_STROKE)
        self.add(ar3)
        self.add(lab("cluster members only", P2_STROKE, max_w=1.2)
                 .move_to([-1.72, -1.7, 0]))

        ar4 = Arrow(a3[0].get_right(), s2[0].get_left(), buff=0.05,
                    stroke_width=2.4, max_tip_length_to_length_ratio=0.14,
                    color=OK_STROKE)
        self.add(ar4)

        proof = Arrow(a3[0].get_corner(DOWN + LEFT), v4[0].get_right(),
                      buff=0.08, stroke_width=2.0,
                      max_tip_length_to_length_ratio=0.08, color=P2_STROKE)
        self.add(proof)
        self.add(lab("finality proof: aggregate + bitmap", P2_STROKE,
                     max_w=1.95).move_to([-1.05, -3.05, 0]))

        local = DashedLine(s2[0].get_bottom(), [-2.35, -3.5, 0],
                           stroke_width=1.5, color=OK_STROKE)
        local.add_tip(tip_length=0.11, tip_width=0.11)
        self.add(local)
        self.add(lab("block applied locally, no ACK required", OK_STROKE,
                     max_w=2.4).move_to([1.7, -3.62, 0]))

        self.add(v3, a3, s2, v4)
