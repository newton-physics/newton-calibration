# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Read-only GL comparison and portable USDA point animation."""

import json
from pathlib import Path

import numpy as np
import torch


class Play:
    def compare(self, baseline, calibrated, references=None, *, accepted=False):
        before = {
            (e.experiment_id, e.repeat): e.trace
            for e in baseline
            if e.status == "completed" and e.trace is not None
        }
        after = {
            (e.experiment_id, e.repeat): e.trace
            for e in calibrated
            if e.status == "completed" and e.trace is not None
        }
        tracks = []
        for key in sorted(before.keys() & after.keys()):
            left, right = before[key], after[key]
            if (
                left.positions is None
                or right.positions is None
                or not torch.equal(left.times, right.times)
            ):
                continue
            for label, trace in (("baseline", left), ("calibrated", right)):
                positions = torch.as_tensor(trace.positions, dtype=torch.float64).cpu()
                if (
                    positions.ndim != 3
                    or positions.shape[0] != len(trace.times)
                    or positions.shape[-1] != 3
                    or not torch.isfinite(positions).all()
                ):
                    raise ValueError("Play requires finite [time, point, xyz] positions")
                tracks.append(
                    {
                        "experiment": key[0],
                        "repeat": key[1],
                        "label": label,
                        "times": trace.times.cpu().clone(),
                        "positions": positions.clone(),
                    }
                )
        return {"accepted": accepted, "tracks": tracks}

    def export_usd(self, comparison, output_path):
        tracks = comparison["tracks"]
        if not tracks:
            raise ValueError("no comparable position traces for USD export")
        path = Path(output_path)
        start = min(float(x["times"][0]) for x in tracks)
        end = max(float(x["times"][-1]) for x in tracks)
        # One timeCode equals one physical second; playback display rate is independent.
        lines = [
            "#usda 1.0",
            "(",
            '    defaultPrim = "Comparison"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            "    timeCodesPerSecond = 1",
            f"    startTimeCode = {start:.12g}",
            f"    endTimeCode = {end:.12g}",
            ")",
            'def Xform "Comparison" {',
            f"    custom bool accepted = {str(comparison['accepted']).lower()}",
        ]
        provenance = {
            key: comparison[key]
            for key in ("identity", "baseline_parameters", "calibrated_parameters")
            if key in comparison
        }
        lines.append(
            "    custom string provenance = " + json.dumps(json.dumps(provenance, sort_keys=True))
        )
        for i, track in enumerate(tracks):
            color = "(0.25, 0.65, 1)" if track["label"] == "baseline" else "(1, 0.5, 0.15)"
            lines.extend(
                [
                    f'    def Points "{track["label"]}_{i}" {{',
                    "        custom string experiment = " + json.dumps(track["experiment"]),
                    f"        custom int repeat = {track['repeat']}",
                    f"        color3f[] primvars:displayColor = [{color}]",
                    '        float[] widths = [0.015] (interpolation = "constant")',
                    "        point3f[] points.timeSamples = {",
                ]
            )
            for time, points in zip(track["times"], track["positions"], strict=True):
                coordinates = ", ".join(
                    "(" + ", ".join(f"{float(v):.12g}" for v in p) + ")" for p in points
                )
                lines.append(f"            {float(time):.12g}: [{coordinates}],")
            lines.extend(["        }", "    }"])
        lines.append("}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def show_gl(self, comparison, *, max_frames=None, screenshot=None, headless=False):
        """Project xyz onto xy. Rendering never advances the simulation."""
        import pyglet

        pyglet.options["headless"] = headless
        from pyglet import shapes

        tracks = comparison["tracks"]
        if not tracks:
            raise ValueError("no tracks to display")
        window = pyglet.window.Window(
            960, 640, caption="Tuning: baseline / calibrated", visible=not headless
        )
        groups = sorted({(x["experiment"], x["repeat"]) for x in tracks})
        coordinates = np.concatenate([x["positions"].numpy().reshape(-1, 3) for x in tracks])
        low, high = coordinates[:, :2].min(0), coordinates[:, :2].max(0)
        span = np.maximum(high - low, 0.1)
        start = min(float(x["times"][0]) for x in tracks)
        duration = max(float(x["times"][-1]) for x in tracks) - start
        row_height = 520 / len(groups)
        frame = 0
        elapsed = 0.0

        @window.event
        def on_draw():
            window.clear()
            batch = pyglet.graphics.Batch()
            objects = []
            for track in tracks:
                times = track["times"].numpy()
                index = min(
                    np.searchsorted(times, start + elapsed, side="right") - 1, len(times) - 1
                )
                index = max(0, index)
                points = track["positions"][index].numpy()
                color = (65, 165, 255) if track["label"] == "baseline" else (255, 128, 38)
                row = groups.index((track["experiment"], track["repeat"]))
                offset = np.array([60, 60 + row * row_height])
                size = np.array([840, max(row_height - 45, 10)])
                xy = offset + (points[:, :2] - (low + high) / 2) / span * size + size / 2
                if points.shape[0] == 1:
                    trail = track["positions"][: index + 1, 0, :2].numpy()
                    trail = offset + (trail - (low + high) / 2) / span * size + size / 2
                    for a, b in zip(trail[:-1], trail[1:]):
                        objects.append(shapes.Line(*a, *b, thickness=1, color=color, batch=batch))
                if track["label"] == "baseline":
                    objects.append(
                        pyglet.text.Label(
                            f"{track['experiment']} / repeat {track['repeat']}",
                            x=60,
                            y=60 + (row + 1) * row_height - 15,
                            batch=batch,
                        )
                    )
                for point in xy:
                    objects.append(shapes.Circle(*point, radius=5, color=color, batch=batch))
                for a, b in zip(xy[:-1], xy[1:]):
                    objects.append(shapes.Line(*a, *b, thickness=2, color=color, batch=batch))
            objects.append(
                pyglet.text.Label(
                    "Blue: baseline    Orange: calibrated    "
                    + ("validated" if comparison["accepted"] else "diagnostic"),
                    x=20,
                    y=615,
                    batch=batch,
                )
            )
            batch.draw()

        def update(dt):
            nonlocal frame, elapsed
            frame += 1
            elapsed = (elapsed + dt) % duration if duration > 0 else 0.0
            if max_frames is not None and frame >= max_frames:
                if screenshot:
                    window.switch_to()
                    on_draw()
                    pyglet.image.get_buffer_manager().get_color_buffer().save(str(screenshot))
                window.close()
                pyglet.app.exit()

        pyglet.clock.schedule_interval(update, 1 / 30)
        try:
            pyglet.app.run()
        finally:
            pyglet.clock.unschedule(update)
            window.close()
