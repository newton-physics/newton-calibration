# Driven cable

A cable clamped to a gripper that follows a recorded TCP trajectory. The case shows the
driven clamp: before every solver step, a kernel moves the kinematic clamp capsule to the
recorded TCP pose at the current simulation time, composed with the attachment transform.
The drive is in `../driven_attachment.py`, so other driven assets can use it.

A robot moved a real 0.5 m cable in front of one camera (`configs/orange_cable.json`).
Each observation is a cable mask in the camera image. The fit projects the simulated free
cable into the camera and minimizes the chamfer score (`chamfer.py`) against the mask.

The fit searches the bending rigidity and the rest shape: one (x, y) bend pair per joint,
`rest_x_k` and `rest_y_k`. Each is the arc displacement [m] that the bend gives at the
cable end, that is, the bend angle times the cable length beyond joint k. The other
material parameters are fixed.

## Recordings

The recordings are not in this repository. A recording is a directory with a
`recording.json`: the observation times, the TCP drive, the attachment of the cable to the
TCP, the camera and one mask per observation. `source` holds where the data came from.
`data.py` describes the fields.

The recordings: `tilt` (the gripper tilts about 54 degrees in the image plane) and `yaw`
(the gripper turns about the vertical) for training; `validation` and `validation2` (the
same tilt with the gripper turned half a turn, so the cable's curvature faces the other way)
for validation. The frames are the camera's rectified color images, cropped; the masks were
made with a video segmentation model. Each observation keeps its own camera time stamp.

## Files

| Path | Purpose |
|---|---|
| `backend.py` | `CableDrivenAsset`, `CableDrivenBatchEnvironment` (one Newton rod per candidate, clamped at the driven capsule) |
| `chamfer.py` | Camera projection and the chamfer score of a projected cable against a mask |
| `../driven_attachment.py` | Shared by the driven examples: drive kernel, settle phase, stepping to the observation times and `run()` |
| `data.py` | Reads recordings into calibration experiments and experiment settings |
| `evaluator.py` | Squared error with residual features in channel scales |
| `run.py` | Builds the Tuning from a configuration |

## Clamp

`discretization.clamp` sets where the kinematic capsule is. With `segment`, the capsule
that holds the grasp point is kinematic. With `stub`, the model is the free cable beyond
the grasp point, and one more capsule inside the gripper ends at the grasp point and is
kinematic. The real cable uses `stub`: the gripper holds its first 0.02 m.

## Units

Stiffness and damping parameters are per length, independent of the discretization: bending
rigidity EI [N m²], stretch rigidity EA [N], shear rigidity GA [N], twist rigidity GJ [N m²],
and dampings [N m² s]. The backend divides by the segment length to get the per-joint gains
of the rod joints, and reads them back the same way. The chamfer score is in cable lengths
on screen: the mean pixel distance divided by the mask's length along its principal axis.

## Run

```bash
python -m examples.cable_driven.run --recordings PATH         # construct only
python -m examples.cable_driven.run --recordings PATH --run   # calibrate (GPU)
```

Each CMA-ES generation has 24 candidates. On one GPU a search runs up to 150 generations
of about 70 s each. The full run, with qualification and validation, takes about 3.5 hours.
