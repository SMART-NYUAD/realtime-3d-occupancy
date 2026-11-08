# How many cameras does the lab need? (November 2026)

The planner found the best layout for 2, 3, 4 and 5 cameras, using only the mounting
places marked with `tools/mount_editor.py`. These are **predictions** from the lab
scan. The study will install each layout, calibrate it and measure how well people
are actually located, so these numbers are what the measurements will be compared
against.

![Predicted good floor by number of cameras](coverage_by_camera_count.png)

## What was run

- **Mounting places:** the 13 lines in `data/mount_lines.json`, each at its own height
  (1.9 to 2.6 m), picked from the scan's side view. Every camera is placed **exactly on
  a line**, at points every 0.5 m along it (74 places in all). The pillar faces (L7 to
  L9) are included. No camera is kept where it is today.
- **Which way a camera can face:** each line faces the side with more open space at its
  height, which is into the room; the search tries directions up to 75° either side of
  that, and tilts from 15° down.
- **Priority:** the work area (2, 2) to (8, 10) and the bottom-left corner (0, 0.5) to
  (4, 2) count twice; the rest of the room counts once.
- **Search:** for each number of cameras, 10 independent runs (seeds 0 to 9), each
  trying 60 randomised starts. The best run is kept.
- **Rules (planner defaults):** a spot is **good** when two cameras see the person's
  head and shoulders and their lines of sight cross at 25° or more; cameras see up to
  9 m; every camera shares at least 2 m² of floor with another (needed for the
  calibration walk); cameras are at least 1 m apart. The first 0.3 m in front of a
  camera is not checked for blocking, since that is the wall or cabinet it is fixed to.
  Seated and standing people are both scored.

## Results

Share of floor that is **good**, written as **seated / standing** (%).

| Cameras | Whole room | Work area (2, 2)-(8, 10) | Bottom-left corner (0, 0.5)-(4, 2) | Unseen, whole room | Gain in good floor |
|---|---|---|---|---|---|
| 2 | 50 / 55 | 66 / 73 | 21 / 24 | 30 / 26 | n/a |
| 3 | 58 / 71 | 79 / 95 | 52 / 52 | 18 / 14 | +8 / +15 |
| 4 | 72 / 82 | 82 / 96 | 81 / 83 | 14 / 10 | +14 / +11 |
| 5 | 79 / 88 | 88 / 98 | 90 / 98 | 10 / 7 | +7 / +7 |

What this shows:

- **Good floor rises with every camera, but each one adds less.** In the whole room it
  goes from 50 / 55 with 2 cameras to 79 / 88 with 5. The 5th camera adds 7 / 7 points.
- **For seated people the 4th camera matters most** (+14 points), more than the 3rd (+8).
  For standing people the 3rd camera matters most (+15).
- **Four cameras cover the work area well for standing people (96%)**, and 82% for seated
  people. A fifth raises these to 98% and 88%.
- **The bottom-left corner needs at least 4 cameras** to be mostly good (81 / 83); with 5
  it reaches 90 / 98.
- **Some floor stays unseen even with 5 cameras** (10 / 7%), mostly the bottom-right
  area (x 5 to 8, y 0 to 1.5) and strips along the walls. The bottom-right area is
  outside both priority zones, so the planner does not spend cameras on it.
- **Seated people are always harder than standing ones**, because desks and monitors
  hide them.

## How much the 10 runs agreed

| Cameras | Runs within 0.5% of the best | Score range | Largest camera move between runs |
|---|---|---|---|
| 2 | 10 of 10 | 80.1 to 80.1 | 0.00 m |
| 3 | 10 of 10 | 98.5 to 98.5 | 0.00 m |
| 4 | 10 of 10 | 109.2 to 109.5 | 4.88 m |
| 5 | 5 of 10 | 115.0 to 116.1 | 4.58 m |

- **2 and 3 cameras:** every run found the same layout.
- **4 cameras:** every run scored within 0.3% of the best, but they did not all pick the
  same layout; one camera can move up to 4.9 m between layouts that score almost the
  same. Several 4-camera layouts are about equally good, so the choice between them can
  be made on practical grounds (cabling, ease of mounting) without losing coverage.
- **5 cameras:** 5 of the 10 runs found the best layout; the others scored up to 1%
  lower. The search is less settled here, so a longer search might find a slightly
  better 5-camera layout.

## Where to mount each layout

Each camera goes **on the given line, at the given position and height**. Turn it to
face the **face towards** point and tilt it down by the given angle. The layouts are not
built from each other: the best 4-camera layout does not contain the best 3-camera
layout, so every camera count needs its own mounting and calibration walk.

**2 cameras** (best run: seed 0; map: [`2_cameras/layout_keep_none.jpg`](2_cameras/layout_keep_none.jpg))

| Camera | Mount on line | Position x, y (m) | Height (m) | Face towards x, y | Tilt down |
|---|---|---|---|---|---|
| A | L2 | 8.09, 8.32 | 2.52 | 0.00, 3.65 | 30° |
| B | L1 | 3.95, 10.39 | 2.60 | 3.95, 0.00 | 30° |

**3 cameras** (best run: seed 0; map: [`3_cameras/layout_keep_none.jpg`](3_cameras/layout_keep_none.jpg))

| Camera | Mount on line | Position x, y (m) | Height (m) | Face towards x, y | Tilt down |
|---|---|---|---|---|---|
| A | L2 | 8.09, 7.28 | 2.52 | 0.00, 1.41 | 28° |
| B | L8 | 0.82, 4.86 | 2.48 | 7.65, 10.39 | 30° |
| C | L11 | 0.81, 9.58 | 2.56 | 8.28, 2.31 | 30° |

**4 cameras** (best run: seed 9; map: [`4_cameras/layout_keep_none.jpg`](4_cameras/layout_keep_none.jpg))

| Camera | Mount on line | Position x, y (m) | Height (m) | Face towards x, y | Tilt down |
|---|---|---|---|---|---|
| A | L2 | 8.09, 7.28 | 2.52 | 0.00, 0.00 | 28° |
| B | L4 | 4.19, 0.44 | 2.44 | 0.00, 7.70 | 30° |
| C | L10 | 0.06, 6.45 | 2.56 | 8.28, 3.76 | 28° |
| D | L3 | 7.87, 3.17 | 2.43 | 2.62, 10.39 | 30° |

**5 cameras** (best run: seed 3; map: [`5_cameras/layout_keep_none.jpg`](5_cameras/layout_keep_none.jpg))

| Camera | Mount on line | Position x, y (m) | Height (m) | Face towards x, y | Tilt down |
|---|---|---|---|---|---|
| A | L2 | 8.09, 5.21 | 2.52 | 0.00, 7.84 | 30° |
| B | L4 | 4.19, 0.44 | 2.44 | 0.00, 5.61 | 30° |
| C | L8 | 0.82, 4.07 | 2.48 | 8.28, 4.07 | 20° |
| D | L6 | 0.00, 0.48 | 2.54 | 8.28, 6.47 | 32° |
| E | L10 | 0.07, 8.02 | 2.56 | 8.28, 8.86 | 30° |

On the maps, each floor square shows the worse of seated and standing. Magenta dots
are the planned cameras and white circles are where the cameras are today (for
reference only). `layout_plan.json` in each folder has the full numbers, every run's
layout and score, and the settings.

## Limits

- These are predictions from the scan. The 25° crossing angle and 9 m range are
  assumptions that the study is testing, not measured values.
- The scan is a snapshot. Objects added since it was taken, such as the TV in front of
  yi05, are not in it yet.
- The lab has four cameras, so the 5-camera layout needs one more before it can be
  tested.
- The numbers are not compared with today's cameras: yi04's temporary calibration puts
  it inside the pillar, which would make today's coverage look worse than it is.

## Reproduce

```bash
for n in 2 3 4 5; do
  python tools/layout_planner.py --mount-lines data/mount_lines.json \
    --weight 2,2,8,10,2 --weight 0,0.5,4,2,2 --cameras $n --runs 10 --restarts 60 \
    --image-format jpg --out docs/coverage/camera_count/${n}_cameras
done
python docs/coverage/camera_count/summary.py   # chart, summary.json and the tables above
```

On a Mac the four searches took 57 s, 102 s, 157 s and 264 s.
