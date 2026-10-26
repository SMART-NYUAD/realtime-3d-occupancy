# Fixing the top strip: camera options (October 2026)

The worst detection in the lab today is along the **top strip**, from (4, 8.5) to
(8.3, 10.4), and especially the top-right corner around (8, 9). The other
priorities are the **work area** (3, 2) to (8, 10), where most people work, and
the **bottom-left corner** (0, 1) to (4, 2). The rest of the room still counts.

The options here were ranked with area weights: a square of floor in the top strip
counts **8 times**, in the work area or the bottom-left corner **2 times**, and
anywhere else once. Every map still shows the whole room; the weighted areas are
outlined in light blue and labelled x8 and x2.

All numbers are the share of each area that is **good** (seen by two cameras whose
lines of sight cross at 25° or more, so height and posture are measured), written
as **seated / standing**. Whole-room results without weights are one folder up, in
[`../README.md`](../README.md).

## Today

| Top strip | Work area | Bottom-left corner | Corner at (8, 9) | Whole room |
|---|---|---|---|---|
| 35 / 54 | 73 / 85 | 2 / 2 | 5 / 10 | 46 / 57 |

Map: `add_one/coverage_current.jpg`

**Why the top right is so bad today:** yi01 cannot see that corner (it is outside
its picture), and yi04 and yi05 see it from 9–10 m away and from nearly the same
direction (26° apart). A seated person there is only about 120–145 pixels tall in
their pictures and is partly hidden by desks and monitors, so their height is
almost never measured and their posture is guessed, often as "standing". Taking a
seated person as standing places them closer to those cameras, which is why the
live system puts people from (8, 9) at around (6, 8).

## The options

| | Option | Diagram | Top strip | Work area | Bottom-left corner | Corner at (8, 9) | Whole room |
|---|---|---|---|---|---|---|---|
| A | Add one camera, nothing moved | `add_one/coverage_option_1.jpg` | 48 / 72 | 82 / 93 | 2 / 2 | 33 / 43 | 58 / 67 |
| **B** | **Move yi01, add one (top strip first)** | **`layout/layout_keep_yi04+yi05.jpg`** | **70 / 93** | **88 / 97** | **2 / 0** | **52 / 81** | **64 / 70** |
| C | Move yi01, add one (balanced with the bottom-left corner) | `balanced/layout_keep_yi04+yi05.jpg` | 54 / 70 | 83 / 93 | 60 / 52 | 43 / 52 | 67 / 73 |
| D | Move yi01 and yi05, add one | `layout/layout_keep_yi04.jpg` | 72 / 91 | 88 / 98 | 22 / 22 | 52 / 76 | 71 / 77 |
| E | Move all four | `layout/layout_keep_none.jpg` | 67 / 96 | 87 / 98 | 65 / 65 | 57 / 86 | 75 / 82 |

What the table shows:

- **One added camera (A)** helps the top strip, but only partly.
- **Moving yi01 as well (B)** roughly doubles the gain on the top strip and the
  (8, 9) corner. A new calibration walk is needed for the fourth camera anyway, so
  moving yi01 at the same time costs only the remounting.
- **With two changes, the top strip and the bottom-left corner cannot both be
  fixed**: B fixes the top strip, C fixes the bottom-left corner and half-fixes the
  top strip.
- **Only moving all four cameras (E)** fixes both, and gives the best coverage of
  the whole room.

## Recommendation: option B

It fixes the area that performs worst today with the least work: yi04 and yi05
stay, yi01 moves down the left wall, and the new camera goes on the right wall.

| Camera | Mount at (x, y), 2.5 m high | Face towards | Tilt down | Picture centre on the floor |
|---|---|---|---|---|
| yi04, yi05 | stay where they are | | | |
| yi01, moved (A on the map) | (0.44, 8.28), left wall | (8.28, 3.19) | 20° | (6.20, 4.54) |
| New camera (B on the map) | (7.74, 3.48), right wall | (2.72, 10.39) | 30° | (5.19, 6.98) |

![Option B: move yi01, add one, top strip first](layout/layout_keep_yi04+yi05.jpg)

"Face towards" is where the camera's facing direction meets the far side of the
room, seen from above. To aim, have someone stand at the "picture centre" point and
centre them in the camera's live view.

If the bottom-left corner turns out to matter as much, the step after B is option
E (move all four); option C trades most of B's top-strip gain for that corner.

## Placements for the other options

**A: add one camera, nothing moved** (`add_one/coverage_option_1.jpg`): left wall
at (0.44, 7.78), facing towards (8.28, 5.23), tilted 30° down (picture centre on the
floor at (4.56, 6.44)). Option 5 in the same
folder, on the top wall at (3.34, 9.98) facing towards (8.28, 5.53) at 32°, is about
as good on the top strip (54 / 72) but helps the rest of the room less.

**C: move yi01, add one, balanced** (`balanced/layout_keep_yi04+yi05.jpg`; ranked
with the bottom-left corner weighted 4 times instead of 2):

| Camera | Mount at (x, y), 2.5 m high | Face towards | Tilt down | Picture centre on the floor |
|---|---|---|---|---|
| yi01, moved slightly and re-aimed (B on the map) | (1.24, 9.48) | (8.28, 1.66) | 18° | (6.39, 3.76) |
| New camera (A on the map) | (7.74, 3.48), right wall | (0.00, 5.99) | 22° | (1.86, 5.39) |

**D: move yi01 and yi05, add one** (`layout/layout_keep_yi04.jpg`): A at (0.44, 7.78)
facing towards (8.28, 3.25) at 15°; B at (7.54, 4.48) facing towards (3.70, 10.39)
at 30°; C at (7.94, 8.98) facing towards (0.00, 4.39) at 26°.

**E: move all four** (`layout/layout_keep_none.jpg`): A at (3.34, 9.98) facing
towards (6.58, 0.00) at 30°; B at (7.94, 7.48) facing towards (0.46, 0.00) at 22°;
C at (7.34, 3.08) facing towards (2.59, 10.39) at 24°; D at (1.04, 4.18), beside
tall furniture, facing towards (7.25, 10.39) at 16°.

On the layout maps, white cameras stay where they are, magenta ones are new
positions (any of the Yi cameras can go to any of them; each keeps its own lens
file), and an outlined dot marks where a camera that moves is today. All
positions, directions and tilts are also in the `layout_plan.json` and
`coverage_plan.json` files.

## Limits

- The (8, 9) corner stays the weakest spot with every option (at best about 57%
  good for a seated person): the monitors there hide seated people's shoulders from
  almost every direction. Rearranging them would help, and so could a tracker
  change that is less ready to call a small, far-away person "standing". A short
  recording of someone seated at (8, 9) would show which part goes wrong.
- The scan is a snapshot. Option B's two positions are on walls, but some others
  are beside tall furniture (D in option E). Check every spot with the live camera
  view before drilling.

## Regenerate

```bash
W="--weight 4,8.5,8.3,10.4,8 --weight 3,2,8,10,2"
python tools/coverage_planner.py $W --weight 0,1,4,2,2 --image-format jpg --out docs/coverage/priority/add_one
python tools/layout_planner.py --variants $W --weight 0,1,4,2,2 --image-format jpg --out docs/coverage/priority/layout
python tools/layout_planner.py --keep yi04 --keep yi05 $W --weight 0,1,4,2,4 --image-format jpg --out docs/coverage/priority/balanced
```

(In zsh, write the `--weight` options out instead of using `$W`.) The per-area
numbers score each rectangle on its own with `--focus x1,y1,x2,y2`.
