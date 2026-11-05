# Reflection plane finder: repeatability protocol

Tests the homodyne finder in the human interface GUI (`scan_managers/homodyne_plane_finder.py`). The acceptance numbers are proposals; fix them before collecting data.

## Baseline (replay on recorded slews, 400 µm/s, per-slew)

| Sample | Slews | Missed | >15 µm off | SD (µm) |
|---|---|---|---|---|
| Flat cuvette | 100 | 0 | 0 | 1.6 |
| 1 in ball | 90 | 3 | 7 | 4.5 |
| Cornea phantom, R 7.12 mm, to 23° | 130 | 3 (all R ≥ 2.5 mm) | 1 | 3.8 |

Not yet known: behaviour at any speed other than 400 µm/s, behaviour on a moving surface, and the offset between the homodyne plane and the Brillouin focus. The finder has never run live on hardware. The replay used 0.3 s of stationary pre-roll; the live finder uses 0.05 s.

## Method common to all blocks

Use the Sweep Scan with `approach_um` 300 and `target_depth_um` 50. One cycle is an in-crossing (forward), a frame, and an out-crossing (backward). The saved scan keeps both channels' raw voltages, the Zaber log and the settings for every crossing, so results can be recomputed offline. Record the LO level at the start of each block (expected about 6.2 V).

A cycle takes about 3.3 s at 400 µm/s, with the crossings about 1.7 s apart (dummy run; confirm on the rig).

## Phantom

| Block | Purpose | Design | Accept |
|---|---|---|---|
| P0 | First live check | 20 cycles at the apex | Plane SD and detection match the baseline; lens overshoot about 150 µm; no missed stop |
| P1 | Choose the speed | Apex and R = 2.25 mm; 400, 800, 1200 µm/s; 20 cycles each | Fastest speed with ≥ 99 % detection and SD ≤ 5 µm |
| P2 | Limit versus tilt | R = 0 to 2.75 mm in 0.25 mm steps, one meridian, 20 cycles each | ≥ 99 % detection to 2.25 mm, ≥ 85 % beyond; no plane > 50 µm off |
| P3 | Start distance | Apex; approach 150, 300, 600 µm; 20 cycles each | Same SD and detection for all three |
| P4 | Re-seating | Remove and re-seat 5 times; eye tracker Move XY and Move Z; 10 cycles at R = 0 and R = 2 mm | SD of the sag z(2 mm) − z(0) across seats ≤ 10 µm |
| P5 | Moving surface | Drive stage Z at ±25 µm, 1 Hz; 20 cycles at the apex; compare the plane with the logged stage position | Error SD ≤ 10 µm (needs a small script) |
| P6 | Focus offset | Axial step scan through the surface; find the Brillouin onset relative to the plane | Sets `z_offset_um`; the current 200 µm was tuned for the old detector |

Run P1 first after P0; use the chosen speed for P2 to P6.

**Combined phantom run (replaces P0, P2 and P6):** `guis/plane_accuracy_test/main.py`, a standalone program; close the human interface GUI first. Set center and edge once, then it runs randomized blocks in which every radius gets one forward find, one backward find and one Andor axial scan. The order comes from a saved seed, so drift is spread evenly over radii and kinds. Precision is the spread of the finds. Accuracy is the finds against the surface position in the axial scans. Every trial is saved raw as it finishes, misses included.

**Sweep gate check:** in P2, count crossings rejected by `min_peak_fraction` (now 0.15). Any rejection of a crossing within 15 µm of its neighbours means the gate is still too tight.

## In vivo

**Do not start until:** the approved protocol covers this finder, and exposure is recalculated. A single find can now dwell up to 3.75 s (1500 µm at 400 µm/s) versus 1.25 s for the old settings, and a sweep is longer. Calculate cumulative exposure per session from the measured power at the eye and compare it with the applicable limit.

- **Subjects:** pilot of 5 subjects, one eye each, two sessions on different days.
- **Positions:** apex and R = 1.5 mm and 2.5 mm on the temporal meridian. On a cornea, 3 mm is about 23°, which is the phantom's limit.
- **Sequence at each position:** eye tracker Move XY, Move Z, sweep of 12 cycles. Repeat 3 times with a blink and realignment between repeats, in randomized position order.
- **Speed:** from P1.
- **Stop a repeat** on loss of fixation; log it and do not discard it from the detection rate.

### Separating the finder from the eye

The eye moves, so the raw SD of z mixes finder noise with real motion. Report these separately:

| Quantity | Definition | Meaning |
|---|---|---|
| Detection rate | Found and accepted crossings, per position | Does it work |
| Wrong surface | The frame at plane + 50 µm has no corneal Brillouin peak | Independent check that the plane is the cornea |
| Direction offset | Mean of z_in − z_out | Timing bias; about +3 to +5 µm so far, should stay constant |
| Short-term σ | SD of (z_in − z_out) / √2 | Finder noise plus about 1.7 s of eye motion |
| Motion spectrum | Allan deviation of z_in against lag | Shows where eye motion takes over |
| Within-position Sw | Pooled SD across the 3 repeats | Realignment plus finder; repeatability coefficient = 2.77 Sw |
| Between-session | Bland-Altman of session means | Day-to-day reproducibility |

Excess over the finder = √(σ_vivo² − σ_phantom²), using P5 for σ_phantom.

**Accept:** ≥ 90 % detection at the apex and R = 1.5 mm; no wrong-surface frames; short-term σ ≤ 10 µm. With 5 subjects, report confidence intervals by subject, because cycles within a subject are correlated.

**Also check** the fraction of sweeps failing `motion_limit_um`. The crossings are now further apart in time than at 2000 µm/s, so more eye motion falls inside that gate.
