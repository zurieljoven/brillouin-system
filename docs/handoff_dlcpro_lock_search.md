# Handoff: automated search for the Rb lock point on the Toptica DLC pro (780 nm)

Read this first, then help the user run an automated current x piezo sweep on the lab PC.

## Situation (2026-10-16)
- Laser: Toptica DL pro ECDL with DLC pro controller, locked to Rb D2 (780.241 nm vacuum, 384.230 THz) via a
  saturated-absorption spectroscopy signal.
- Last known good point (about a week ago): current 228 mA, piezo scan range about 26-29 V.
- Now the lock cannot be found:
  - near 228 mA there are at most faint traces of the lines; sliding the piezo offset gives a rapid sawtooth (mode hopping).
  - at 209-211 mA bumps appear from the right of the scan, but as soon as the main lines come in, it mode hops.
- Recent changes on the table that may matter:
  - an LO tap-off arm with fiber coupling was added for a homodyne experiment (possible back-reflection into the laser).
  - a waveplate was once turned so that the spectroscopy balanced detector saturated (flat at -4 V); it was fixed then,
    but check that the spectroscopy detector is not saturated or pinned.
- Mode-hop reasoning already given to the user: current moves both frequency and the hop boundaries; the piezo moves
  frequency at nearly fixed hop boundaries. Hop-free windows repeat every few mA and shift with temperature.

## Goal
Write and run a script (Python, `toptica-lasersdk`) that maps the DLC pro over current and piezo offset, saves the scope
trace at each point, and scores each trace for (1) scan continuity (no mode-hop jumps) and (2) presence of the Rb Doppler
dip pattern (four dips over ~6-7 GHz: 87Rb F=2, 85Rb F=3, 85Rb F=2, 87Rb F=1). Output: a current x piezo map image, a CSV
of scores, raw traces on disk, and the best candidate operating points. The user does the final zoom and lock by hand.

## Safety rules (agreed with the user)
- Lock / feedback OFF during the sweep.
- Current only inside a window the user approves (start: 205-232 mA) and never above the DLC pro's current clip or the
  diode's maximum. Read the clip from the device first and refuse to run if the requested window exceeds it.
- No temperature changes unless the user explicitly says so.
- No grating adjustments (not possible remotely anyway).
- Log every set point; at the end (or on error / Ctrl-C) restore the starting current and piezo offset.
- Dry-run mode that prints the plan without touching the laser. Run it first.

## Facts to get from the user before writing code
1. How the DLC pro is connected: Ethernet (IP address) or USB.
2. DLC pro firmware version (system info menu); pick the matching `toptica.lasersdk.dlcpro.vX_Y_Z` module.
3. Which DLC pro input the spectroscopy photodiode feeds (e.g. Fine In 1, Fine In 2, Main In).
4. Approved current window and the diode's maximum current; current feedforward on or off, and its value last week.

## Implementation notes (verify every name against the installed SDK version; do not trust these blindly)
- `pip install toptica_lasersdk`. Connection: `DLCpro(NetworkConnection(ip))` or `DLCpro(SerialConnection(port))` as a
  context manager.
- Parameters (v2.x naming): `laser1.dl.cc.current_set`, `laser1.dl.cc.current_clip`, `laser1.scan.offset`,
  `laser1.scan.amplitude`, `laser1.scan.frequency`, `laser1.scan.output_channel`, `laser1.scope.channel1.signal`,
  `laser1.scope.data` (binary; decode with `toptica.lasersdk.utils.dlcpro.extract_float_arrays`).
- Read and print the current values of all of these before changing anything, and keep them for the restore.
- Wait for the scan to settle (a few scan periods) after each set point before reading the scope.
- Mode-hop detection: jumps in the trace much larger than the point-to-point noise, or a sawtooth pattern.
- Dip detection: smooth, subtract a linear/low-order baseline, find dips of a few hundred MHz width; score by count and depth.
- Start coarse (0.5 mA, wide scan amplitude), then refine around the best region.

## User preferences
- Never git push; commit locally only if asked.
- Keep explanations short; show the map image.
