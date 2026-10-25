# Predictions written BEFORE the experiments they refer to

## Q2.5 (written 2026-10-25, after Figure 1, Table 2 and Figure 2, before any experiment of "Testing the Baseline")

Mirror: I expect model A to mostly pass. Figure 1(a) shows directions spread over all angles (no strongly preferred direction), so mirrored trajectories are familiar,
and Figure 2 shows that A predicts soft blobs rather than sharp digits, so it should rely little on what a digit looks like; I expect a small consistency gap C_M and
E_M close to E(5) on the original clips.
Rotation: also mostly pass, but slightly worse than mirror (C_R >= C_M and E_R >= E_M) because rotated digits look even less familiar and the 4x4 patch folding and
fixed kernels are not rotation-symmetric; the direction of motion is still familiar from Figure 1(a).
Speed: my reach estimate for A is 8 original pixels per step (Q2.3) and the median training speed is 3.3 px per frame (Figure 1b). So 2x (about 6.6 px per frame, the fastest digits about 9 px)
should still be tracked but with a visible loss of accuracy, and 3x (about 10 px, the fastest about 14 px) is beyond the reach, so I expect A to break down at 3x:
digits stall or vanish, and E(2) at 3x should be at least as bad as Copy-last.
