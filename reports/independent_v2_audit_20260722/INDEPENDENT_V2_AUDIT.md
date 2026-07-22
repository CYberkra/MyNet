# Independent V2 Promotion Audit

Date: 2026-07-22

## Decision

No existing V2 candidate is approved for formal training. The formal data gate
remains closed.

The audit distinguishes provenance from release evidence:

| Family | Provenance | Recorded scientific status | Audit disposition |
|---|---|---|---|
| F01 | Non-Line9-conditioned | Independent pilot passed sparse gates | Blocked by material-file hash mismatch; rebuild as a new family revision before native-256 runs. |
| F02 | Line9-conditioned mechanism transfer | Development-only comparison | Permanently excluded from strict Line9 formal training; also has the hash mismatch. |
| F03 | Non-Line9-conditioned | Independent physics candidate, visually not preferred | Hold; do not scale or promote; also has the hash mismatch. |

## Evidence Checked

- The registered simulation catalog contains 13 cases. Nine are explicitly
  Line9-conditioned development/transfer cases; F01 and F03 are the only
  independently declared positive/negative candidate pairs.
- The F01 and F03 contracts declare an indivisible positive/target-absent
  split group, a shared geometry state, and a target-absent negative equal to
  the positive no-basal material state.
- The independent-family, V2-control, packaging, and promotion regression
  suite passed: `45 passed`.
- Existing pilot reports do not contain complete native-256 released solver
  evidence or independent human release approval.

## Integrity Blocker

For F01, F02, and F03, the recorded SHA-256 of the positive no-basal material
file and the matched negative full material file does not equal the current
versioned file bytes. The two files in each pair remain byte-identical to one
another, but their recorded provenance hash is stale or was computed under a
different byte representation.

The likely historical cause is cross-platform text serialization, but this is
not treated as proof. Recomputing manifests in place would erase the evidence
gap, so this audit does not modify the old assets or their hashes.

## Required Successor

Create a new F01 successor rather than repairing the old pilot in place:

1. Make all hash-protected generator text output explicitly LF and add a
   cross-platform hash regression test.
2. Regenerate the F01 positive, exact no-basal control, and matched
   target-absent negative in a new immutable family directory.
3. Verify raw file hashes, shared geometry, and material-map equivalence
   before any solver run.
4. Run one-trace full/control plus negative equality, then the required
   native-256 positive full/control/air and negative full runs.
5. Extract visible phase only from the solved signed pair, package immutable
   evidence, and require an independent human release decision.

F02 remains a development-only mechanism ablation. F03 remains a retained
physics comparison and is not the preferred morphology successor.
