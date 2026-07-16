# Instrument-Band Pulse Proxies

Use this reference when a time-domain gprMax source must approximate a
wideband or stepped-frequency UAV-GPR instrument without measured complex
system phase.

## Evidence hierarchy

1. Project-wide antenna centre and hardware passband may define a source-family
   envelope.
2. A measured complex transfer function may define an instrument-faithful
   transient, provided it is acquired independently of the held-out line.
3. Magnitude-only frequency samples cannot recover causal phase. A zero-phase
   inverse-FFT pulse is an explicit proxy, not a reconstruction of the emitted
   waveform.
4. A held-out B-scan may evaluate generalization after parameters are frozen;
   it must not select source frequency, phase, or pulse width for formal data.

## Required ablation contract

Keep geometry, material fields, grid, PML, acquisition, and random seeds fixed.
Vary only the source family. Predeclare candidate pulses from hardware and
literature constraints, run static checks and one-trace matched controls for
all candidates, then run sparse B-scans only for candidates that pass.

Evaluate at least:

- causal full-minus-control timing and early equality;
- signed-lobe count and polarity sequence;
- aligned peak and centroid frequency;
- target/adjacent-background RMS;
- envelope coefficient of variation and dropout;
- blind raw, time-power, and AGC morphology under shared scales.

Do not select a pulse solely because AGC makes it visually strong. Reject
isolated hyperbolas, full-depth vertical partitions, missing target packets,
or a packet that becomes implausibly thin/high-frequency relative to the
declared hardware family.

## Family 03 lesson

The project-wide 100 MHz, 20-170 MHz zero-phase band proxy had a source peak of
100.10 MHz and centroid of 98.43 MHz. On independent Family 01 geometry it
passed causal and geometry gates, retained seven signed lobes, and had no
dropout. The solved aligned packet nevertheless shifted to a 116.20 MHz
centroid, became visually narrower than the accepted development morphology,
and reduced target/adjacent-background RMS from Family 02's 14.96 to 9.65.

Therefore hardware consistency alone is not a visual-morphology guarantee.
Retain this pulse as a valid diversity candidate, not a universal template.
Use a predeclared source-family ablation rather than tuning it against Line9.
