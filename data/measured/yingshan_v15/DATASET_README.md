# YingShan V15 Final Labels

This is the final audited label release, but it is not a formal training release.

- Original CSV waveform, GNSS, ground elevation, flight height, and acquisition order are preserved.
- `soft_mask_v14_original` preserves V14 exactly.
- `soft_mask_review_v15_final` stores the complete V15 review geometry.
- `soft_mask_train` excludes ambiguous and transition regions through `ignore_mask`.
- Accepted cross-line relocations remain weak labels.
- Line9 labels were not moved and Line9 remains test-only.
- X1 remains review-only/excluded.
- The split is locked as Train=LineL1/Line3/Line7, Validation=Line6,
  Test=Line9, and Review=LineX1.
- Formal training remains blocked by missing true negatives and approved
  independent simulations.
- Original source archives are retained under `source/`; canonical arrays
  remain in acquisition order and profile reversal is display-only.
