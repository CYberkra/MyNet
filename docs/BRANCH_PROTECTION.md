# Master Branch Protection

Repository policy is enforced partly in code and partly in GitHub settings.
An administrator should configure `master` with the following controls after
the `research-ci` workflow has completed successfully at least once.

## Required GitHub settings

1. Require a pull request before merging.
2. Require the `research-ci / cpu-contract-and-regression` status check.
3. Require the branch to be up to date before merging.
4. Block force pushes and branch deletion.
5. Restrict direct pushes to release administrators only.
6. Require conversation resolution when more than one reviewer is involved.

The self-hosted `official-mamba2-cuda` workflow is manual and must **not** be a
universal required status check. It is only required for changes that affect
the official Mamba2 backend, CUDA runtime contract, or formal 501x256 GPU
claim.

## Exceptions

An emergency direct commit is permitted only to restore repository access or
repair a broken CI workflow. It must be followed immediately by a retrospective
PR or handoff record stating the reason, validation result, and corrective
action.

## Why this remains a manual setting

Branch protection belongs to the remote repository administration plane rather
than source code. The versioned policy here makes the intended configuration
reviewable; GitHub must apply it using repository settings or an authenticated
administrator API.
