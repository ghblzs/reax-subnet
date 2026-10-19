# Owner award approvals

`approvals.json` holds the subnet owner's signed award approvals (docs/COMPETITIONS.md, "Owner approval").
`deploy/validator-1.2` mounts this directory into the validator by default (`REAX_AWARD_APPROVALS_DIR`, default
`../approvals`), so a validator that runs from a checkout of this repository picks up new approvals with `git pull`.
The validator re-reads the file when it changes; no restart is needed.

Every entry is signed with the owner's Ed25519 key (release constant `award.AWARD_APPROVAL_PUBKEY`); a changed or
forged entry fails verification and pays nothing. Check the file with:

    reax-award-approve verify --approvals deploy/approvals/approvals.json

If you keep `REAX_AWARD_APPROVALS_DIR` elsewhere, copy this file there after each update. Never mount the approvals
directory into the competition worker.
