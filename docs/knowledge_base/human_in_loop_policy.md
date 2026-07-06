# Human-in-the-Loop Policy

The governance stance for this prototype. It defines how model outputs may and
may not be used. This policy is intentionally conservative.

## Model Outputs Are Advisory

Every output of this system — RUL estimate, risk band, retrieved evidence, and
generated diagnostic summary — is advisory decision support, not a decision.
The system surfaces evidence and a suggested reading; a qualified human decides
what to do. No output should be forwarded to a work order or a parts request
without human confirmation.

## Human Review Is Mandatory

A qualified reviewer must confirm any high-risk flag before it drives action.
Review is not optional and is not a rubber stamp: the reviewer checks data
quality, corroborates sensors, and weighs uncertainty (see
maintenance_review_checklist). The model's role is to focus attention, not to
replace judgment.

## No Safety-Critical Automation

The system must not close any safety-critical loop. It does not command
shutdowns, dispatch maintenance automatically, or gate
airworthiness/operational-readiness decisions. It is a monitoring and triage
aid built on public simulation data, and it is explicitly not qualified for
autonomous control of real equipment.

## Uncertainty Must Travel With the Output

Any diagnostic summary must carry its uncertainty and its evidence. A number
without a stated confidence, and without citations to the signals that support
it, is not actionable. When uncertainty is high — particularly near
end-of-life — the correct output is "escalate to human review," not a
confident recommendation.

## Escalation Path

Flags follow a defined path: automated flag → reviewer triage → senior review
on disagreement or high uncertainty → maintenance decision by an accountable
human. Disagreements between the model and the sensor evidence are escalated,
not silently overridden in either direction, and the resolution is recorded.
