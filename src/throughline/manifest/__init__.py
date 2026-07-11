"""Live environment manifest capture and policy-based verification.

``capture_environment`` probes the workspace into a provenance-structured
manifest (``live`` vs ``harness``-attested). ``verify_manifest`` diffs a
lockfile against the flat view (``flatten_observed``). ``ManifestGate``
wires that check into a Flow; session helpers record the same facts for
post-hoc audit with an honest guarantee: Throughline verified the
workspace directly and verified harness-attested agent configuration.
"""

from .capture import (HARNESS_KEYS, LIVE_KEYS, PROVENANCE_SECTIONS,
                      SOURCE_HARNESS, SOURCE_LIVE_PROBE, capture_environment,
                      env_hash, flatten_observed, git_snapshot, observed_sources,
                      runtime_snapshot)
from .session import (SessionRecorder, capture_drift, declared_config,
                      effective_environment, preflight_session_start,
                      session_start_event, verify_live)
from .verify import (DEFAULT_VERIFY_POLICY, VerifyResult, Violation,
                     load_lockfile, policy_action, verify_manifest)

__all__ = [
    "DEFAULT_VERIFY_POLICY",
    "HARNESS_KEYS",
    "LIVE_KEYS",
    "PROVENANCE_SECTIONS",
    "SOURCE_HARNESS",
    "SOURCE_LIVE_PROBE",
    "SessionRecorder",
    "VerifyResult",
    "Violation",
    "capture_drift",
    "capture_environment",
    "declared_config",
    "effective_environment",
    "env_hash",
    "flatten_observed",
    "git_snapshot",
    "load_lockfile",
    "observed_sources",
    "policy_action",
    "preflight_session_start",
    "runtime_snapshot",
    "session_start_event",
    "verify_live",
    "verify_manifest",
]
