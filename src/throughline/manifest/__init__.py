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
                      runtime_snapshot, short_digest)
from .harness import HarnessKind, detect_harness, extract_harness_config
from .lockfile_io import (capture_lockfile, update_lockfile, verify_lockfile,
                          write_lockfile)
from .sanitize import AUDIT_KEYS, redact_secrets, sanitize_for_audit
from .session import (SessionRecorder, capture_drift, declared_config,
                      effective_environment, preflight_session_start,
                      session_start_event, verify_live)
from .verify import (DEFAULT_VERIFY_POLICY, VerifyResult, Violation,
                     load_lockfile, policy_action, verify_manifest)

__all__ = [
    "AUDIT_KEYS",
    "DEFAULT_VERIFY_POLICY",
    "HARNESS_KEYS",
    "LIVE_KEYS",
    "PROVENANCE_SECTIONS",
    "SOURCE_HARNESS",
    "SOURCE_LIVE_PROBE",
    "HarnessKind",
    "SessionRecorder",
    "VerifyResult",
    "Violation",
    "capture_drift",
    "capture_environment",
    "capture_lockfile",
    "declared_config",
    "detect_harness",
    "effective_environment",
    "env_hash",
    "extract_harness_config",
    "flatten_observed",
    "git_snapshot",
    "load_lockfile",
    "observed_sources",
    "policy_action",
    "preflight_session_start",
    "redact_secrets",
    "runtime_snapshot",
    "sanitize_for_audit",
    "session_start_event",
    "short_digest",
    "update_lockfile",
    "verify_live",
    "verify_lockfile",
    "verify_manifest",
    "write_lockfile",
]
