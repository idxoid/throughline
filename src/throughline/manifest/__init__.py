"""Live environment manifest capture and policy-based verification.

``capture_environment`` probes the current workspace; ``verify_manifest``
diffs an expected lockfile against observed facts and returns pass/warn/block.
``ManifestGate`` wires that check into a Flow before steps run, while session
helpers let external harnesses record the same observed facts for post-hoc
audit.
"""

from .capture import capture_environment, env_hash, git_snapshot, runtime_snapshot
from .session import (SessionRecorder, capture_drift, declared_config,
                      effective_environment, preflight_session_start,
                      session_start_event, verify_live)
from .verify import (DEFAULT_VERIFY_POLICY, VerifyResult, Violation,
                     load_lockfile, policy_action, verify_manifest)

__all__ = [
    "DEFAULT_VERIFY_POLICY",
    "SessionRecorder",
    "VerifyResult",
    "Violation",
    "capture_drift",
    "capture_environment",
    "declared_config",
    "effective_environment",
    "env_hash",
    "git_snapshot",
    "load_lockfile",
    "policy_action",
    "preflight_session_start",
    "runtime_snapshot",
    "session_start_event",
    "verify_live",
    "verify_manifest",
]
