import type { ReactElement } from "react";
import { useSyncExternalStore } from "react";
import { canActOnSelectedRun, getState, subscribe } from "../../app/store";
import { Milknado } from "../../design-system";
import { sendSessionCommand } from "./sessionCommand";

/** The `sidecar-action` contribution: Approve/Deny for the oldest pending permission request. */
export function PermissionActions(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button, StatusBadge } = Milknado;
  const owner = store.capabilities?.owner;
  const permissionId = owner?.permission_ids?.[0];
  const permissionCommand = owner?.permission_commands?.find(([id]) => id === permissionId)?.[1];

  if (!canActOnSelectedRun(store) || !permissionId || permissionCommand === undefined) {
    return null;
  }

  return (
    <section className="mk-section mk-permission" aria-label="Permission requested">
      <div className="mk-badge-row">
        <StatusBadge state="at-risk">Permission requested</StatusBadge>
        <span className="mk-text-data" aria-label={`Request ID ${permissionId}`} style={{ color: "var(--mk-at-risk)" }}>
          {permissionId}
        </span>
      </div>
      {permissionCommand.length > 0 && (
        <div className="mk-section">
          <span className="mk-kicker">Command line</span>
          <code className="mk-code">{permissionCommand}</code>
        </div>
      )}
      <p className="mk-text-body">
        The agent waits for a decision on this request.
      </p>
      <div className="mk-button-row">
        <Button
          variant="primary"
          className="mk-btn-sm"
          onClick={() =>
            void sendSessionCommand("approve", { requestId: permissionId })
          }
        >
          Approve
        </Button>
        <Button
          className="mk-btn-sm"
          onClick={() =>
            void sendSessionCommand("deny", { requestId: permissionId })
          }
        >
          Deny
        </Button>
      </div>
    </section>
  );
}