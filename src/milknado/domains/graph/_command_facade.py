from __future__ import annotations

import milknado.domains.graph._command_persistence as _command_persistence
from milknado.domains.graph._analytics_facade import synchronized
from milknado.domains.graph._command_claim import ClaimRequest, claim_queued_commands
from milknado.domains.graph._command_close import close_owner
from milknado.domains.graph._facade_base import SubFacade
from milknado.domains.graph.commands import (
    CommandReceipt,
    CommandStatus,
    GraphCommand,
    OwnerCapabilities,
)


class _CommandFacade(SubFacade):
    @synchronized
    def publish_capabilities(  # noqa: PLR0913
        self,
        run_id: str,
        node_id: int,
        invocation_id: str,
        owner_incarnation: str,
        actions: tuple[str, ...],
        permission_ids: tuple[str, ...] = (),
        *,
        published_at: str | None = None,
    ) -> OwnerCapabilities:
        return _command_persistence.publish_capabilities(
            self._conn,
            run_id,
            node_id,
            invocation_id,
            owner_incarnation,
            actions,
            permission_ids,
            published_at=published_at,
        )

    @synchronized
    def capabilities(self, run_id: str) -> OwnerCapabilities | None:
        return _command_persistence.get_capabilities(self._conn, run_id)

    @synchronized
    def admit(  # noqa: V105 - public graph command API
        self,
        command: GraphCommand,
        *,
        now: str | None = None,
        max_pending: int = 64,
    ) -> CommandReceipt:
        return _command_persistence.admit_command(
            self._conn, command, now=now, max_pending=max_pending
        )

    @synchronized
    def command(self, command_id: str) -> GraphCommand | None:
        return _command_persistence.get_command(self._conn, command_id)

    @synchronized
    def receipt(self, command_id: str) -> CommandReceipt | None:
        return _command_persistence.get_receipt(self._conn, command_id)

    @synchronized
    def pending(
        self,
        run_id: str | None = None,
        *,
        now: str | None = None,
        limit: int = 64,
    ) -> tuple[GraphCommand, ...]:
        return _command_persistence.queued_commands(self._conn, run_id, now=now, limit=limit)

    @synchronized
    def claim_pending(
        self, run_id: str, owner_incarnation: str, *, now: str | None = None, limit: int = 64
    ) -> tuple[GraphCommand, ...]:
        return claim_queued_commands(
            self._conn, ClaimRequest(run_id, owner_incarnation), now=now, limit=limit
        )

    @synchronized
    def history(self, command_id: str) -> tuple[CommandReceipt, ...]:
        return _command_persistence.receipt_history(self._conn, command_id)

    @synchronized
    def expire(  # noqa: V105 - public graph command API
        self, *, now: str | None = None
    ) -> tuple[CommandReceipt, ...]:
        return _command_persistence.expire_commands(self._conn, now=now)

    @synchronized
    def transition(  # noqa: PLR0913
        self,
        command_id: str,
        status: CommandStatus,
        *,
        node_id: int,
        run_id: str,
        invocation_id: str,
        owner_incarnation: str,
        permission_id: str | None = None,
        now: str | None = None,
        detail: str | None = None,
    ) -> CommandReceipt:
        return _command_persistence.transition_command(
            self._conn,
            command_id,
            status,
            node_id=node_id,
            run_id=run_id,
            invocation_id=invocation_id,
            owner_incarnation=owner_incarnation,
            permission_id=permission_id,
            now=now,
            detail=detail,
        )

    def _transition(
        self,
        command: GraphCommand,
        status: CommandStatus,
        *,
        now: str | None = None,
        detail: str | None = None,
    ) -> CommandReceipt:
        return self.transition(
            command.command_id,
            status,
            node_id=command.node_id,
            run_id=command.run_id,
            invocation_id=command.invocation_id,
            owner_incarnation=command.owner_incarnation,
            permission_id=command.permission_id,
            now=now,
            detail=detail,
        )

    def submit(self, command: GraphCommand, *, now: str | None = None) -> CommandReceipt:
        return self._transition(command, "submitted", now=now)

    @synchronized
    def deliver(  # noqa: V105 - public graph command API
        self, command: GraphCommand, *, now: str | None = None
    ) -> CommandReceipt:
        return self._transition(command, "delivered", now=now)

    def reject(  # noqa: V105 - public graph command API
        self, command: GraphCommand, *, now: str | None = None, detail: str | None = None
    ) -> CommandReceipt:
        return self._transition(command, "rejected", now=now, detail=detail)

    def unconfirm(  # noqa: V105 - public graph command API
        self, command: GraphCommand, *, now: str | None = None, detail: str | None = None
    ) -> CommandReceipt:
        return self._transition(command, "unconfirmed", now=now, detail=detail)

    @synchronized
    def close_owner(self, run_id: str, owner_incarnation: str, invocation_id: str) -> None:
        close_owner(self._conn, run_id, owner_incarnation, invocation_id)


__all__ = ["_CommandFacade"]
