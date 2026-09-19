from milknado.web import (
    ObserverHandlers,
    OwnerHandlers,
    WebCommands,
    observer_commands,
    owner_commands,
)


def test_host_builders_set_capabilities() -> None:
    assert owner_commands(OwnerHandlers()).session_input is None
    assert observer_commands(ObserverHandlers()).force_stop is None
    assert WebCommands().cancel is None
