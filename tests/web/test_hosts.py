from milknado.web import WebCommands, observer_commands, owner_commands


def test_host_builders_set_capabilities() -> None:
    assert owner_commands(lambda: None).session_input is not None
    assert observer_commands().force_stop is None
    assert WebCommands().cancel is None
