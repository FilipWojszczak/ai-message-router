import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    import message_router

    assert message_router.__name__ == "message_router"
