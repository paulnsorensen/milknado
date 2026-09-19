# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import zipfile

from tests.web.support import client


def test_logged_in_root_serves_index() -> None:
    response = client()[0].get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Milknado" in response.text


def test_logged_out_root_serves_login_page() -> None:
    test_client, _ = client()
    test_client.cookies.clear()
    response = test_client.get("/")
    assert response.status_code == 200
    assert 'name="token"' in response.text
    assert 'action="/auth"' in response.text


def test_static_assets_require_login() -> None:
    test_client, _ = client()
    test_client.cookies.clear()
    assert test_client.get("/assets/missing.txt").status_code == 401


def test_wheel_contains_index() -> None:
    import subprocess
    from pathlib import Path

    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", ".static-wheel-test"], check=True
    )
    assert result.returncode == 0
    wheels = list(Path(".static-wheel-test").glob("*.whl"))
    assert wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert "milknado/web/static/index.html" in wheel.namelist()
