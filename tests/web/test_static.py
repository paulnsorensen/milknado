# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import subprocess
import zipfile
from pathlib import Path

from tests.web.support import client


def test_logged_in_root_serves_index() -> None:
    response = client()[0].get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Execution dashboard loading." in response.text
    assert 'name="token"' not in response.text


def test_logged_out_root_serves_login_page() -> None:
    test_client, _ = client()
    test_client.cookies.clear()
    response = test_client.get("/")
    assert response.status_code == 200
    assert 'name="token"' in response.text
    assert 'action="/auth"' in response.text


def test_login_form_extracts_token_from_launch_url() -> None:
    test_client, login = client()
    test_client.cookies.clear()
    launch_url = f"http://127.0.0.1/?token={login.value}"
    login_script = test_client.get("/login.js")
    assert login_script.status_code == 200
    assert "submitLaunchUrl" in login_script.text
    script = """
const { submitLaunchUrl } = require(process.argv[1]);
const token = { value: "" };
let submitted = false;
submitLaunchUrl({ submit: () => { submitted = true; } }, { value: process.argv[2] }, token);
if (token.value !== process.argv[3] || !submitted) process.exit(1);
"""
    result = subprocess.run(
        [
            "node",
            "-e",
            script,
            str(Path("src/milknado/web/static/login.js").resolve()),
            launch_url,
            login.value,
        ],
        check=False,
    )
    assert result.returncode == 0
    authenticated = test_client.get("/auth", params={"token": login.value}, follow_redirects=False)
    assert authenticated.status_code == 303


def test_static_assets_require_login() -> None:
    test_client, _ = client()
    test_client.cookies.clear()
    assert test_client.get("/assets/missing.txt").status_code == 401


def test_wheel_contains_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "wheel"
    _ = subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output_dir)], check=True)
    wheels = list(output_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert "milknado/web/static/index.html" in wheel.namelist()


def test_static_assets_serve_committed_asset() -> None:
    test_client, _ = client()
    response = test_client.get("/assets/app.js")
    assert response.status_code == 200
    assert response.text == "console.log('Milknado');\n"


def test_static_assets_reject_traversal() -> None:
    test_client, _ = client()
    assert test_client.get("/assets/%2e%2e/index.html").status_code == 404


def test_static_assets_reject_encoded_nul() -> None:
    test_client, _ = client()
    assert test_client.get("/assets/%00").status_code == 404
