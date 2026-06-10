"""Test that the Streamlit UI actually loads end-to-end.

Uses Streamlit's official AppTest harness — the same code path that runs
when a real browser opens a session. Regression guard for the import bugs
hit during development:
  1. ImportError (script-name colliding with package name)
  2. ModuleNotFoundError (project root missing from sys.path)

If the UI breaks again, this test catches it without needing a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "ui" / "streamlit_app.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="UI script not present")
def test_streamlit_ui_loads_without_import_error():
    """The Streamlit script must execute end-to-end without ImportError /
    ModuleNotFoundError. We don't assert on widget contents — we only
    care that the imports + page setup succeed.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(SCRIPT))
    at.run(timeout=60)

    # If anything threw during script execution, AppTest.exception is set.
    assert at.exception is None or at.exception == [], (
        f"Streamlit script raised: {at.exception}"
    )
    # Sanity: the title we set should be present.
    titles = [t.value for t in at.title]
    assert any("SAP SDGs" in t for t in titles), (
        f"Expected page title not found. Got: {titles}"
    )
