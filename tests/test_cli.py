import json

import pytest

from inversa.cli import main


def test_cli_rejects_bank_without_targets(tmp_path):
    bank = tmp_path / "bad.json"
    bank.write_text(json.dumps({"domain": "math_equation"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--bank", str(bank)])
