import json

import pytest

from inversa.cli_calibration import main


def test_rejects_non_integer_levels(tmp_path):
    bank = tmp_path / "b.json"
    bank.write_text(json.dumps({"targets": [3]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--bank", str(bank), "--levels", "abc"])


def test_rejects_bank_without_targets(tmp_path):
    bank = tmp_path / "b.json"
    bank.write_text(json.dumps({"domain": "math_equation"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--bank", str(bank), "--levels", "1,2"])
