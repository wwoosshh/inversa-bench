from inversa.adapters.fake import FakeAdapter


def test_fake_adapter_returns_scripted_outputs_in_order():
    a = FakeAdapter(["2*x = 6", "x = 1"])
    assert a.generate("ignored prompt") == "2*x = 6"
    assert a.generate("ignored prompt") == "x = 1"


def test_fake_adapter_cycles_when_exhausted():
    a = FakeAdapter(["only"])
    assert a.generate("p") == "only"
    assert a.generate("p") == "only"


def test_fake_adapter_records_prompts():
    a = FakeAdapter(["x = 0"])
    a.generate("first prompt")
    assert a.prompts == ["first prompt"]
