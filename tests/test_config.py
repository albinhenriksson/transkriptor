from ljudanteckning.config import load_settings


def test_load_settings_no_files():
    s = load_settings(None)
    assert s.chunk_seconds > 0
    assert s.sample_rate == 16000
    assert s.device == "cuda"
