# tests/test_public_api.py
import cutedsl_decode as cd


def test_public_surface():
    assert hasattr(cd, "PagedDecodeWrapper")
    assert hasattr(cd, "get_capabilities")
    assert hasattr(cd, "is_supported")
    assert isinstance(cd.__version__, str)
    # arch-specific names must NOT be exported
    assert not hasattr(cd, "GroupedQueryAttentionDecodePagedSm90")
    assert "Sm90" not in " ".join(getattr(cd, "__all__", []))
