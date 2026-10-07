from app.core.security import create_access_token_claims, decode_access_token


def test_access_token_round_trip_contains_expected_identity_claims() -> None:
    token = create_access_token_claims(1, "user", "演示用户", "CUSTOMER")

    payload = decode_access_token(token)

    assert payload["sub"] == "1"
    assert payload["username"] == "user"
    assert payload["name"] == "演示用户"
    assert payload["role"] == "CUSTOMER"
    assert "exp" in payload
    assert "jti" in payload
