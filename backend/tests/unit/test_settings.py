import pytest
from pydantic import ValidationError

from talentscout.config.settings import Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "TalentScout API"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"
    assert settings.debug is False


def test_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALENTSCOUT_APP_NAME", "Test API")
    monkeypatch.setenv("TALENTSCOUT_ENVIRONMENT", "testing")
    monkeypatch.setenv("TALENTSCOUT_DEBUG", "true")

    settings = Settings()

    assert settings.app_name == "Test API"
    assert settings.environment == "testing"
    assert settings.debug is True


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALENTSCOUT_ENVIRONMENT", "invalid")

    with pytest.raises(ValidationError):
        Settings()


def test_database_url_has_expected_driver() -> None:
    settings = Settings()

    assert settings.database_url.startswith("postgresql+psycopg://")
