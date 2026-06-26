from __future__ import annotations

from typing import Any


def _set_if_supported(sensor: Any, option: Any, value: float) -> None:
    if sensor.supports(option):
        sensor.set_option(option, float(value))


def apply_color_options(
    sensor: Any,
    option_namespace: Any,
    exposure: float | None = None,
    gain: float | None = None,
    brightness: float | None = None,
    auto_exposure: bool | None = None,
    auto_exposure_priority: bool | None = None,
) -> None:
    """Apply RGB sensor options without touching the depth/stereo sensor."""

    if auto_exposure is not None:
        _set_if_supported(
            sensor,
            option_namespace.enable_auto_exposure,
            1.0 if auto_exposure else 0.0,
        )
    if exposure is not None:
        _set_if_supported(sensor, option_namespace.enable_auto_exposure, 0.0)
        _set_if_supported(sensor, option_namespace.exposure, exposure)
    if gain is not None:
        _set_if_supported(sensor, option_namespace.gain, gain)
    if brightness is not None:
        _set_if_supported(sensor, option_namespace.brightness, brightness)
    if auto_exposure_priority is not None:
        _set_if_supported(
            sensor,
            option_namespace.auto_exposure_priority,
            1.0 if auto_exposure_priority else 0.0,
        )
