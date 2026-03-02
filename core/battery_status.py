from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BatteryStatus:
    available: bool
    percent: int | None = None
    voltage_volts: float | None = None
    state: str | None = None
    source: str | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        if not self.available:
            return "Battery: No sensor"

        parts: list[str] = ["Battery:"]
        if self.percent is not None:
            parts.append(f"{self.percent}%")
        elif self.voltage_volts is not None:
            parts.append(f"{self.voltage_volts:.2f}V")
        else:
            parts.append("Detected")

        if self.state:
            parts.append(f"({self.state})")
        return " ".join(parts)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _parse_percent(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        value = int(float(raw))
    except Exception:
        return None
    return max(0, min(100, value))


def _parse_voltage(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if value > 1000:
        value /= 1_000_000.0
    return max(0.0, value)


def _status_from_base(base: Path) -> BatteryStatus | None:
    if not base.exists():
        return None

    percent = _parse_percent(_read_text(base / "capacity"))
    voltage = _parse_voltage(
        _read_text(base / "voltage_now")
        or _read_text(base / "voltage_avg")
    )
    state = _read_text(base / "status")

    if percent is None and voltage is None:
        return None

    return BatteryStatus(
        available=True,
        percent=percent,
        voltage_volts=voltage,
        state=state,
        source=str(base),
    )


def _read_i2c_byte(bus: int, address: int, register: int) -> int | None:
    i2cget = shutil.which("i2cget")
    if not i2cget:
        return None
    try:
        result = subprocess.run(
            [i2cget, "-y", str(bus), hex(address), hex(register)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    raw = result.stdout.strip().lower()
    if not raw.startswith("0x"):
        return None
    try:
        return int(raw, 16)
    except Exception:
        return None


def _read_max1704x_status() -> BatteryStatus | None:
    bus = int(os.getenv("ASL_BATTERY_I2C_BUS", "1"))
    address = int(os.getenv("ASL_BATTERY_I2C_ADDR", "0x36"), 0)

    soc_msb = _read_i2c_byte(bus, address, 0x04)
    soc_lsb = _read_i2c_byte(bus, address, 0x05)
    vcell_msb = _read_i2c_byte(bus, address, 0x02)
    vcell_lsb = _read_i2c_byte(bus, address, 0x03)

    if soc_msb is None or soc_lsb is None:
        return None

    percent = int(max(0.0, min(100.0, soc_msb + (soc_lsb / 256.0))))
    voltage = None
    if vcell_msb is not None and vcell_lsb is not None:
        raw_vcell = (vcell_msb << 4) | (vcell_lsb >> 4)
        voltage = raw_vcell * 0.00125

    return BatteryStatus(
        available=True,
        percent=percent,
        voltage_volts=voltage,
        source=f"i2c:{bus}:{hex(address)}",
    )


def read_battery_status() -> BatteryStatus:
    capacity_override = os.getenv("ASL_BATTERY_CAPACITY_PATH", "").strip()
    voltage_override = os.getenv("ASL_BATTERY_VOLTAGE_PATH", "").strip()

    if capacity_override or voltage_override:
        percent = _parse_percent(_read_text(Path(capacity_override))) if capacity_override else None
        voltage = _parse_voltage(_read_text(Path(voltage_override))) if voltage_override else None
        if percent is not None or voltage is not None:
            return BatteryStatus(
                available=True,
                percent=percent,
                voltage_volts=voltage,
                source="env-override",
            )

    power_supply_root = Path("/sys/class/power_supply")
    if power_supply_root.exists():
        for device in sorted(power_supply_root.iterdir()):
            status = _status_from_base(device)
            if status is not None:
                return status

    max1704x_status = _read_max1704x_status()
    if max1704x_status is not None:
        return max1704x_status

    return BatteryStatus(available=False)
