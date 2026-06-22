import re
import threading

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


class SerialManager:
    """Serial communication helper for STM32 telemetry streaming."""

    TELEMETRY_PATTERN = re.compile(
        r"(Vin|Vo|Iin|Io):\s*([-+]?\d+(?:\.\d+)?)\s*([VAvabB])",
        re.IGNORECASE
    )

    def __init__(self, on_telemetry_callback=None, on_status_callback=None, on_raw_line_callback=None):
        self.on_telemetry_callback = on_telemetry_callback
        self.on_status_callback = on_status_callback
        self.on_raw_line_callback = on_raw_line_callback

        self._serial = None
        self._rx_thread = None
        self._stop_event = threading.Event()

    @property
    def is_connected(self):
        return self._serial is not None and self._serial.is_open

    @staticmethod
    def list_ports():
        """Return available COM ports as a list of names."""
        if list_ports is None:
            return []
        return [port.device for port in list_ports.comports()]

    def connect(self, port, baudrate=115200, timeout=0.2):
        """Open serial port and start receiver thread."""
        if serial is None:
            return False, "pyserial is not installed. Install with: pip install pyserial"

        if self.is_connected:
            self.disconnect()

        try:
            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
            self._stop_event.clear()
            self._rx_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._rx_thread.start()
            self._notify_status(f"Connected to {port}")
            return True, f"Connected to {port}"
        except Exception as exc:
            self._serial = None
            error_text = f"Could not connect to {port}: {exc}"
            self._notify_status(error_text)
            return False, error_text

    def disconnect(self):
        """Close serial port and stop receiver thread."""
        self._stop_event.set()

        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=0.5)
        self._rx_thread = None

        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None

        self._notify_status("Disconnected")

    def send_line(self, text):
        """Send one command line to the STM32."""
        if not self.is_connected:
            return False
        payload = (text.rstrip("\r\n") + "\n").encode("utf-8")
        self._serial.write(payload)
        return True

    def _read_loop(self):
        """Continuously parse incoming telemetry lines while connected."""
        while not self._stop_event.is_set() and self.is_connected:
            try:
                raw = self._serial.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if self.on_raw_line_callback:
                    self.on_raw_line_callback(line)
                telemetry = self._parse_telemetry_line(line)
                if telemetry and self.on_telemetry_callback:
                    self.on_telemetry_callback(telemetry)
            except Exception as exc:
                self._notify_status(f"Serial read error: {exc}")
                break

        # Ensure clean state if loop exits due to read error or disconnect.
        if self._serial is not None and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _parse_telemetry_line(self, line):
        """Parse lines in either scaled V/A format or raw ADC b format."""
        parsed_values = {}
        parsed_units = {}

        for name, value_text, unit_text in self.TELEMETRY_PATTERN.findall(line):
            key = name.lower()
            unit = unit_text.lower()
            parsed_units[key] = unit
            if unit == "b":
                parsed_values[key] = int(float(value_text))
            else:
                parsed_values[key] = float(value_text)

        if not parsed_values:
            return None

        telemetry = {}

        is_raw_adc = all(unit == "b" for unit in parsed_units.values())

        if is_raw_adc:
            if "vin" in parsed_values:
                telemetry["v in (v)"] = f"{parsed_values['vin']} b"
            if "iin" in parsed_values:
                telemetry["i in (a)"] = f"{parsed_values['iin']} b"
            if "vo" in parsed_values:
                telemetry["v out (v)"] = f"{parsed_values['vo']} b"
            if "io" in parsed_values:
                telemetry["i out (a)"] = f"{parsed_values['io']} b"

            telemetry["p in (w)"] = "N/A"
            telemetry["p out (w)"] = "N/A"
            telemetry["eff (%)"] = "N/A"
            telemetry["mode"] = "RAW ADC"
            return telemetry

        if "vin" in parsed_values:
            telemetry["v in (v)"] = f"{parsed_values['vin']:.3f}"
        if "iin" in parsed_values:
            telemetry["i in (a)"] = f"{parsed_values['iin']:.3f}"
        if "vo" in parsed_values:
            telemetry["v out (v)"] = f"{parsed_values['vo']:.3f}"
        if "io" in parsed_values:
            telemetry["i out (a)"] = f"{parsed_values['io']:.3f}"

        if all(key in parsed_values for key in ("vin", "iin", "vo", "io")):
            p_in = parsed_values["vin"] * parsed_values["iin"]
            p_out = parsed_values["vo"] * parsed_values["io"]
            telemetry["p in (w)"] = f"{p_in:.3f}"
            telemetry["p out (w)"] = f"{p_out:.3f}"
            telemetry["eff (%)"] = f"{(100.0 * p_out / p_in):.2f}" if p_in > 0 else "0.00"
            telemetry["mode"] = "SCALED"

        return telemetry if telemetry else None

    def _notify_status(self, text):
        if self.on_status_callback:
            self.on_status_callback(text)
