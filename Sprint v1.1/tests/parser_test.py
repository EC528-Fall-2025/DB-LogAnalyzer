import tempfile
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from log_parser import parse_logs

def test_json_parsing():
    log_line = '{"Severity": "10", "DateTime": "2025-09-05T20:59:14Z", "Type": "GrvLatency", "Machine": "2.1.1.4:2"}\n'

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
        temp_file.write(log_line)   
        temp_file.flush()
        path = temp_file.name

    events = list(parse_logs(path))

    assert len(events) == 1
    assert events[0].severity == 10
    assert events[0].event == "GrvLatency"

def test_plaintext_parsing():
    log_line = 'Severity=20 DateTime=2025-09-05T21:00:00Z Type=TxnDone Machine=2.1.1.5:2\n'
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as temp_file:
        temp_file.write(log_line)
        temp_file.flush()
        path = temp_file.name

    events = list(parse_logs(path))
    assert len(events) == 1
    assert events[0].severity == 20
    assert events[0].event == "TxnDone"