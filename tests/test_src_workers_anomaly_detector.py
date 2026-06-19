import pytest
from unittest.mock import MagicMock, patch
from src.workers.anomaly_detector import AnomalyDetector, detect_anomalies, process_signal


@pytest.fixture
def mock_db():
    with patch("src.workers.anomaly_detector.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.workers.anomaly_detector.redis") as mock:
        yield mock


@pytest.fixture
def mock_logger():
    with patch("src.workers.anomaly_detector.logger") as mock:
        yield mock


@pytest.fixture
def detector(mock_db, mock_redis):
    return AnomalyDetector(window_size=10, threshold=2.0)


class TestAnomalyDetector:
    def test_init_sets_parameters(self, mock_db, mock_redis):
        detector = AnomalyDetector(window_size=5, threshold=3.0)
        assert detector.window_size == 5
        assert detector.threshold == 3.0
        assert len(detector.buffer) == 0

    def test_update_buffer_adds_values(self, detector):
        detector.update_buffer([1.0, 2.0, 3.0])
        assert detector.buffer == [1.0, 2.0, 3.0]

    def test_update_buffer_truncates_to_window_size(self, detector):
        detector.window_size = 3
        detector.update_buffer([1.0, 2.0, 3.0, 4.0, 5.0])
        assert detector.buffer == [3.0, 4.0, 5.0]

    def test_calculate_statistics_empty_buffer_raises(self, detector):
        with pytest.raises(ValueError, match="Buffer is empty"):
            detector.calculate_statistics()

    def test_calculate_statistics_single_value(self, detector):
        detector.update_buffer([5.0])
        mean, std = detector.calculate_statistics()
        assert mean == 5.0
        assert std == 0.0

    def test_calculate_statistics_multiple_values(self, detector):
        detector.update_buffer([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = detector.calculate_statistics()
        assert abs(mean - 3.0) < 1e-9
        assert abs(std - 1.41421356) < 1e-8

    def test_detect_single_value_no_anomaly(self, detector):
        detector.update_buffer([2.0])
        result = detector.detect()
        assert result == []

    def test_detect_no_anomalies(self, detector):
        detector.update_buffer([2.0, 2.1, 1.9, 2.05])
        result = detector.detect()
        assert result == []

    def test_detect_anomalies(self, detector):
        detector.update_buffer([2.0, 2.0, 2.0, 10.0])
        result = detector.detect()
        assert len(result) == 1
        assert result[0]["index"] == 3
        assert result[0]["value"] == 10.0

    def test_detect_anomalies_multiple(self, detector):
        detector.update_buffer([1.0, 1.0, 1.0, 10.0, 1.0, 10.0])
        result = detector.detect()
        assert len(result) == 2
        assert result[0]["index"] == 3
        assert result[1]["index"] == 5

    def test_detect_with_zero_std_no_anomalies(self, detector):
        detector.update_buffer([5.0, 5.0, 5.0])
        result = detector.detect()
        assert result == []

    def test_reset_clears_buffer(self, detector):
        detector.update_buffer([1.0, 2.0])
        detector.reset()
        assert len(detector.buffer) == 0


class TestDetectAnomalies:
    def test_detect_anomalies_calls_detector(self, mock_db, mock_redis, mock_logger):
        with patch("src.workers.anomaly_detector.AnomalyDetector") as MockDetector:
            mock_instance = MockDetector.return_value
            mock_instance.detect.return_value = [{"index": 0, "value": 10.0}]
            result = detect_anomalies([1.0, 2.0], window_size=5, threshold=2.0)
            assert result == [{"index": 0, "value": 10.0}]
            mock_instance.update_buffer.assert_called_once_with([1.0, 2.0])

    def test_detect_anomalies_logs_result(self, mock_db, mock_redis, mock_logger):
        with patch("src.workers.anomaly_detector.AnomalyDetector") as MockDetector:
            mock_instance = MockDetector.return_value
            mock_instance.detect.return_value = [{"index": 0, "value": 10.0}]
            detect_anomalies([1.0, 2.0])
            mock_logger.info.assert_called_once_with("Detected 1 anomalies")


class TestProcessSignal:
    def test_process_signal_saves_to_db(self, mock_db, mock_redis, mock_logger):
        mock_db.save_signal.return_value = True
        mock_redis.publish.return_value = 1
        result = process_signal([1.0, 2.0, 3.0], channel="signals")
        assert result["status"] == "processed"
        mock_db.save_signal.assert_called_once_with([1.0, 2.0, 3.0])

    def test_process_signal_publishes_anomalies(self, mock_db, mock_redis, mock_logger):
        mock_db.save_signal.return_value = True
        mock_redis.publish.return_value = 1
        with patch("src.workers.anomaly_detector.detect_anomalies") as mock_detect:
            mock_detect.return_value = [{"index": 2, "value": 3.0}]
            result = process_signal([1.0, 2.0, 3.0], channel="signals")
            assert result["anomalies"] == [{"index": 2, "value": 3.0}]
            mock_redis.publish.assert_called_once_with("signals", '[{"index": 2, "value": 3.0}]')

    def test_process_signal_handles_empty_signal(self, mock_db, mock_redis, mock_logger):
        mock_db.save_signal.return_value = True
        mock_redis.publish.return_value = 0
        result = process_signal([], channel="signals")
        assert result["status"] == "processed"
        assert result["anomalies"] == []
        mock_redis.publish.assert_not_called()

    def test_process_signal_logs_error_on_db_failure(self, mock_db, mock_redis, mock_logger):
        mock_db.save_signal.return_value = False
        result = process_signal([1.0, 2.0], channel="signals")
        assert result["status"] == "error"
        mock_logger.error.assert_called_once_with("Failed to save signal to DB")