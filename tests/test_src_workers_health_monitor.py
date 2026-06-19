import pytest
from unittest.mock import MagicMock, patch
from src.workers.health_monitor import HealthMonitor, check_system_health, get_battery_status, get_cpu_usage, get_memory_usage, schedule_task_if_healthy


@pytest.fixture
def mock_db():
    with patch("src.workers.health_monitor.db") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.workers.health_monitor.redis_client") as mock:
        yield mock


@pytest.fixture
def mock_psutil():
    with patch("src.workers.health_monitor.psutil") as mock:
        mock.cpu_percent.return_value = 45.0
        mock.virtual_memory().percent = 60.0
        yield mock


@pytest.fixture
def mock_battery():
    with patch("src.workers.health_monitor.psutil.sensors_battery") as mock:
        mock.return_value = MagicMock(percent=85, secsleft=18000, power_plugged=False)
        yield mock


class TestHealthMonitor:
    def test_check_system_health_all_good(self, mock_psutil, mock_battery):
        result = check_system_health()
        assert result["cpu_ok"] is True
        assert result["memory_ok"] is True
        assert result["battery_ok"] is True
        assert result["overall_ok"] is True

    def test_check_system_health_high_cpu(self, mock_psutil, mock_battery):
        mock_psutil.cpu_percent.return_value = 95.0
        result = check_system_health()
        assert result["cpu_ok"] is False
        assert result["overall_ok"] is False

    def test_check_system_health_high_memory(self, mock_psutil, mock_battery):
        mock_psutil.virtual_memory.return_value.percent = 92.0
        result = check_system_health()
        assert result["memory_ok"] is False
        assert result["overall_ok"] is False

    def test_check_system_health_low_battery(self, mock_psutil, mock_battery):
        mock_battery.return_value = MagicMock(percent=10, secsleft=-1, power_plugged=False)
        result = check_system_health()
        assert result["battery_ok"] is False
        assert result["overall_ok"] is False

    def test_check_system_health_charging(self, mock_psutil, mock_battery):
        mock_battery.return_value = MagicMock(percent=20, secsleft=-1, power_plugged=True)
        result = check_system_health()
        assert result["battery_ok"] is True
        assert result["overall_ok"] is True

    def test_get_battery_status(self, mock_battery):
        status = get_battery_status()
        assert status["percent"] == 85
        assert status["power_plugged"] is False
        assert status["time_remaining"] == 18000

    def test_get_cpu_usage(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 30.5
        usage = get_cpu_usage()
        assert usage == 30.5

    def test_get_memory_usage(self, mock_psutil):
        usage = get_memory_usage()
        assert usage == 60.0

    def test_schedule_task_if_healthy_success(self, mock_db, mock_redis, mock_psutil, mock_battery):
        mock_db.get_task_priority.return_value = 5
        mock_redis.get.return_value = None  # not throttled
        result = schedule_task_if_healthy("task_123")
        assert result is True
        mock_db.log_health_check.assert_called_once()
        mock_redis.setex.assert_called_once()

    def test_schedule_task_if_healthy_unhealthy(self, mock_db, mock_redis, mock_psutil, mock_battery):
        mock_psutil.cpu_percent.return_value = 99.0
        result = schedule_task_if_healthy("task_456")
        assert result is False
        mock_db.log_health_check.assert_called_once()

    def test_schedule_task_if_healthy_throttled(self, mock_db, mock_redis, mock_psutil, mock_battery):
        mock_redis.get.return_value = b"1"
        result = schedule_task_if_healthy("task_789")
        assert result is False
        mock_redis.get.assert_called_once_with("health:throttle:task_789")

    def test_health_monitor_initialization(self):
        monitor = HealthMonitor()
        assert monitor is not None
        assert monitor.cpu_threshold == 80
        assert monitor.memory_threshold == 85
        assert monitor.battery_threshold == 20

    def test_health_monitor_check(self, mock_psutil, mock_battery):
        monitor = HealthMonitor()
        report = monitor.check()
        assert "cpu" in report
        assert "memory" in report
        assert "battery" in report
        assert report["overall"] is True

    def test_health_monitor_check_cpu_high(self, mock_psutil, mock_battery):
        mock_psutil.cpu_percent.return_value = 85.0
        monitor = HealthMonitor()
        report = monitor.check()
        assert report["cpu"] > monitor.cpu_threshold
        assert report["overall"] is False

    def test_health_monitor_check_memory_high(self, mock_psutil, mock_battery):
        mock_psutil.virtual_memory.return_value.percent = 90.0
        monitor = HealthMonitor()
        report = monitor.check()
        assert report["memory"] > monitor.memory_threshold
        assert report["overall"] is False

    def test_health_monitor_check_battery_low(self, mock_psutil, mock_battery):
        mock_battery.return_value = MagicMock(percent=15, secsleft=-1, power_plugged=False)
        monitor = HealthMonitor()
        report = monitor.check()
        assert report["battery"] < monitor.battery_threshold
        assert report["overall"] is False

    def test_health_monitor_check_battery_charging(self, mock_psutil, mock_battery):
        mock_battery.return_value = MagicMock(percent=10, secsleft=-1, power_plugged=True)
        monitor = HealthMonitor()
        report = monitor.check()
        assert report["battery"] >= monitor.battery_threshold
        assert report["overall"] is True