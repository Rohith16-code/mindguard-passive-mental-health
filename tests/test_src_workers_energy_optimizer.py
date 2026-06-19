import pytest
from unittest.mock import MagicMock, patch
from src.workers.energy_optimizer import (
    EnergyOptimizer,
    optimize_cpu_throttle,
    optimize_gpu_throttle,
    get_battery_level,
    apply_throttling,
    restore_normal_performance,
)


@pytest.fixture
def mock_db():
    with patch("src.workers.energy_optimizer.db") as mock:
        mock.get_battery_level.return_value = 85
        mock.get_system_load.return_value = {"cpu": 45, "gpu": 30}
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.workers.energy_optimizer.redis_client") as mock:
        mock.get.return_value = None
        yield mock


@pytest.fixture
def optimizer(mock_db, mock_redis):
    return EnergyOptimizer(db=mock_db, redis=mock_redis)


def test_energy_optimizer_initialization(optimizer, mock_db, mock_redis):
    assert optimizer.db is mock_db
    assert optimizer.redis is mock_redis
    assert optimizer.min_cpu_freq == 800
    assert optimizer.max_cpu_freq == 2400
    assert optimizer.min_gpu_freq == 200
    assert optimizer.max_gpu_freq == 1200


def test_get_battery_level(optimizer, mock_db):
    level = optimizer.get_battery_level()
    assert level == 85
    mock_db.get_battery_level.assert_called_once()


def test_optimize_cpu_throttle_low_battery_high_load(optimizer, mock_db):
    mock_db.get_battery_level.return_value = 15
    mock_db.get_system_load.return_value = {"cpu": 90, "gpu": 70}
    
    throttle_cmd = optimize_cpu_throttle(optimizer)
    
    assert throttle_cmd["action"] == "throttle"
    assert throttle_cmd["target_freq"] <= optimizer.min_cpu_freq
    assert throttle_cmd["reason"] == "low_battery_high_load"


def test_optimize_cpu_throttle_high_battery_low_load(optimizer, mock_db):
    mock_db.get_battery_level.return_value = 95
    mock_db.get_system_load.return_value = {"cpu": 10, "gpu": 5}
    
    throttle_cmd = optimize_cpu_throttle(optimizer)
    
    assert throttle_cmd["action"] == "boost"
    assert throttle_cmd["target_freq"] >= optimizer.max_cpu_freq * 0.9
    assert throttle_cmd["reason"] == "high_battery_low_load"


def test_optimize_gpu_throttle_low_battery_high_load(optimizer, mock_db):
    mock_db.get_battery_level.return_value = 10
    mock_db.get_system_load.return_value = {"cpu": 30, "gpu": 95}
    
    throttle_cmd = optimize_gpu_throttle(optimizer)
    
    assert throttle_cmd["action"] == "throttle"
    assert throttle_cmd["target_freq"] <= optimizer.min_gpu_freq
    assert throttle_cmd["reason"] == "low_battery_high_gpu_load"


def test_optimize_gpu_throttle_high_battery_low_load(optimizer, mock_db):
    mock_db.get_battery_level.return_value = 90
    mock_db.get_system_load.return_value = {"cpu": 20, "gpu": 15}
    
    throttle_cmd = optimize_gpu_throttle(optimizer)
    
    assert throttle_cmd["action"] == "boost"
    assert throttle_cmd["target_freq"] >= optimizer.max_gpu_freq * 0.9
    assert throttle_cmd["reason"] == "high_battery_low_gpu_load"


def test_apply_throttling(optimizer, mock_db):
    cmd = {"cpu_freq": 800, "gpu_freq": 200}
    
    with patch("src.workers.energy_optimizer.os") as mock_os:
        apply_throttling(optimizer, cmd)
        
        mock_os.cpu_freq_set.assert_called_once_with(800)
        mock_os.gpu_freq_set.assert_called_once_with(200)


def test_restore_normal_performance(optimizer, mock_db):
    with patch("src.workers.energy_optimizer.os") as mock_os:
        restore_normal_performance(optimizer)
        
        mock_os.cpu_freq_set.assert_called_once_with(optimizer.max_cpu_freq)
        mock_os.gpu_freq_set.assert_called_once_with(optimizer.max_gpu_freq)


def test_full_optimization_cycle(optimizer, mock_db):
    mock_db.get_battery_level.return_value = 25
    mock_db.get_system_load.return_value = {"cpu": 60, "gpu": 50}
    
    cpu_cmd = optimize_cpu_throttle(optimizer)
    gpu_cmd = optimize_gpu_throttle(optimizer)
    
    assert cpu_cmd["action"] == "throttle"
    assert gpu_cmd["action"] == "throttle"
    
    with patch("src.workers.energy_optimizer.os") as mock_os:
        apply_throttling(optimizer, cpu_cmd)
        apply_throttling(optimizer, gpu_cmd)
        
        assert mock_os.cpu_freq_set.call_count == 1
        assert mock_os.gpu_freq_set.call_count == 1


def test_energy_optimizer_handles_missing_db(mock_redis):
    optimizer = EnergyOptimizer(db=None, redis=mock_redis)
    
    with pytest.raises(ValueError):
        optimizer.get_battery_level()


def test_energy_optimizer_handles_missing_redis(optimizer, mock_db):
    optimizer.redis = None
    
    with patch("src.workers.energy_optimizer.os") as mock_os:
        restore_normal_performance(optimizer)
        mock_os.cpu_freq_set.assert_called_once()
        mock_os.gpu_freq_set.assert_called_once()