import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.workers.scheduler import (
    analyze_periodically,
    refresh_model_periodically,
    start_scheduler,
    stop_scheduler,
    scheduler_instance,
    _analyze_task,
    _refresh_model_task
)


@pytest.fixture
def mock_db():
    with patch('src.workers.scheduler.db') as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch('src.workers.scheduler.redis_client') as mock:
        yield mock


@pytest.fixture
def mock_model_manager():
    with patch('src.workers.scheduler.model_manager') as mock:
        yield mock


@pytest.fixture
def mock_analyzer():
    with patch('src.workers.scheduler.analyzer') as mock:
        yield mock


@pytest.fixture
def mock_time():
    with patch('src.workers.scheduler.time') as mock:
        mock.sleep = MagicMock()
        yield mock


@pytest.fixture
def mock_thread():
    with patch('src.workers.scheduler.Thread') as mock:
        thread_instance = MagicMock()
        mock.return_value = thread_instance
        yield mock


def test_analyze_task_calls_analyzer(mock_db, mock_redis, mock_analyzer):
    _analyze_task()
    mock_analyzer.analyze.assert_called_once()


def test_analyze_task_updates_redis_timestamp(mock_db, mock_redis, mock_analyzer):
    _analyze_task()
    mock_redis.set.assert_called_once()
    assert mock_redis.set.call_args[0][0] == "last_analysis_timestamp"
    assert isinstance(mock_redis.set.call_args[0][1], str)


def test_refresh_model_task_calls_model_manager(mock_db, mock_redis, mock_model_manager):
    _refresh_model_task()
    mock_model_manager.refresh.assert_called_once()


def test_refresh_model_task_updates_redis_timestamp(mock_db, mock_redis, mock_model_manager):
    _refresh_model_task()
    mock_redis.set.assert_called_once()
    assert mock_redis.set.call_args[0][0] == "last_model_refresh_timestamp"
    assert isinstance(mock_redis.set.call_args[0][1], str)


def test_analyze_periodically_runs_once(mock_db, mock_redis, mock_analyzer, mock_time):
    mock_time.time.return_value = 1000.0
    analyze_periodically(interval=10, stop_flag=MagicMock(side_effect=[False, True]))
    mock_analyzer.analyze.assert_called_once()


def test_analyze_periodically_respects_interval(mock_db, mock_redis, mock_analyzer, mock_time):
    mock_time.time.return_value = 1000.0
    mock_time.sleep.side_effect = [None, None, KeyboardInterrupt]
    stop_flag = MagicMock(side_effect=[False, False, True])
    with pytest.raises(KeyboardInterrupt):
        analyze_periodically(interval=10, stop_flag=stop_flag)
    assert mock_time.sleep.call_count == 2
    assert mock_time.sleep.call_args_list[0][0][0] == 10.0
    assert mock_time.sleep.call_args_list[1][0][0] == 10.0


def test_refresh_model_periodically_runs_once(mock_db, mock_redis, mock_model_manager, mock_time):
    mock_time.time.return_value = 1000.0
    refresh_model_periodically(interval=3600, stop_flag=MagicMock(side_effect=[False, True]))
    mock_model_manager.refresh.assert_called_once()


def test_refresh_model_periodically_respects_interval(mock_db, mock_redis, mock_model_manager, mock_time):
    mock_time.time.return_value = 1000.0
    mock_time.sleep.side_effect = [None, None, KeyboardInterrupt]
    stop_flag = MagicMock(side_effect=[False, False, True])
    with pytest.raises(KeyboardInterrupt):
        refresh_model_periodically(interval=3600, stop_flag=stop_flag)
    assert mock_time.sleep.call_count == 2
    assert mock_time.sleep.call_args_list[0][0][0] == 3600.0
    assert mock_time.sleep.call_args_list[1][0][0] == 3600.0


def test_start_scheduler_creates_threads(mock_db, mock_redis, mock_analyzer, mock_model_manager, mock_thread):
    start_scheduler()
    assert mock_thread.call_count == 2
    thread_args = mock_thread.call_args_list
    assert any(args[1]['target'] == analyze_periodically for args in thread_args)
    assert any(args[1]['target'] == refresh_model_periodically for args in thread_args)


def test_start_scheduler_sets_stop_flag(mock_db, mock_redis, mock_analyzer, mock_model_manager, mock_thread):
    start_scheduler()
    assert scheduler_instance.stop_flag is not None
    assert isinstance(scheduler_instance.stop_flag, MagicMock.__class__)


def test_stop_scheduler_sets_stop_flag(mock_db, mock_redis, mock_analyzer, mock_model_manager):
    scheduler_instance.stop_flag = MagicMock()
    stop_scheduler()
    scheduler_instance.stop_flag.assert_called_once_with()


def test_scheduler_instance_exists():
    assert scheduler_instance is not None
    assert hasattr(scheduler_instance, 'stop_flag')