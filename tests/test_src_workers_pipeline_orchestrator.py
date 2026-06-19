import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.workers.pipeline_orchestrator import PipelineOrchestrator, run_pipeline, process_batch


@pytest.fixture
def mock_db():
    with patch('src.workers.pipeline_orchestrator.AsyncDatabase') as mock:
        instance = AsyncMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_redis():
    with patch('src.workers.pipeline_orchestrator.RedisClient') as mock:
        instance = MagicMock()
        instance.get = AsyncMock()
        instance.set = AsyncMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def orchestrator(mock_db, mock_redis):
    return PipelineOrchestrator(db=mock_db, redis=mock_redis)


@pytest.mark.asyncio
async def test_orchestrator_init(orchestrator, mock_db, mock_redis):
    assert orchestrator.db is mock_db
    assert orchestrator.redis is mock_redis


@pytest.mark.asyncio
async def test_orchestrator_fetch_data_success(orchestrator, mock_db):
    mock_db.fetch.side_effect = lambda table, limit: [{'id': 1, 'data': 'test'}]
    result = await orchestrator.fetch_data('users', limit=10)
    assert result == [{'id': 1, 'data': 'test'}]
    mock_db.fetch.assert_awaited_once_with('users', limit=10)


@pytest.mark.asyncio
async def test_orchestrator_fetch_data_empty(orchestrator, mock_db):
    mock_db.fetch.return_value = []
    result = await orchestrator.fetch_data('logs', limit=5)
    assert result == []
    mock_db.fetch.assert_awaited_once_with('logs', limit=5)


@pytest.mark.asyncio
async def test_orchestrator_transform_data(orchestrator):
    data = [{'id': 1, 'value': '100'}, {'id': 2, 'value': '200'}]
    result = await orchestrator.transform_data(data, lambda x: {**x, 'value': int(x['value']) * 2})
    assert result == [{'id': 1, 'value': 200}, {'id': 2, 'value': 400}]


@pytest.mark.asyncio
async def test_orchestrator_store_data_success(orchestrator, mock_db):
    data = [{'id': 1, 'processed': True}]
    await orchestrator.store_data(data, 'results')
    mock_db.insert_many.assert_awaited_once_with('results', data)


@pytest.mark.asyncio
async def test_orchestrator_store_data_empty(orchestrator, mock_db):
    await orchestrator.store_data([], 'results')
    mock_db.insert_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_update_status_success(orchestrator, mock_redis):
    await orchestrator.update_status('pipeline_123', 'running')
    mock_redis.set.assert_called_once_with('pipeline_status:pipeline_123', 'running')


@pytest.mark.asyncio
async def test_orchestrator_get_status_success(orchestrator, mock_redis):
    mock_redis.get.return_value = b'completed'
    status = await orchestrator.get_status('pipeline_456')
    assert status == 'completed'
    mock_redis.get.assert_awaited_once_with('pipeline_status:pipeline_456')


@pytest.mark.asyncio
async def test_orchestrator_get_status_missing(orchestrator, mock_redis):
    mock_redis.get.return_value = None
    status = await orchestrator.get_status('unknown')
    assert status is None


@pytest.mark.asyncio
async def test_run_pipeline_success(orchestrator, mock_db, mock_redis):
    mock_db.fetch.return_value = [{'id': 1, 'data': 'raw'}]
    mock_redis.get.return_value = b'pending'

    result = await run_pipeline(orchestrator, 'users', 'results', transform_fn=lambda x: x)

    assert result == [{'id': 1, 'data': 'raw'}]
    mock_db.fetch.assert_awaited_once()
    mock_db.insert_many.assert_awaited_once_with('results', [{'id': 1, 'data': 'raw'}])


@pytest.mark.asyncio
async def test_run_pipeline_no_data(orchestrator, mock_db, mock_redis):
    mock_db.fetch.return_value = []
    result = await run_pipeline(orchestrator, 'empty_table', 'output', transform_fn=lambda x: x)
    assert result == []
    mock_db.insert_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_batch_success(orchestrator, mock_db, mock_redis):
    batch = [{'id': 1}, {'id': 2}]
    mock_db.insert_many.return_value = None
    result = await process_batch(orchestrator, batch, 'output_table')
    assert result == len(batch)
    mock_db.insert_many.assert_awaited_once_with('output_table', batch)


@pytest.mark.asyncio
async def test_process_batch_empty(orchestrator, mock_db):
    result = await process_batch(orchestrator, [], 'output_table')
    assert result == 0
    mock_db.insert_many.assert_not_awaited()