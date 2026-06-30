/**
 * MindGuard Dashboard Frontend
 * Calls backend endpoints at /api/v1/*
 */

const API_BASE = '/api/v1';

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function setStatus(healthy) {
  const el = $('#systemStatus');
  if (healthy) {
    el.className = 'pill pill-ok';
    el.innerHTML = '<span class="dot"></span> System Healthy';
  } else {
    el.className = 'pill pill-bad';
    el.innerHTML = '<span class="dot"></span> Degraded';
  }
}

async function refreshHealth() {
  try {
    const data = await api('/health');
    $('#envValue').textContent = data.env || '-';
    $('#apiValue').textContent = 'OK';
    $('#modelValue').textContent = data.model_loaded ? 'Loaded' : 'Not Loaded';
    $('#modelValue').className = 'metric-value ' + (data.model_loaded ? 'text-success' : 'text-warn');
    $('#modelVersion').textContent = data.model_version || '-';
    $('#diskValue').textContent = `${(data.disk_usage_percent || 0).toFixed(1)}%`;
    $('#memValue').textContent = `${(data.memory_usage_percent || 0).toFixed(1)}%`;
    setStatus(data.status === 'healthy');
  } catch (e) {
    setStatus(false);
    $('#apiValue').textContent = 'Error';
  }
}

async function submitAssessment() {
  const resultEl = $('#assessmentResult');
  resultEl.textContent = 'Analyzing signals...';
  const sleep = parseFloat($('#inputSleep').value) || 0.5;
  const social = parseFloat($('#inputSocial').value) || 0.5;
  const typing = parseFloat($('#inputTyping').value) || 0.5;

  const payload = {
    user_id: 'dashboard-demo',
    signals: [
      { type: 'sleep_quality', value: sleep, timestamp: new Date().toISOString() },
      { type: 'social_engagement', value: social, timestamp: new Date().toISOString() },
      { type: 'typing_velocity', value: typing, timestamp: new Date().toISOString() },
    ],
  };

  try {
    const data = await api('/predict', { method: 'POST', body: JSON.stringify(payload) });
    resultEl.textContent = JSON.stringify(data, null, 2);
    updateSeverityBadge(data.severity);
  } catch (e) {
    resultEl.textContent = `Error: ${e.message}`;
  }
}

function updateSeverityBadge(severity) {
  const badge = $('#severityBadge');
  const map = {
    low: ['pill-ok', 'Low Risk'],
    medium: ['pill-warn', 'Medium Risk'],
    high: ['pill-bad', 'High Risk'],
  };
  const [cls, label] = map[severity] || ['pill-warn', severity];
  badge.className = `pill ${cls}`;
  badge.innerHTML = `<span class="dot"></span> ${label}`;
}

async function loadAlerts() {
  const list = $('#alertsList');
  list.innerHTML = '<li class="empty-state">Loading alerts...</li>';
  try {
    const data = await api('/alerts');
    const alerts = data.alerts || [];
    if (!alerts.length) {
      list.innerHTML = '<li class="empty-state">No active alerts. All signals nominal.</li>';
      return;
    }
    list.innerHTML = alerts.map(a => `
      <li class="alert-item severity-${a.severity}">
        <span class="alert-icon">${a.severity === 'high' ? '🚨' : a.severity === 'medium' ? '⚠️' : '✅'}</span>
        <div>
          <p class="alert-title">${escapeHtml(a.message)}</p>
          <p class="alert-meta">${a.id} · ${new Date(a.created_at).toLocaleString()}</p>
        </div>
      </li>
    `).join('');
  } catch (e) {
    list.innerHTML = `<li class="empty-state">Could not load alerts: ${e.message}</li>`;
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function init() {
  refreshHealth();
  setInterval(refreshHealth, 10000);

  $('#btnAssess').addEventListener('click', submitAssessment);
  $('#btnAlerts').addEventListener('click', loadAlerts);

  // Initial alert load
  loadAlerts();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
