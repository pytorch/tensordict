#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate a static HTML dashboard for flaky test visualization.

This script creates an index.html file with:
- Summary statistics
- Interactive chart showing flaky test trends over time
- Table of current flaky tests with details
"""

import argparse
from pathlib import Path

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TensorDict Flaky Test Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-primary: #0d1117;
      --bg-secondary: #161b22;
      --bg-tertiary: #21262d;
      --text-primary: #c9d1d9;
      --text-secondary: #8b949e;
      --text-link: #58a6ff;
      --border-color: #30363d;
      --success-color: #3fb950;
      --warning-color: #d29922;
      --danger-color: #f85149;
      --info-color: #58a6ff;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
      padding: 20px;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 30px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-color);
    }

    h1 {
      font-size: 2rem;
      margin-bottom: 10px;
    }

    .subtitle {
      color: var(--text-secondary);
      font-size: 0.9rem;
    }

    .summary-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 30px;
    }

    .card {
      background-color: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 20px;
      text-align: center;
    }

    .card-value {
      font-size: 2.5rem;
      font-weight: bold;
      margin-bottom: 5px;
    }

    .card-label {
      color: var(--text-secondary);
      font-size: 0.9rem;
    }

    .card-value.success { color: var(--success-color); }
    .card-value.warning { color: var(--warning-color); }
    .card-value.danger { color: var(--danger-color); }
    .card-value.info { color: var(--info-color); }

    .section {
      background-color: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 30px;
    }

    .section-title {
      font-size: 1.2rem;
      margin-bottom: 15px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
    }

    .chart-container {
      position: relative;
      height: 300px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
    }

    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }

    th {
      background-color: var(--bg-tertiary);
      font-weight: 600;
      color: var(--text-secondary);
    }

    tr:hover {
      background-color: var(--bg-tertiary);
    }

    .test-name {
      font-family: monospace;
      font-size: 0.85rem;
      word-break: break-all;
    }

    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
    }

    .badge-new {
      background-color: var(--info-color);
      color: white;
    }

    .badge-high {
      background-color: var(--danger-color);
      color: white;
    }

    .badge-medium {
      background-color: var(--warning-color);
      color: black;
    }

    .badge-low {
      background-color: var(--success-color);
      color: white;
    }

    .progress-bar {
      height: 8px;
      background-color: var(--bg-tertiary);
      border-radius: 4px;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      transition: width 0.3s ease;
    }

    .empty-state {
      text-align: center;
      padding: 40px;
      color: var(--text-secondary);
    }

    .empty-state-icon {
      font-size: 3rem;
      margin-bottom: 10px;
    }

    footer {
      text-align: center;
      padding-top: 20px;
      border-top: 1px solid var(--border-color);
      color: var(--text-secondary);
      font-size: 0.85rem;
    }

    a {
      color: var(--text-link);
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    @media (max-width: 768px) {
      .summary-cards {
        grid-template-columns: repeat(2, 1fr);
      }

      table {
        font-size: 0.85rem;
      }

      th, td {
        padding: 8px;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>TensorDict Flaky Test Dashboard</h1>
      <p class="subtitle">Last updated: <span id="last-updated">Loading...</span></p>
    </header>

    <div class="summary-cards">
      <div class="card">
        <div class="card-value" id="flaky-count">-</div>
        <div class="card-label">Flaky Tests</div>
      </div>
      <div class="card">
        <div class="card-value info" id="new-flaky-count">-</div>
        <div class="card-label">Newly Flaky (7 days)</div>
      </div>
      <div class="card">
        <div class="card-value success" id="resolved-count">-</div>
        <div class="card-label">Resolved</div>
      </div>
      <div class="card">
        <div class="card-value" id="runs-analyzed">-</div>
        <div class="card-label">CI Runs Analyzed</div>
      </div>
    </div>

    <div class="section">
      <h2 class="section-title">Trend (Last 30 Days)</h2>
      <div class="chart-container">
        <canvas id="trend-chart"></canvas>
      </div>
    </div>

    <div class="section">
      <h2 class="section-title">Flaky Tests</h2>
      <div id="flaky-tests-table">
        <div class="empty-state">
          <div class="empty-state-icon">Loading data...</div>
        </div>
      </div>
    </div>

    <footer>
      <p>
        <a href="https://github.com/pytorch/tensordict">TensorDict GitHub</a> |
        <a href="flaky-tests.json">Raw JSON</a> |
        <a href="flaky-tests.md">Markdown Report</a>
      </p>
      <p>
        Flaky tests are identified by analyzing failure patterns across recent CI runs.
        A test is considered flaky if it fails intermittently (5-95% failure rate) with at least 2 failures.
      </p>
    </footer>
  </div>

  <script>
    async function loadData() {
      try {
        const reportResponse = await fetch('flaky-tests.json');
        const report = await reportResponse.json();

        document.getElementById('last-updated').textContent = new Date(report.generated_at).toLocaleString();

        const flakyCount = report.summary.flaky_count;
        const flakyCountEl = document.getElementById('flaky-count');
        flakyCountEl.textContent = flakyCount;
        flakyCountEl.className = 'card-value ' + (flakyCount === 0 ? 'success' : flakyCount <= 5 ? 'warning' : 'danger');

        document.getElementById('new-flaky-count').textContent = report.summary.new_flaky_count || 0;
        document.getElementById('resolved-count').textContent = report.summary.resolved_count || 0;
        document.getElementById('runs-analyzed').textContent = report.analysis_period.runs_analyzed;

        renderFlakyTestsTable(report.flaky_tests);

        try {
          const historyResponse = await fetch('data.json');
          const history = await historyResponse.json();
          renderTrendChart(history.history);
        } catch (e) {
          renderTrendChart([{
            date: new Date().toISOString().split('T')[0],
            flaky_count: flakyCount
          }]);
        }

      } catch (error) {
        console.error('Failed to load data:', error);
        document.getElementById('flaky-tests-table').innerHTML =
          '<div class="empty-state"><div class="empty-state-icon">Failed to load data.</div></div>';
      }
    }

    function renderFlakyTestsTable(flakyTests) {
      const container = document.getElementById('flaky-tests-table');

      if (!flakyTests || flakyTests.length === 0) {
        container.innerHTML =
          '<div class="empty-state"><div class="empty-state-icon">No flaky tests detected!</div></div>';
        return;
      }

      const rows = flakyTests.map(test => {
        const failureRate = (test.failure_rate * 100).toFixed(1);
        const badgeClass = test.flaky_score >= 0.7 ? 'badge-high' : test.flaky_score >= 0.4 ? 'badge-medium' : 'badge-low';
        const lastFailed = test.recent_failures && test.recent_failures.length > 0
          ? test.recent_failures[test.recent_failures.length - 1].split('T')[0]
          : 'N/A';

        return '<tr>' +
          '<td class="test-name">' + escapeHtml(test.nodeid) + '</td>' +
          '<td>' + failureRate + '% (' + test.failures + '/' + test.executions + ')' +
          '<div class="progress-bar"><div class="progress-fill ' + badgeClass + '" style="width:' + failureRate + '%"></div></div></td>' +
          '<td>' + test.flaky_score.toFixed(2) + '</td>' +
          '<td>' + lastFailed + '</td>' +
          '<td>' + (test.is_new ? '<span class="badge badge-new">NEW</span>' : '') + '</td>' +
          '</tr>';
      }).join('');

      container.innerHTML =
        '<table><thead><tr><th>Test</th><th>Failure Rate</th><th>Flaky Score</th><th>Last Failed</th><th>Status</th></tr></thead>' +
        '<tbody>' + rows + '</tbody></table>';
    }

    function renderTrendChart(history) {
      const ctx = document.getElementById('trend-chart').getContext('2d');

      new Chart(ctx, {
        type: 'line',
        data: {
          labels: history.map(h => h.date),
          datasets: [{
            label: 'Flaky Tests',
            data: history.map(h => h.flaky_count),
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { mode: 'index', intersect: false }
          },
          scales: {
            x: {
              grid: { color: '#30363d' },
              ticks: { color: '#8b949e' }
            },
            y: {
              beginAtZero: true,
              grid: { color: '#30363d' },
              ticks: { color: '#8b949e', stepSize: 1 }
            }
          }
        }
      });
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    loadData();
  </script>
</body>
</html>
"""


def generate_dashboard(input_dir: Path, output_dir: Path) -> None:
    """Generate the dashboard HTML file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "index.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(DASHBOARD_HTML)

    print(f"Dashboard written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate flaky test dashboard")
    parser.add_argument(
        "--input-dir", default="flaky-reports", help="Input directory with reports"
    )
    parser.add_argument(
        "--output-dir", default="flaky-reports", help="Output directory for dashboard"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not (input_dir / "flaky-tests.json").exists():
        print(f"Warning: {input_dir / 'flaky-tests.json'} not found")

    generate_dashboard(input_dir, output_dir)

    print("Dashboard generation complete!")


if __name__ == "__main__":
    main()
