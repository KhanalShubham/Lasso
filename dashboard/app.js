/**
 * Academic Minimalist Dashboard Logic (dashboard/app.js)
 * Clean Black & White Styling with Green/Red Statistical Indicators.
 */

document.addEventListener("DOMContentLoaded", () => {
  const data = window.DASHBOARD_DATA;
  if (!data) {
    console.error("DASHBOARD_DATA not found. Make sure dashboard_data.js is loaded.");
    return;
  }

  // 1. Setup Academic Paper Tab Navigation
  const tabButtons = document.querySelectorAll(".paper-tab-btn");
  const viewPanels = document.querySelectorAll(".view-panel");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.getAttribute("data-view");
      tabButtons.forEach((b) => b.classList.remove("active"));
      viewPanels.forEach((p) => p.classList.remove("active-panel"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(targetView);
      if (targetPanel) {
        targetPanel.classList.add("active-panel");
      }

      window.dispatchEvent(new Event("resize"));
    });
  });

  // 2. Populate Executive Overview KPIs
  populateKPIs(data.kpis, data.targets);

  // 3. Render Overview Charts (Clean Academic Pie & Bar)
  renderOverviewCharts(data.targets, data.seasons);

  // 4. Setup Feature Distribution Tab
  setupFeatureDistribution(data.features);

  // 5. Render Correlation Heatmap & Collinear Pairs
  renderCorrelationMatrix(data.correlations, data.top_collinear_pairs);

  // 6. Setup Warm-Up & Missingness Table
  setupWarmUpView(data.seasons);

  // 7. Setup Head-to-Head Matchup Inspector
  setupMatchInspector(data.sample_fixtures, data.features);
});

/* =====================================================================
   POPULATE KPIS & OVERVIEW
   ===================================================================== */
function populateKPIs(kpis, targets) {
  const el = (id) => document.getElementById(id);
  if (el("kpi-total-matches")) el("kpi-total-matches").innerText = kpis.total_fixtures.toLocaleString();
  if (el("kpi-ready-matches")) el("kpi-ready-matches").innerText = kpis.modelling_ready.toLocaleString();
  if (el("kpi-warmup-matches")) el("kpi-warmup-matches").innerText = kpis.warmup_excluded.toLocaleString();
  if (el("kpi-features-count")) el("kpi-features-count").innerText = kpis.features_count;
  if (el("kpi-home-win-pct")) el("kpi-home-win-pct").innerText = targets.modelling_ready.home_win_pct + "%";
}

function renderOverviewCharts(targets, seasons) {
  // Chart A: Target Outcome Distribution (Clean Pie Chart)
  const ctxTarget = document.getElementById("chart-target-dist");
  if (ctxTarget) {
    new Chart(ctxTarget, {
      type: "pie",
      data: {
        labels: ["Home Win", "Draw", "Away Win"],
        datasets: [{
          data: [
            targets.modelling_ready.home_wins,
            targets.modelling_ready.draws,
            targets.modelling_ready.away_wins
          ],
          backgroundColor: ["#15803d", "#94a3b8", "#b91c1c"],
          borderColor: "#ffffff",
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#0f172a", font: { family: "Arial", size: 12, weight: "bold" } }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const total = targets.modelling_ready.home_wins + targets.modelling_ready.away_wins + targets.modelling_ready.draws;
                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} matches (${pct}%)`;
              }
            }
          }
        }
      }
    });
  }

  // Chart B: Seasonal Breakdown (Clean Stacked Bar)
  const ctxSeason = document.getElementById("chart-season-breakdown");
  if (ctxSeason) {
    const labels = seasons.map(s => s.season);
    const readyData = seasons.map(s => s.ready);
    const warmupData = seasons.map(s => s.warmup);

    new Chart(ctxSeason, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Modelling-Ready (k ≥ 10)",
            data: readyData,
            backgroundColor: "#0f172a",
            borderColor: "#0f172a",
            borderWidth: 1,
            borderRadius: 2
          },
          {
            label: "Warm-Up Excluded (Cold-Start)",
            data: warmupData,
            backgroundColor: "#cbd5e1",
            borderColor: "#94a3b8",
            borderWidth: 1,
            borderRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            grid: { display: false },
            ticks: { color: "#0f172a", font: { family: "Arial", weight: "bold" } }
          },
          y: {
            stacked: true,
            max: 420,
            grid: { color: "#e2e8f0" },
            ticks: { color: "#334155", font: { family: "JetBrains Mono" } }
          }
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#0f172a", font: { family: "Arial", size: 12 } }
          }
        }
      }
    });
  }
}

/* =====================================================================
   FEATURE DISTRIBUTION VIEW
   ===================================================================== */
let currentHistogramChart = null;

function setupFeatureDistribution(features) {
  const select = document.getElementById("select-feature-dist");
  if (!select) return;

  select.innerHTML = "";
  Object.keys(features).forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.innerText = `${features[key].title} (${key})`;
    select.appendChild(opt);
  });

  function updateDisplay(featureKey) {
    const feat = features[featureKey];
    if (!feat) return;

    document.getElementById("dist-title").innerText = feat.title;
    document.getElementById("dist-key").innerText = feat.key;
    document.getElementById("dist-category").innerText = feat.category;
    document.getElementById("dist-unit").innerText = feat.unit || "Normalized units";
    document.getElementById("dist-cluster").innerText = feat.cluster;
    document.getElementById("dist-desc").innerText = feat.description;

    const s = feat.stats;
    document.getElementById("stat-count").innerText = s.count.toLocaleString();
    document.getElementById("stat-mean").innerText = s.mean.toFixed(4);
    document.getElementById("stat-std").innerText = s.std.toFixed(4);
    document.getElementById("stat-median").innerText = s.median.toFixed(4);
    document.getElementById("stat-range").innerText = `[${s.min.toFixed(2)}, ${s.max.toFixed(2)}]`;
    document.getElementById("stat-iqr").innerText = `[${s.p25.toFixed(2)}, ${s.p75.toFixed(2)}]`;

    const ctx = document.getElementById("chart-feature-histogram");
    if (!ctx) return;

    if (currentHistogramChart) {
      currentHistogramChart.destroy();
    }

    currentHistogramChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: feat.histogram.labels,
        datasets: [{
          label: "Observation Count",
          data: feat.histogram.counts,
          backgroundColor: "#334155",
          borderColor: "#0f172a",
          borderWidth: 1,
          borderRadius: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: "#334155",
              font: { family: "JetBrains Mono", size: 9 },
              maxRotation: 45,
              minRotation: 45
            }
          },
          y: {
            grid: { color: "#e2e8f0" },
            ticks: { color: "#334155", font: { family: "JetBrains Mono" } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `Bin: ${items[0].label}`,
              label: (item) => ` Fixtures: ${item.parsed.y}`
            }
          }
        }
      }
    });
  }

  select.addEventListener("change", (e) => updateDisplay(e.target.value));
  updateDisplay("diff_sot_k5");
}

/* =====================================================================
   CORRELATION MATRIX & COLLINEARITY CLUSTERS
   ===================================================================== */
function renderCorrelationMatrix(corr, topPairs) {
  const container = document.getElementById("heatmap-grid-target");
  if (!container) return;

  container.innerHTML = "";
  const features = corr.features;
  const values = corr.values;

  // Header row
  const headerEmpty = document.createElement("div");
  headerEmpty.className = "hm-col-header";
  headerEmpty.style.height = "auto";
  headerEmpty.innerText = "";
  container.appendChild(headerEmpty);

  features.forEach((f) => {
    const colHeader = document.createElement("div");
    colHeader.className = "hm-col-header";
    colHeader.innerText = f.replace("diff_", "");
    container.appendChild(colHeader);
  });

  // Rows
  for (let r = 0; r < features.length; r++) {
    const rowLabel = document.createElement("div");
    rowLabel.className = "hm-row-label";
    rowLabel.innerText = features[r].replace("diff_", "");
    rowLabel.title = features[r];
    container.appendChild(rowLabel);

    for (let c = 0; c < features.length; c++) {
      const val = values[r][c];
      const cell = document.createElement("div");
      cell.className = "hm-cell";
      cell.innerText = val.toFixed(2);
      cell.title = `${features[r]} vs ${features[c]}: r = ${val.toFixed(3)}`;

      // Academic monochrome shading with green/red indicator tints
      if (r === c) {
        cell.style.background = "#0f172a";
        cell.style.color = "#ffffff";
      } else if (val > 0) {
        const intensity = Math.min(1, Math.abs(val));
        cell.style.background = `rgba(21, 128, 61, ${0.08 + intensity * 0.75})`;
        cell.style.color = intensity > 0.5 ? "#ffffff" : "#15803d";
      } else {
        const intensity = Math.min(1, Math.abs(val));
        cell.style.background = `rgba(185, 28, 28, ${0.08 + intensity * 0.75})`;
        cell.style.color = intensity > 0.5 ? "#ffffff" : "#b91c1c";
      }

      container.appendChild(cell);
    }
  }

  // Populate Top Collinear Pairs Table
  const tbody = document.getElementById("table-top-collinear-body");
  if (tbody) {
    tbody.innerHTML = "";
    topPairs.forEach((p) => {
      const tr = document.createElement("tr");
      const rFormatted = p.r > 0 ? `<span class="tag-pos">▲ +${p.r.toFixed(4)}</span>` : `<span class="tag-neg">▼ ${p.r.toFixed(4)}</span>`;
      tr.innerHTML = `
        <td style="font-weight:700;">${p.feat1_title} <span style="color:#64748b; font-weight:normal;">(${p.feat1})</span></td>
        <td style="font-weight:700;">${p.feat2_title} <span style="color:#64748b; font-weight:normal;">(${p.feat2})</span></td>
        <td>${rFormatted}</td>
        <td><span class="paper-badge" style="background:#fee2e2; color:#b91c1c; border-color:#b91c1c;">${p.strength}</span></td>
        <td style="color:#334155; font-size:12px;">High variance inflation; causes unstable MLE Hessian. Requires Ridge/Lasso.</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

/* =====================================================================
   WARM-UP AUDIT VIEW
   ===================================================================== */
function setupWarmUpView(seasons) {
  const tbody = document.getElementById("table-warmup-body");
  if (!tbody) return;

  tbody.innerHTML = "";
  seasons.forEach((s) => {
    const tr = document.createElement("tr");
    const readyPct = ((s.ready / s.total) * 100).toFixed(1);
    const warmupPct = ((s.warmup / s.total) * 100).toFixed(1);
    tr.innerHTML = `
      <td style="font-weight:700;">${s.season}</td>
      <td class="font-mono">${s.total}</td>
      <td class="font-mono" style="font-weight:700; color:#15803d;">${s.ready} (${readyPct}%)</td>
      <td class="font-mono" style="font-weight:700; color:#b91c1c;">${s.warmup} (${warmupPct}%)</td>
      <td class="font-mono">${s.home_win_pct}%</td>
      <td class="font-mono">${s.draw_pct}%</td>
      <td class="font-mono">${s.away_win_pct}%</td>
      <td class="font-mono" style="color:#15803d; font-weight:700;">0 (100% complete)</td>
    `;
    tbody.appendChild(tr);
  });
}

/* =====================================================================
   HEAD-TO-HEAD MATCHUP INSPECTOR
   ===================================================================== */
function setupMatchInspector(matches, featuresMeta) {
  const select = document.getElementById("select-inspector-match");
  if (!select || !matches || matches.length === 0) return;

  select.innerHTML = "";
  matches.forEach((m, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.innerText = `[${m.season}] ${m.date} — ${m.home_team} vs ${m.away_team} (${m.score})`;
    select.appendChild(opt);
  });

  function renderMatch(matchIdx) {
    const m = matches[matchIdx];
    if (!m) return;

    document.getElementById("insp-home-team").innerText = m.home_team;
    document.getElementById("insp-away-team").innerText = m.away_team;
    document.getElementById("insp-score").innerText = m.score;
    document.getElementById("insp-meta").innerText = `${m.season} • Kickoff on ${m.date} at ${m.kickoff_time}`;

    const pill = document.getElementById("insp-outcome-pill");
    pill.innerText = m.outcome;
    pill.className = "";
    if (m.target_class === 2) {
      pill.className = "badge-outcome-hw";
      pill.innerText = "Home Win";
    } else if (m.target_class === 1) {
      pill.className = "badge-outcome-d";
      pill.innerText = "Draw";
    } else {
      pill.className = "badge-outcome-aw";
      pill.innerText = "Away Win";
    }

    const container = document.getElementById("insp-diff-bars-container");
    container.innerHTML = "";

    const featKeys = Object.keys(m.features);
    featKeys.forEach((key) => {
      const val = m.features[key];
      const meta = featuresMeta[key] || {};
      const title = meta.title || key;

      const box = document.createElement("div");
      box.className = "diff-item-box";

      let valHtml = "";
      if (val > 0) {
        valHtml = `<span class="tag-pos">▲ +${val.toFixed(2)} (${m.home_team} Edge)</span>`;
      } else if (val < 0) {
        valHtml = `<span class="tag-neg">▼ ${val.toFixed(2)} (${m.away_team} Edge)</span>`;
      } else {
        valHtml = `<span class="tag-zero">■ 0.00 (Neutral)</span>`;
      }

      box.innerHTML = `
        <span class="diff-item-name">${title}</span>
        <span class="diff-item-val">${valHtml}</span>
      `;
      container.appendChild(box);
    });
  }

  select.addEventListener("change", (e) => renderMatch(e.target.value));
  renderMatch(0);
}
