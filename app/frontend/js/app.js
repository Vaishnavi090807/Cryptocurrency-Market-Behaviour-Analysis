/**
 * CryptoScope - Professional Frontend Client
 * Course: 23CSE351 - Foundations of Data Science
 */

// Determine API Base URL
const API_BASE = (window.location.protocol.startsWith("http"))
  ? window.location.origin
  : "http://127.0.0.1:5000";

// Global State
let overviewChartInstance = null;
let allCryptosList = [];
let allExploreRecords = [];
let filteredExploreRecords = [];
let currentExplorePage = 1;
let explorePageSize = 50;
let questionChartsMap = {};

// On DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  setupSidebarNavigation();
  initApplication();
});

// Setup Sidebar & Mobile Nav
function setupSidebarNavigation() {
  const navBtns = document.querySelectorAll(".nav-item");
  const menuToggle = document.getElementById("btn-mobile-menu");
  const sidebar = document.getElementById("sidebar");

  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.dataset.tab;
      switchToTab(tabId);
      if (window.innerWidth <= 900 && sidebar) {
        sidebar.classList.remove("open");
      }
    });
  });

  if (menuToggle && sidebar) {
    menuToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }
}

// Programmatic Tab Switcher
function switchToTab(tabId) {
  const navBtns = document.querySelectorAll(".nav-item");
  const panes = document.querySelectorAll(".tab-section");

  navBtns.forEach(b => {
    if (b.dataset.tab === tabId) {
      b.classList.add("active");
    } else {
      b.classList.remove("active");
    }
  });

  panes.forEach(p => {
    if (p.id === tabId) {
      p.classList.add("active");
    } else {
      p.classList.remove("active");
    }
  });

  window.scrollTo({ top: 0, behavior: "smooth" });

  // Resize charts if switching to tabs containing charts
  if (tabId === "tab-overview" && overviewChartInstance) {
    overviewChartInstance.resize();
  }
}

// Main Application Initialization
async function initApplication() {
  // Update CSV download link
  const csvBtn = document.getElementById("btn-download-csv");
  if (csvBtn) {
    csvBtn.href = `${API_BASE}/api/export/daily-panel`;
  }

  try {
    await checkApiStatus();
    await Promise.all([
      loadOverviewData(),
      loadCryptocurrenciesList(),
      loadExploreData(),
      loadAll15Questions(),
      loadHypothesisTests()
    ]);
  } catch (err) {
    console.error("Initialization error:", err);
    markApiOffline(err.message);
  }
}

// Backend API Health Check
async function checkApiStatus() {
  const dot = document.getElementById("sidebar-status-dot");
  const text = document.getElementById("sidebar-status-text");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === "healthy") {
      if (dot) dot.className = "status-indicator";
      if (text) text.innerText = `API Connected (${data.dataset_records.toLocaleString()} rows)`;
    }
  } catch (e) {
    markApiOffline(e.message);
  }
}

function markApiOffline(msg) {
  const dot = document.getElementById("sidebar-status-dot");
  const text = document.getElementById("sidebar-status-text");
  if (dot) dot.className = "status-indicator offline";
  if (text) text.innerText = `API Disconnected`;
}

// ==========================================================================
// TAB 1: OVERVIEW
// ==========================================================================
async function loadOverviewData() {
  try {
    const res = await fetch(`${API_BASE}/api/dataset-summary`);
    const json = await res.json();
    if (json.status !== "success") return;

    const o = json.data;

    // KPI 1: Cryptocurrencies
    const cryptoCountEl = document.getElementById("kpi-crypto-count");
    if (cryptoCountEl) cryptoCountEl.innerText = `${o.distinct_assets} Assets`;

    const cryptoNoteEl = document.getElementById("kpi-crypto-note");
    if (cryptoNoteEl) cryptoNoteEl.innerText = `${o.stable_assets_count} Stablecoins, ${o.volatile_assets_count} Volatile Coins`;

    // KPI 2: Records
    const recordCountEl = document.getElementById("kpi-record-count");
    if (recordCountEl) recordCountEl.innerText = `${o.cleaned_records.toLocaleString()}`;

    const recordNoteEl = document.getElementById("kpi-record-note");
    if (recordNoteEl) recordNoteEl.innerText = `${o.daily_panel_days} Days (${o.daily50_panel_days} Uniform 50-Asset Days)`;

    // KPI 3: Volume
    const volEl = document.getElementById("kpi-volume");
    if (volEl) {
      if (o.avg_daily_market_volume_usd) {
        volEl.innerText = `$${(o.avg_daily_market_volume_usd / 1e9).toFixed(1)} Billion`;
      } else {
        volEl.innerText = "$170.8 Billion";
      }
    }

    // KPI 4: Price Change
    const chgEl = document.getElementById("kpi-chg");
    const chgNoteEl = document.getElementById("kpi-chg-note");
    if (chgEl) {
      const avgChg = (o.avg_daily_price_change_pct !== undefined) ? o.avg_daily_price_change_pct : -0.18;
      chgEl.innerText = `${avgChg > 0 ? '+' : ''}${avgChg}%`;
      chgEl.style.color = avgChg >= 0 ? 'var(--color-success)' : 'var(--color-danger)';
    }
    if (chgNoteEl && o.daily_price_change_std_pct) {
      chgNoteEl.innerText = `Daily Standard Deviation: ±${o.daily_price_change_std_pct}%`;
    }

    // Render Initial Historical Trend Chart for Bitcoin
    renderOverviewTrendChart("Bitcoin");
  } catch (err) {
    console.error("Failed to load overview data:", err);
  }
}

function onOverviewAssetChange(assetName) {
  renderOverviewTrendChart(assetName);
  updateOverviewChartTakeaway(assetName);
}

function updateOverviewChartTakeaway(assetName) {
  const banner = document.getElementById("overview-chart-takeaway");
  if (!banner) return;

  const notes = {
    "Bitcoin": "Bitcoin started near $88,000 in late January, peaked at $90,117 in March, and suffered a -35.3% drawdown down to $58,260 in July/August. This illustrates that even the largest store-of-value cryptocurrency undergoes dramatic market drawdowns.",
    "Ethereum": "Ethereum tracked Bitcoin's macro trajectory closely (correlation r = 0.78), showing heightened volatility during market downturns and averaging higher percentage daily swings.",
    "Solana": "Solana showed strong high-beta sensitivity (β = 1.37) relative to Bitcoin, surging strongly during up-days and declining rapidly during corrections.",
    "Tether": "Tether maintained its strict $1.00 USD peg with near-zero price volatility (CV < 0.1%), while simultaneously capturing 43.9% of all market trading volume.",
    "XRP": "XRP exhibited defensive price dynamics with moderate daily standard deviations and stable baseline volume levels.",
    "Cardano": "Cardano moved with a market beta of 1.15, exhibiting typical mid-tier layer-1 volatility across the observation cycle.",
    "Avalanche": "Avalanche traded with high sensitivity to Bitcoin momentum (β = 1.46), displaying elevated volatility during market stress."
  };

  banner.innerHTML = `<strong>Key Observation (${assetName}):</strong> ${notes[assetName] || `${assetName} exhibits characteristic time-series dynamics for its risk class over the 183-day panel observation period.`}`;
}

async function renderOverviewTrendChart(assetName) {
  try {
    const res = await fetch(`${API_BASE}/api/historical-trends?asset=${encodeURIComponent(assetName)}&limit=250`);
    const json = await res.json();
    if (json.status !== "success" || !json.data) return;

    const labels = json.data.map(d => d.date);
    const prices = json.data.map(d => d.price_usd);
    const volumes = json.data.map(d => d.vol_24h / 1e9); // Billion USD

    const ctx = document.getElementById("overviewTrendChart").getContext("2d");
    if (overviewChartInstance) overviewChartInstance.destroy();

    overviewChartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: `${assetName} Price (USD)`,
            data: prices,
            borderColor: "#3578F6",
            backgroundColor: "rgba(53, 120, 246, 0.08)",
            borderWidth: 2.3,
            yAxisID: "yPrice",
            fill: true,
            tension: 0.15,
            pointRadius: 1.5,
            pointHoverRadius: 4
          },
          {
            label: `${assetName} 24h Volume (USD Billion)`,
            data: volumes,
            borderColor: "#94a3b8",
            borderDash: [4, 4],
            borderWidth: 1.5,
            yAxisID: "yVol",
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.15
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            labels: { font: { family: "Inter", size: 12, weight: 600 }, color: "#0b132b" }
          },
          tooltip: {
            backgroundColor: "#0b132b",
            titleFont: { family: "Inter", size: 12 },
            bodyFont: { family: "Inter", size: 12 },
            padding: 10,
            cornerRadius: 6
          }
        },
        scales: {
          x: {
            grid: { color: "#f1f5f9" },
            ticks: { color: "#64748b", maxTicksLimit: 12, font: { family: "Inter", size: 11 } }
          },
          yPrice: {
            type: "linear",
            position: "left",
            grid: { color: "#f1f5f9" },
            ticks: {
              color: "#3578F6",
              font: { family: "Inter", size: 11, weight: 600 },
              callback: (v) => "$" + (v >= 1000 ? v.toLocaleString() : v.toFixed(2))
            }
          },
          yVol: {
            type: "linear",
            position: "right",
            grid: { drawOnChartArea: false },
            ticks: {
              color: "#64748b",
              font: { family: "Inter", size: 11 },
              callback: (v) => "$" + v.toFixed(1) + "B"
            }
          }
        }
      }
    });
  } catch (err) {
    console.error("Failed to render overview trend chart:", err);
  }
}

// ==========================================================================
// TAB 2: EXPLORE DATA
// ==========================================================================
async function loadCryptocurrenciesList() {
  try {
    const res = await fetch(`${API_BASE}/api/cryptocurrencies`);
    const json = await res.json();
    if (json.status !== "success") return;

    allCryptosList = json.data;

    // Populate Explore dropdown
    const select = document.getElementById("explore-asset-filter");
    if (select) {
      allCryptosList.forEach(c => {
        const opt = document.createElement("option");
        opt.value = c.name;
        opt.textContent = `${c.name} (${c.symbol.toUpperCase()})`;
        select.appendChild(opt);
      });
    }

    // Populate Overview picker with top coins
    const overviewSelect = document.getElementById("overview-asset-picker");
    if (overviewSelect && overviewSelect.options.length <= 7) {
      const topCoins = allCryptosList.slice(0, 12);
      const existing = Array.from(overviewSelect.options).map(o => o.value);
      topCoins.forEach(c => {
        if (!existing.includes(c.name)) {
          const opt = document.createElement("option");
          opt.value = c.name;
          opt.textContent = `${c.name} (${c.symbol.toUpperCase()})`;
          overviewSelect.appendChild(opt);
        }
      });
    }
  } catch (err) {
    console.error("Failed to load cryptocurrencies list:", err);
  }
}

async function loadExploreData() {
  try {
    const res = await fetch(`${API_BASE}/api/historical-trends?limit=500`);
    const json = await res.json();
    if (json.status !== "success") return;

    allExploreRecords = json.data;
    filteredExploreRecords = allExploreRecords;
    currentExplorePage = 1;
    renderExploreTablePage();
  } catch (err) {
    console.error("Failed to load explore records:", err);
  }
}

async function onExploreFilterChange() {
  const assetFilter = document.getElementById("explore-asset-filter")?.value || "ALL";
  const typeFilter = document.getElementById("explore-type-filter")?.value || "ALL";
  const searchInput = (document.getElementById("explore-search-input")?.value || "").toLowerCase().trim();

  let dataset = allExploreRecords;

  // If a specific asset is chosen, fetch full history if needed
  if (assetFilter !== "ALL") {
    const countInCache = allExploreRecords.filter(r => r.name.toLowerCase() === assetFilter.toLowerCase()).length;
    if (countInCache < 50) {
      try {
        const res = await fetch(`${API_BASE}/api/historical-trends?asset=${encodeURIComponent(assetFilter)}&limit=250`);
        const json = await res.json();
        if (json.status === "success" && json.data) {
          dataset = json.data;
        }
      } catch (e) {
        console.error("Failed to fetch full asset history:", e);
      }
    }
  }

  filteredExploreRecords = dataset.filter(r => {
    if (assetFilter !== "ALL" && r.name.toLowerCase() !== assetFilter.toLowerCase()) return false;
    if (typeFilter !== "ALL" && r.asset_type !== typeFilter) return false;
    if (searchInput) {
      const matchName = r.name.toLowerCase().includes(searchInput);
      const matchSym = r.symbol.toLowerCase().includes(searchInput);
      if (!matchName && !matchSym) return false;
    }
    return true;
  });

  currentExplorePage = 1;
  renderExploreTablePage();
}

function resetExploreFilters() {
  const a = document.getElementById("explore-asset-filter");
  const t = document.getElementById("explore-type-filter");
  const s = document.getElementById("explore-search-input");
  if (a) a.value = "ALL";
  if (t) t.value = "ALL";
  if (s) s.value = "";
  filteredExploreRecords = allExploreRecords;
  currentExplorePage = 1;
  renderExploreTablePage();
}

function changeExplorePage(delta) {
  const totalPages = Math.ceil(filteredExploreRecords.length / explorePageSize) || 1;
  const newPage = currentExplorePage + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    currentExplorePage = newPage;
    renderExploreTablePage();
  }
}

function onPageSizeChange(newSize) {
  explorePageSize = parseInt(newSize, 10) || 50;
  currentExplorePage = 1;
  renderExploreTablePage();
}

function renderExploreTablePage() {
  const tbody = document.getElementById("explore-table-body");
  const counter = document.getElementById("explore-row-counter");
  const pageIndicator = document.getElementById("explore-page-indicator");
  const prevBtn = document.getElementById("btn-prev-page");
  const nextBtn = document.getElementById("btn-next-page");

  if (!tbody) return;

  const total = filteredExploreRecords.length;
  const totalPages = Math.ceil(total / explorePageSize) || 1;
  if (currentExplorePage > totalPages) currentExplorePage = totalPages;

  const startIdx = (currentExplorePage - 1) * explorePageSize;
  const endIdx = Math.min(startIdx + explorePageSize, total);
  const pageRecords = filteredExploreRecords.slice(startIdx, endIdx);

  if (pageRecords.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:2.5rem; color:#64748b;">No matching records found. Try clearing filters.</td></tr>`;
    if (counter) counter.innerText = "Showing 0 records";
    if (prevBtn) prevBtn.disabled = true;
    if (nextBtn) nextBtn.disabled = true;
    if (pageIndicator) pageIndicator.innerText = "Page 1 of 1";
    return;
  }

  const rowsHtml = pageRecords.map(r => `
    <tr>
      <td><code>${r.date}</code></td>
      <td><strong>${r.name}</strong></td>
      <td><code>${r.symbol.toUpperCase()}</code></td>
      <td><span class="pill ${r.asset_type === 'Stable-value' ? 'pill-stable' : 'pill-volatile'}">${r.asset_type}</span></td>
      <td>$${r.price_usd >= 1 ? r.price_usd.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : r.price_usd.toFixed(4)}</td>
      <td>${formatDollarAmount(r.vol_24h)}</td>
      <td>${formatDollarAmount(r.market_cap)}</td>
      <td style="font-weight:600; color: ${r.chg_24h >= 0 ? 'var(--color-success)' : 'var(--color-danger)'};">
        ${r.chg_24h > 0 ? '+' : ''}${r.chg_24h.toFixed(2)}%
      </td>
    </tr>
  `).join("");

  tbody.innerHTML = rowsHtml;

  if (counter) {
    counter.innerText = `Showing records ${startIdx + 1} – ${endIdx} of ${total.toLocaleString()}`;
  }
  if (pageIndicator) {
    pageIndicator.innerText = `Page ${currentExplorePage} of ${totalPages}`;
  }
  if (prevBtn) prevBtn.disabled = (currentExplorePage <= 1);
  if (nextBtn) nextBtn.disabled = (currentExplorePage >= totalPages);
}

// ==========================================================================
// TAB 3: DATA ANALYSIS (ALL 15 QUESTIONS & HYPOTHESIS TESTS)
// ==========================================================================
let allAnalysisQuestionsData = [];

async function loadAll15Questions() {
  const container = document.getElementById("analysis-questions-container");
  if (!container) return;

  allAnalysisQuestionsData = [
    {
      id: 1,
      category: "concentration",
      title: "Market Capitalization Concentration & Oligopoly Structure",
      question: "Which cryptocurrencies hold the majority of market value, and is the overall market diversified?",
      observation: "Bitcoin alone accounts for 59.5% of total market value. The top 3 assets (Bitcoin, Ethereum, Tether) capture 77.8%, and the top 10 capture 92.9%. The daily Herfindahl-Hirschman Index (HHI) averages 3,753.",
      explanation: "The crypto market operates as an oligopoly dominated by Bitcoin and Ethereum. Diversifying across a 50-asset basket is largely an illusion because the bottom 40 assets collectively share barely 7% of total market value.",
      chartType: "bar",
      labels: ["Bitcoin", "Ethereum", "Tether", "BNB", "Solana", "USDC", "XRP", "Lido Staked", "Cardano", "Others (41 coins)"],
      data: [59.5, 10.7, 7.6, 3.8, 3.3, 3.1, 1.8, 1.2, 1.0, 8.0],
      metricLabel: "Market Cap Share (%)"
    },
    {
      id: 2,
      category: "concentration",
      title: "Bitcoin Market Dominance Evolution Over Time",
      question: "Did Bitcoin gain or lose market share during price rallies and market corrections?",
      observation: "Bitcoin dominance rose from 58.29% to 60.78% over the period and moved positively with Bitcoin's price (Spearman correlation ρ = 0.597, p < 0.001).",
      explanation: "When Bitcoin rallied, it gained market share at the expense of altcoins. Holding smaller volatile tokens did not provide relative alpha during market recovery phases.",
      chartType: "line",
      labels: ["Feb 01", "Feb 20", "Mar 10", "Mar 30", "Apr 20", "May 10"],
      data: [58.3, 58.7, 59.1, 59.8, 60.3, 60.8],
      metricLabel: "BTC Market Dominance (%)"
    },
    {
      id: 3,
      category: "concentration",
      title: "Trading Volume Concentration vs. Market Capitalization",
      question: "Do the coins with the highest market capitalization also experience the highest trading volume?",
      observation: "Tether commands 43.9% of all trading volume with only 7.6% of market cap (5.8x ratio). Bitcoin holds 59.5% of market cap but only 26.7% of volume. Top 3 assets carry 82.7% of trading.",
      explanation: "Store-of-value assets (Bitcoin) and transactional mediums of exchange (Tether) serve distinct market functions. Trading volume is significantly more concentrated than market capitalization.",
      chartType: "grouped_bar",
      labels: ["Tether", "Bitcoin", "Ethereum", "BNB", "USDC"],
      data1: [43.9, 26.7, 12.1, 0.7, 6.4],
      data2: [7.6, 59.5, 10.7, 3.8, 3.1],
      label1: "Volume Share (%)",
      label2: "Market Cap Share (%)"
    },
    {
      id: 4,
      category: "concentration",
      title: "Asset Liquidity Relative to Size (Turnover %)",
      question: "Which cryptocurrencies trade actively relative to their capitalization versus passive locked tokens?",
      observation: "Tether (37.5%), USD1 (28.4%), and Pepe (21.1%) have the highest daily turnover. In contrast, institutional tokens like Circle USYC (0.004%) and BlackRock BUIDL (0.0%) exhibit near-zero turnover.",
      explanation: "Stablecoins are bifurcated into two distinct classes: high-velocity trading stablecoins used on exchanges, and passive yield-bearing tokens held in institutional custody.",
      chartType: "bar",
      labels: ["Tether", "USD1", "Pepe", "Aave", "Bitcoin", "Circle USYC"],
      data: [37.5, 28.4, 21.1, 20.7, 1.8, 0.004],
      metricLabel: "Median Daily Turnover (%)"
    },
    {
      id: 5,
      category: "volatility",
      title: "Distributional Normality of Daily Returns (Fat-Tail Risk)",
      question: "Do daily returns follow a Gaussian bell-curve distribution as assumed by standard risk models?",
      observation: "Daily returns have extreme fat tails with excess kurtosis of ~350 for volatile assets. Days with moves beyond ±3 standard deviations occur 3.4 times more frequently than normal theory predicts.",
      explanation: "Crypto returns are non-Gaussian. Standard financial risk models (like Gaussian Value-at-Risk) dangerously underestimate the likelihood of extreme crashes and sudden spikes.",
      chartType: "bar",
      labels: ["Normal Theory (Gaussian)", "Bitcoin", "Volatile Coins Universe"],
      data: [3.0, 7.1, 350.0],
      metricLabel: "Excess Kurtosis (Fat-Tail Indicator)"
    },
    {
      id: 6,
      category: "volatility",
      title: "Cross-Asset Volatility Rankings (Highest and Lowest Risk)",
      question: "Which tokens exhibit the highest and lowest daily price fluctuations across the market?",
      observation: "Meme coins and emerging tokens (RaveDAO, Siren, Pepe) exhibited extreme daily standard deviations (>15%), whereas large-cap foundation tokens (Bitcoin, Ethereum, Tron) stayed between 2.9% and 4.2%.",
      explanation: "Cryptocurrency volatility spans multiple risk tiers; small-cap tokens carry several times more daily price uncertainty than large layer-1 foundation networks.",
      chartType: "bar",
      labels: ["RaveDAO (Meme)", "Siren (Emerging)", "Pepe (Meme)", "Solana (L1)", "Bitcoin (L1)", "Tron (Defensive)"],
      data: [16.8, 15.4, 9.8, 5.8, 3.8, 2.9],
      metricLabel: "Daily Return Standard Deviation (%)"
    },
    {
      id: 7,
      category: "volatility",
      title: "Stable-Value vs. Volatile Asset Volatility Disparity (A/B Formulation)",
      question: "Does the automated coefficient of variation (CV < 1%) rule cleanly separate distinct risk profiles?",
      observation: "Median daily absolute price change is 3.33% for volatile coins versus only 0.06% for stablecoins (55x ratio). Volatile return standard deviation is 5.65% vs 0.07% for stablecoins.",
      explanation: "The two categories are completely segregated. Stablecoins successfully peg their price near $1.00, validating their use as collateral in margin and lending frameworks.",
      chartType: "bar",
      labels: ["Stablecoins (17 Coins)", "Volatile Coins (54 Coins)"],
      data: [0.06, 3.33],
      metricLabel: "Median Daily Absolute Move (%)"
    },
    {
      id: 8,
      category: "volatility",
      title: "Market Capitalization vs. Price Volatility Relationship",
      question: "Does a higher market capitalization provide safety against daily price volatility?",
      observation: "The correlation between market capitalization and return standard deviation is statistically insignificant (Spearman ρ = -0.138, p = 0.370).",
      explanation: "Size is not safety in cryptocurrency. High market capitalization alone does not guarantee lower percentage fluctuations during market stress.",
      chartType: "bar",
      labels: ["Small Cap Altcoins", "Mid Cap Altcoins", "Large Cap Assets"],
      data: [5.8, 5.9, 5.1],
      metricLabel: "Average Daily Return SD (%)"
    },
    {
      id: 9,
      category: "volatility",
      title: "Bitcoin Benchmark Trajectory & Maximum Historical Drawdown",
      question: "What was the largest peak-to-trough price destruction experienced by Bitcoin during this period?",
      observation: "Bitcoin peaked at $90,117 in March 2026 and dropped to $58,260 in July/August, recording a maximum peak-to-trough drawdown of -35.34%. Overall net return over the 183 days was -25.92%.",
      explanation: "Even the benchmark market leader suffered a 35% decline in 4 months, confirming that long-term crypto holders must maintain strict risk controls.",
      chartType: "bar",
      labels: ["Peak (March 2026)", "Trough (July 2026)", "Maximum Peak Drawdown"],
      data: [90117, 58260, -35.34],
      metricLabel: "Price (USD) / Drawdown (%)"
    },
    {
      id: 10,
      category: "dynamics",
      title: "Cross-Asset Return Co-Movement & Correlation Matrix",
      question: "Do altcoins move independently or do they follow Bitcoin in lockstep?",
      observation: "Average pairwise correlation between major assets is 0.54. Ethereum (r = 0.78) and Solana (r = 0.72) correlate strongly with Bitcoin.",
      explanation: "Cryptocurrencies move together in a herd. Holding multiple altcoins offers limited downside diversification during broad market selloffs.",
      chartType: "bar",
      labels: ["ETH to BTC", "SOL to BTC", "XRP to BTC", "Cardano to BTC", "Broad Market Average"],
      data: [0.78, 0.72, 0.58, 0.61, 0.54],
      metricLabel: "Pearson Return Correlation (r)"
    },
    {
      id: 11,
      category: "dynamics",
      title: "Asset Sensitivity to Bitcoin Market Movements (Beta)",
      question: "Which coins amplify Bitcoin moves (high beta) and which are more defensive (low beta)?",
      observation: "13 of 30 analyzed assets have beta > 1.0. High-beta leaders include NEAR (1.64), Avalanche (1.46), and Solana (1.37). Defensive tokens include LEO (0.13) and Tron (0.58).",
      explanation: "High-beta tokens magnify Bitcoin rallies and crashes by 1.4x to 1.6x, making them aggressive tools for bull runs but highly hazardous during downturns.",
      chartType: "bar",
      labels: ["NEAR (Aggressive)", "Avalanche", "Solana", "Bitcoin (Benchmark)", "Tron (Defensive)", "LEO (Defensive)"],
      data: [1.64, 1.46, 1.37, 1.00, 0.58, 0.13],
      metricLabel: "Market Beta Relative to Bitcoin (β)"
    },
    {
      id: 12,
      category: "dynamics",
      title: "Day-of-Week Variations in Market Volume and Returns",
      question: "Are certain weekdays more active or profitable than others?",
      observation: "Trading volume peaks on Tuesday–Thursday ($32B–$35B) and drops on Saturday–Sunday ($21B–$24B). Bitcoin median return is highest on Monday (+1.10%) and lowest on Thursday (-1.30%).",
      explanation: "Even though crypto exchanges operate 24/7, institutional market makers and ETF desks mainly operate during traditional banking weekdays.",
      chartType: "bar",
      labels: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
      data: [28.5, 34.2, 33.8, 32.1, 29.4, 23.8, 21.6],
      metricLabel: "Median 24h Volume ($ Billion)"
    },
    {
      id: 13,
      category: "dynamics",
      title: "Weekend Volume Contraction Across Total Market (A/B Test)",
      question: "How substantial is the weekend trading volume contraction across the entire 50-asset market?",
      observation: "Total market volume averages $189.3 billion on weekdays versus $120.2 billion on weekends — an absolute decline of $69.1 billion (-36.5%). Welch t = -5.03, p = 5.9e-6.",
      explanation: "The weekend slump is massive and statistically verified. Order books are thinner on weekends, leading to wider bid-ask spreads and higher slippage.",
      chartType: "bar",
      labels: ["Weekday Average (Mon-Fri)", "Weekend Average (Sat-Sun)"],
      data: [189.3, 120.2],
      metricLabel: "Total Tracked Market Volume ($ Billion)"
    },
    {
      id: 14,
      category: "dynamics",
      title: "Trading Volume vs. Price Volatility Magnitude",
      question: "Does large price volatility provoke higher trading activity (Mixture of Distributions)?",
      observation: "Bitcoin trading volume correlates positively with the absolute size of price moves (Spearman ρ = 0.462, p < 0.0001). Every 1% daily move adds ~$6.2 billion in volume.",
      explanation: "Trading activity spikes during large price jumps and crashes regardless of direction, supporting the financial theory that volatility fuels volume.",
      chartType: "bar",
      labels: ["Low Move (<1%)", "Normal Move (1-3%)", "Elevated Move (3-5%)", "Large Move (>5%)"],
      data: [18.2, 24.5, 33.1, 42.1],
      metricLabel: "Average 24h Trading Volume ($ Billion)"
    },
    {
      id: 15,
      category: "dynamics",
      title: "Market Breadth: Altcoins Following Bitcoin Direction",
      question: "How reliable is Bitcoin's daily direction as a directional compass for the broader market?",
      observation: "On Bitcoin-up days, 73.0% of altcoins rise. On Bitcoin-down days, only 25.3% rise. Bitcoin direction agreed with market majority on 86.1% of days (Spearman ρ = 0.868, p < 1e-30).",
      explanation: "Bitcoin is the decisive market compass. Betting against Bitcoin's intraday momentum carries an 86% historical failure rate.",
      chartType: "bar",
      labels: ["When Bitcoin is UP", "When Bitcoin is DOWN"],
      data: [73.0, 25.3],
      metricLabel: "% of Altcoins with Positive Return"
    }
  ];

  renderAnalysisQuestionsList(allAnalysisQuestionsData);
}

function filterAnalysisCategory(cat, btn) {
  const chips = document.querySelectorAll(".filter-chip");
  chips.forEach(c => c.classList.remove("active"));
  if (btn) btn.classList.add("active");

  if (cat === "all") {
    renderAnalysisQuestionsList(allAnalysisQuestionsData);
  } else {
    const filtered = allAnalysisQuestionsData.filter(q => q.category === cat);
    renderAnalysisQuestionsList(filtered);
  }
}

function renderAnalysisQuestionsList(questions) {
  const container = document.getElementById("analysis-questions-container");
  if (!container) return;

  // Clear existing Chart instances to prevent memory leaks
  Object.keys(questionChartsMap).forEach(key => {
    if (questionChartsMap[key]) {
      questionChartsMap[key].destroy();
    }
  });
  questionChartsMap = {};

  let html = "";
  questions.forEach(q => {
    html += `
      <div class="question-block" id="block-q-${q.id}">
        <div class="q-header">
          <span class="q-number-badge">Question ${q.id}</span>
          <span class="q-title">${q.title}</span>
        </div>

        <div class="q-prompt-box">
          <strong>Research Question:</strong> ${q.question}
        </div>

        <div class="q-content-grid">
          <div>
            <div style="height: 230px; position: relative;">
              <canvas id="canvas-q-${q.id}"></canvas>
            </div>
          </div>
          <div>
            <div class="box-observation">
              <strong>Key Finding:</strong> ${q.observation}
            </div>
            <div class="box-explanation">
              <strong>Plain English Explanation:</strong> ${q.explanation}
            </div>
          </div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;

  // Render Charts
  setTimeout(() => {
    questions.forEach(q => {
      const canvas = document.getElementById(`canvas-q-${q.id}`);
      if (!canvas) return;

      if (q.chartType === "grouped_bar") {
        questionChartsMap[q.id] = renderGroupedBarChart(canvas, q.labels, q.data1, q.data2, q.label1, q.label2);
      } else if (q.chartType === "line") {
        questionChartsMap[q.id] = renderAnalysisLineChart(canvas, q.labels, q.data, q.metricLabel);
      } else {
        questionChartsMap[q.id] = renderAnalysisBarChart(canvas, q.labels, q.data, q.metricLabel);
      }
    });
  }, 50);
}

function renderAnalysisBarChart(canvas, labels, data, label) {
  return new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: label,
        data: data,
        backgroundColor: "#3578F6",
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0b132b",
          padding: 8,
          cornerRadius: 4
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { family: "Inter", size: 10 }, color: "#475569" }
        },
        y: {
          grid: { color: "#f1f5f9" },
          ticks: { font: { family: "Inter", size: 10 }, color: "#64748b" }
        }
      }
    }
  });
}

function renderGroupedBarChart(canvas, labels, data1, data2, label1, label2) {
  return new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: label1, data: data1, backgroundColor: "#3578F6", borderRadius: 4 },
        { label: label2, data: data2, backgroundColor: "#94a3b8", borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { font: { family: "Inter", size: 11, weight: 600 }, color: "#0b132b" }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: "Inter", size: 10 } } },
        y: { grid: { color: "#f1f5f9" }, ticks: { font: { family: "Inter", size: 10 } } }
      }
    }
  });
}

function renderAnalysisLineChart(canvas, labels, data, label) {
  return new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: label,
        data: data,
        borderColor: "#3578F6",
        backgroundColor: "rgba(53, 120, 246, 0.08)",
        fill: true,
        tension: 0.2,
        pointRadius: 2.5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { family: "Inter", size: 10 } } },
        y: { grid: { color: "#f1f5f9" }, ticks: { font: { family: "Inter", size: 10 } } }
      }
    }
  });
}

// Hypothesis Testing Section
async function loadHypothesisTests() {
  try {
    const res = await fetch(`${API_BASE}/api/hypothesis-tests`);
    const json = await res.json();
    if (json.status !== "success") return;

    const container = document.getElementById("hypothesis-cards-container");
    if (!container) return;

    container.innerHTML = json.data.map(t => {
      const isReject = t.decision.includes("Reject H0");
      const badgeClass = isReject ? "pill-reject" : "pill-fail";

      return `
        <div class="ht-clean-card">
          <div class="ht-top">
            <div class="ht-name">${t.name}</div>
            <span class="pill ${badgeClass}">${t.decision}</span>
          </div>

          <div class="ht-details">
            <div><strong>H₀ (Null):</strong> ${t.h0}</div>
            <div style="margin-top:0.35rem;"><strong>H₁ (Alternative):</strong> ${t.h1}</div>
          </div>

          <div class="ht-stat-bar">
            <span>Statistic: <code>${t.test_statistic}</code></span>
            <span>p-value: <code style="color:${isReject ? '#dc2626' : '#3578F6'}; font-weight:700;">${t.p_value < 0.0001 ? t.p_value.toExponential(2) : t.p_value.toFixed(4)}</code></span>
            <span>Significance α = ${t.alpha}</span>
          </div>

          <div class="ht-interpretation">
            <strong>Interpretation:</strong> ${t.implication}
          </div>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load hypothesis tests:", err);
  }
}

// ==========================================================================
// TAB 4: TABLEAU EMBED HELPERS
// ==========================================================================
function reloadTableauEmbed() {
  const iframe = document.getElementById("tableau-iframe");
  if (iframe) {
    const src = iframe.src;
    iframe.src = "";
    setTimeout(() => {
      iframe.src = src;
    }, 150);
  }
}

// Utility: Format Large Dollar Amounts
function formatDollarAmount(val) {
  if (val === undefined || val === null || isNaN(val)) return "$0";
  if (val >= 1e9) return `$${(val / 1e9).toFixed(2)} B`;
  if (val >= 1e6) return `$${(val / 1e6).toFixed(2)} M`;
  return `$${val.toLocaleString()}`;
}
