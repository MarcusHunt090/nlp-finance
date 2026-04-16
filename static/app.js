let appData = null;
let benchmarkData = null;
let explainData = null;
let lastAnalyzedText = '';
let lastBatchResults = [];
let stockCharts = {};
let currentPage = 'home';

// ===================== ROUTING =====================

const PAGE_TITLES = {
    home: 'Home', analyze: 'Analyze', batch: 'Batch Analysis',
    compare: 'Compare', news: 'Live News', methodology: 'Methodology',
    benchmarks: 'Benchmarks', explainability: 'Explainability',
    dashboard: 'Dashboard', history: 'History',
};

function navigate(pageId, navEl) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    // Show target page
    const page = document.getElementById('page-' + pageId);
    if (page) page.classList.add('active');
    if (navEl) navEl.classList.add('active');

    // Update breadcrumb
    const bc = document.getElementById('topbar-breadcrumb');
    if (bc) bc.textContent = PAGE_TITLES[pageId] || pageId;

    currentPage = pageId;
    window.location.hash = pageId;

    // Lazy-render charts when tabs become visible
    if (pageId === 'benchmarks' && benchmarkData) {
        showROC('negative', document.querySelector('#page-benchmarks .tab'));
        showPR('negative', document.querySelector('#page-benchmarks .bench-grid .chart-card:nth-child(2) .tab'));
        renderLearningCurves(benchmarkData.learning_curves);
    }
    if (pageId === 'explainability' && explainData) {
        showExplainModel('logistic_regression', document.querySelector('#page-explainability .tab'));
    }
    if (pageId === 'dashboard' && appData) {
        renderDistChart(appData.dataset.label_distribution);
        renderWordBars('negative');
    }
    if (pageId === 'history') {
        loadHistory();
    }

    // Close mobile sidebar
    document.getElementById('sidebar').classList.remove('open');
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// Read initial hash
function initRoute() {
    const hash = window.location.hash.replace('#', '');
    if (hash && document.getElementById('page-' + hash)) {
        const navEl = document.querySelector(`[data-page="${hash}"]`);
        navigate(hash, navEl);
    }
}

const MODEL_COLORS = {
    logistic_regression: '#6366f1',
    naive_bayes: '#f59e0b',
    random_forest: '#10b981',
    svc: '#0ea5e9',
    finbert: '#8b5cf6',
};
const SENT_COLORS = { negative: '#ef4444', neutral: '#eab308', positive: '#22c55e' };

// Language code display names
const LANG_NAMES = {
    en: 'English', es: 'Spanish', fr: 'French', de: 'German', zh: 'Chinese',
    ja: 'Japanese', ko: 'Korean', pt: 'Portuguese', it: 'Italian', ru: 'Russian',
    ar: 'Arabic', hi: 'Hindi', nl: 'Dutch', sv: 'Swedish', pl: 'Polish',
};

// ===================== INIT =====================

document.addEventListener('DOMContentLoaded', () => {
    initRoute();

    fetch('/api/stats').then(r => r.json()).then(data => {
        appData = data;
        renderStats(data.dataset);
        renderHeroStats(data);
        renderSampleChips(data.samples);
        renderModelCards(data.models);
        document.getElementById('feedback-count').textContent = data.feedback_count || 0;
        if (document.getElementById('m-train-count')) document.getElementById('m-train-count').textContent = (data.dataset.train_size || 0).toLocaleString();
        if (document.getElementById('m-test-count')) document.getElementById('m-test-count').textContent = (data.dataset.test_size || 0).toLocaleString();
        updateFinbertBanner(data.finbert_status);
        // topbar
        const ts = document.getElementById('topbar-stats');
        const models = data.models || {};
        const bestAcc = Math.max(...Object.values(models).map(m => m.accuracy).filter(Boolean));
        if (ts) ts.innerHTML = `<span class="topbar-stat"><strong>${(data.dataset.train_size||0).toLocaleString()}</strong> samples</span><span class="topbar-stat"><strong>${bestAcc}%</strong> accuracy</span>`;
        // render dashboard if already on that tab
        if (currentPage === 'dashboard') { renderDistChart(data.dataset.label_distribution); renderWordBars('negative'); }
    });

    fetch('/api/benchmarks').then(r => r.json()).then(data => {
        benchmarkData = data;
        if (currentPage === 'benchmarks') {
            showROC('negative', document.querySelector('#page-benchmarks .tab'));
            showPR('negative', document.querySelector('#page-benchmarks .bench-grid .chart-card:nth-child(2) .tab'));
            renderLearningCurves(data.learning_curves);
        }
    });

    fetch('/api/explainability').then(r => r.json()).then(data => {
        explainData = data;
        if (currentPage === 'explainability') showExplainModel('logistic_regression', document.querySelector('#page-explainability .tab'));
    });

    loadHistory();

    // Poll FinBERT status until ready or failed
    pollFinbertStatus();
});

function renderHeroStats(data) {
    const models = data.models || {};
    const accs = Object.values(models).map(m => m.accuracy).filter(Boolean);
    const bestAcc = accs.length ? Math.max(...accs) : null;
    const trainEl = document.getElementById('hm-train');
    const accEl  = document.getElementById('hm-acc');
    if (trainEl) trainEl.textContent = (data.dataset.train_size || 0).toLocaleString();
    if (accEl)   accEl.textContent  = bestAcc !== null ? bestAcc + '%' : '—';
}

// ===================== HERO PANEL =====================

async function heroAnalyze() {
    const input = document.getElementById('hero-input');
    const text = input?.value.trim();
    if (!text) return;
    const btn = document.getElementById('hero-btn-text');
    const sp  = document.getElementById('hero-spinner');
    const resultDiv = document.getElementById('hero-result');
    const inner = document.getElementById('hero-result-inner');
    if (btn) btn.textContent = 'Analyzing…';
    if (sp)  sp.classList.remove('hidden');
    try {
        const res = await fetch('/api/predict', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({text}),
        });
        const d = await res.json();
        const ens = d.predictions?.ensemble || {};
        const conf = ens.confidence || {};
        inner.innerHTML = `
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span class="sentiment-badge sentiment-${ens.label}">${(ens.label||'').toUpperCase()}</span>
                <span style="font-size:0.75rem;color:var(--text-muted)">Ensemble result</span>
            </div>
            <div class="confidence-bars">
                ${['negative','neutral','positive'].map(s=>`
                <div class="conf-row">
                    <span class="conf-label">${s}</span>
                    <div class="conf-bar-bg"><div class="conf-bar ${s}" style="width:${conf[s]||0}%"></div></div>
                    <span class="conf-val">${conf[s]||0}%</span>
                </div>`).join('')}
            </div>`;
        resultDiv.style.display = 'block';
    } catch(e) { console.error(e); }
    finally {
        if (btn) btn.textContent = 'Analyze Sentiment';
        if (sp)  sp.classList.add('hidden');
    }
}

function updateFinbertBanner(status) {
    // Sidebar pill
    const pill = document.getElementById('finbert-status-sidebar');
    const pillText = document.getElementById('finbert-pill-text');
    const dot = pill ? pill.querySelector('.finbert-dot') : null;
    if (dot) dot.className = 'finbert-dot';
    if (status === 'loading') {
        if (dot) dot.classList.add('loading');
        if (pillText) pillText.textContent = 'FinBERT loading…';
    } else if (status === 'ready') {
        if (dot) dot.classList.add('ready');
        if (pillText) pillText.textContent = 'FinBERT ready';
    } else if (status === 'failed') {
        if (dot) dot.classList.add('failed');
        if (pillText) pillText.textContent = 'FinBERT unavailable';
    }
}

function pollFinbertStatus() {
    let polls = 0;
    const interval = setInterval(async () => {
        polls++;
        if (polls > 60) { clearInterval(interval); return; } // give up after ~5 min
        try {
            const res = await fetch('/api/finbert-status');
            const { status } = await res.json();
            updateFinbertBanner(status);
            if (status === 'ready' || status === 'failed') {
                clearInterval(interval);
            }
        } catch (e) { /* ignore */ }
    }, 5000);
}

// ===================== ANALYZE =====================

async function analyzeText() {
    const input = document.getElementById('text-input');
    const text = input.value.trim();
    if (!text) return;

    const btn = document.getElementById('analyze-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('btn-spinner');
    btn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.classList.remove('hidden');

    // Show skeleton
    document.getElementById('results').classList.add('hidden');
    document.getElementById('results-skeleton').classList.remove('hidden');

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();
        lastAnalyzedText = text;

        document.getElementById('results-skeleton').classList.add('hidden');
        renderResults(data.predictions, data.entities, data.language);
        renderInlineExplanation(data.explanations);
        renderStockSection(data.entities || []);
        loadHistory();

        // Update FinBERT banner if status changed
        if (data.finbert_status) updateFinbertBanner(data.finbert_status);
    } catch (e) {
        console.error(e);
        document.getElementById('results-skeleton').classList.add('hidden');
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Analyze Sentiment';
        spinner.classList.add('hidden');
    }
}

function renderResults(predictions, entities, language) {
    const grid = document.getElementById('results-grid');
    grid.innerHTML = '';

    const order = ['ensemble', 'logistic_regression', 'naive_bayes', 'random_forest', 'svc', 'finbert'];
    for (const key of order) {
        const pred = predictions[key];
        if (!pred) continue;
        const card = document.createElement('div');
        card.className = 'result-card' + (key === 'ensemble' ? ' ensemble-card' : '');

        let confHTML = '';
        if (pred.confidence) {
            confHTML = `<div class="confidence-bars">
                ${['negative', 'neutral', 'positive'].map(s => `
                    <div class="conf-row">
                        <span class="conf-label">${s}</span>
                        <div class="conf-bar-bg"><div class="conf-bar ${s}" style="width: ${pred.confidence[s] || 0}%"></div></div>
                        <span class="conf-val">${pred.confidence[s] || 0}%</span>
                    </div>`).join('')}
            </div>`;
        }

        let extraBadge = '';
        if (key === 'finbert') {
            extraBadge = '<span style="font-size:0.7rem;color:var(--accent);font-weight:600;margin-left:4px">TRANSFORMER</span>';
        }
        if (key === 'ensemble' && pred.stacking) {
            extraBadge = '<span style="font-size:0.7rem;color:#10b981;font-weight:600;margin-left:4px">STACKING</span>';
        } else if (key === 'ensemble' && predictions.ensemble?.finbert_included) {
            extraBadge = '<span style="font-size:0.7rem;color:#8b5cf6;font-weight:600;margin-left:4px">+FinBERT</span>';
        }

        card.innerHTML = `<h4>${pred.model_name}${extraBadge}</h4>
            <span class="sentiment-badge sentiment-${pred.label}">${pred.label.toUpperCase()}</span>${confHTML}`;
        grid.appendChild(card);
    }

    // Language badge
    const langBadge = document.getElementById('lang-badge');
    if (language && language !== 'en') {
        const name = LANG_NAMES[language] || language.toUpperCase();
        langBadge.textContent = name;
        langBadge.className = 'lang-badge non-english';
        langBadge.classList.remove('hidden');
    } else if (language === 'en') {
        langBadge.textContent = 'EN';
        langBadge.className = 'lang-badge';
        langBadge.classList.remove('hidden');
    } else {
        langBadge.classList.add('hidden');
    }

    const entDiv = document.getElementById('entities-display');
    entDiv.innerHTML = (entities || []).map(e => `<span class="entity-tag">${e.ticker} — ${e.name}</span>`).join('');

    document.getElementById('results').classList.remove('hidden');
    document.getElementById('feedback-area').classList.remove('hidden');
    document.getElementById('feedback-status').textContent = '';
}

function renderInlineExplanation(explanations) {
    if (!explanations) return;
    const section = document.getElementById('explain-section');
    const grid = document.getElementById('explain-grid');
    section.classList.remove('hidden');

    grid.innerHTML = Object.entries(explanations).map(([key, ex]) => {
        const words = ex.top_words || [];
        if (!words.length) return '';
        const barsHTML = words.slice(0, 6).map(w => {
            const maxW = Math.abs(words[0].weight);
            const pct = maxW > 0 ? Math.min(100, (Math.abs(w.weight) / maxW) * 100) : 0;
            const color = w.direction === 'opposing' ? SENT_COLORS.negative : SENT_COLORS.positive;
            return `<div class="word-bar-row">
                <span class="word-bar-label">${w.word}</span>
                <div class="word-bar-fill-bg"><div class="word-bar-fill" style="width:${pct}%;background:${color}"></div></div>
                <span class="word-bar-count">${w.weight}</span>
            </div>`;
        }).join('');
        return `<div class="explain-card">
            <h5>${ex.model_name}</h5>
            <span class="sentiment-badge sentiment-${ex.prediction}" style="font-size:0.75rem">${ex.prediction}</span>
            ${barsHTML}
        </div>`;
    }).join('');
}

function switchAnalyzeTab(tab, el) {
    document.querySelectorAll('#page-analyze .tab-bar .tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('analyze-url-panel').classList.toggle('hidden', tab !== 'url');
    document.getElementById('analyze-text-panel').classList.toggle('hidden', tab !== 'text');
    // Hide results when switching tabs
    document.getElementById('results').classList.add('hidden');
    document.getElementById('results-skeleton').classList.add('hidden');
    if (document.getElementById('article-meta-card')) {
        document.getElementById('article-meta-card').classList.add('hidden');
    }
}

async function analyzeUrl() {
    const input = document.getElementById('url-input');
    const url = input.value.trim();
    if (!url) return;

    const btn = document.getElementById('url-analyze-btn');
    const btnText = document.getElementById('url-btn-text');
    const spinner = document.getElementById('url-btn-spinner');
    btn.disabled = true;
    btnText.textContent = 'Fetching Article...';
    spinner.classList.remove('hidden');

    document.getElementById('results').classList.add('hidden');
    document.getElementById('results-skeleton').classList.remove('hidden');
    document.getElementById('article-meta-card').classList.add('hidden');

    try {
        const res = await fetch('/api/analyze-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });
        const data = await res.json();

        if (data.error) {
            document.getElementById('results-skeleton').classList.add('hidden');
            alert(data.error);
            return;
        }

        lastAnalyzedText = url;

        // Show article metadata
        const meta = data.article;
        const metaCard = document.getElementById('article-meta-card');
        const metaContent = document.getElementById('article-meta-content');
        if (meta) {
            metaContent.innerHTML = `
                ${meta.title ? `<div style="font-weight:600;font-size:0.95rem;margin-bottom:6px;">${meta.title}</div>` : ''}
                ${meta.snippet ? `<div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:8px;line-height:1.5;">${meta.snippet}</div>` : ''}
                <div style="display:flex;gap:14px;font-size:0.72rem;color:var(--text-muted);flex-wrap:wrap;">
                    ${meta.source ? `<span>Source: ${meta.source}</span>` : ''}
                    ${meta.word_count ? `<span>${meta.word_count} words analyzed</span>` : ''}
                    ${meta.publish_date ? `<span>${meta.publish_date.split(' ')[0]}</span>` : ''}
                </div>`;
            metaCard.classList.remove('hidden');
        }

        document.getElementById('results-skeleton').classList.add('hidden');
        renderResults(data.predictions, data.entities, data.language);
        renderInlineExplanation(data.explanations);
        renderStockSection(data.entities || []);
        loadHistory();

        if (data.finbert_status) updateFinbertBanner(data.finbert_status);
    } catch (e) {
        console.error(e);
        document.getElementById('results-skeleton').classList.add('hidden');
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Analyze Article';
        spinner.classList.add('hidden');
    }
}

// ===================== STOCK CORRELATION =====================

async function renderStockSection(entities) {
    const section = document.getElementById('stock-section');
    const container = document.getElementById('stock-charts');
    const tickers = [...new Set(entities.map(e => e.ticker))].slice(0, 4);

    if (!tickers.length) {
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');
    container.innerHTML = tickers.map(t => `
        <div class="stock-card" id="stock-${t}">
            <div class="stock-card-header">
                <span class="stock-ticker">${t}</span>
                <span class="stock-price" id="price-${t}">Loading...</span>
            </div>
            <canvas class="stock-canvas" id="canvas-${t}"></canvas>
        </div>`).join('');

    try {
        const res = await fetch('/api/stock-data', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers, period: '1mo' }),
        });
        const data = await res.json();
        for (const [ticker, info] of Object.entries(data.stocks || {})) {
            renderMiniStockChart(ticker, info);
        }
    } catch (e) {
        section.classList.add('hidden');
    }
}

function renderMiniStockChart(ticker, info) {
    const priceEl = document.getElementById(`price-${ticker}`);
    const canvas = document.getElementById(`canvas-${ticker}`);
    if (!priceEl || !canvas) return;

    const changeSign = info.change_pct >= 0 ? '+' : '';
    priceEl.innerHTML = `$${info.current} <span class="stock-change ${info.change_pct >= 0 ? 'up' : 'down'}">${changeSign}${info.change_pct}%</span>`;

    const ctx = canvas.getContext('2d');
    const color = info.change_pct >= 0 ? '#22c55e' : '#ef4444';

    // Destroy existing chart if any
    if (stockCharts[ticker]) stockCharts[ticker].destroy();

    stockCharts[ticker] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: info.dates,
            datasets: [{
                data: info.closes,
                borderColor: color,
                backgroundColor: color + '22',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: false,
            plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
            scales: {
                x: { display: false },
                y: { display: false },
            },
        },
    });
}

// ===================== FEEDBACK =====================

async function submitFeedback(label) {
    if (!lastAnalyzedText) return;
    const res = await fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: lastAnalyzedText, label }),
    });
    const data = await res.json();
    document.getElementById('feedback-status').textContent = `Saved! (${data.total_feedback} total)`;
    document.getElementById('feedback-count').textContent = data.total_feedback;
}

async function retrainModels() {
    const status = document.getElementById('retrain-status');
    status.textContent = 'Retraining...';
    const res = await fetch('/api/retrain', { method: 'POST' });
    const data = await res.json();
    status.textContent = `Done! ${data.train_size} samples.`;
    location.reload();
}

// ===================== BENCHMARKS: ROC =====================

let rocChart = null;
function showROC(sentiment, el) {
    if (!benchmarkData) return;
    el.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');

    const ctx = document.getElementById('roc-chart').getContext('2d');
    if (rocChart) rocChart.destroy();

    const datasets = [];
    for (const [key, model] of Object.entries(benchmarkData.roc_pr)) {
        const roc = model.roc[sentiment];
        if (!roc) continue;
        datasets.push({
            label: `${model.name} (AUC=${roc.auc})`,
            data: roc.fpr.map((x, i) => ({ x, y: roc.tpr[i] })),
            borderColor: MODEL_COLORS[key] || '#888',
            fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2,
        });
    }
    datasets.push({
        label: 'Random (AUC=0.5)',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: '#444', borderDash: [5, 5], fill: false, pointRadius: 0, borderWidth: 1,
    });

    rocChart = new Chart(ctx, {
        type: 'scatter', data: { datasets },
        options: {
            responsive: true, showLine: true,
            scales: {
                x: { title: { display: true, text: 'False Positive Rate', color: '#8892a8' }, min: 0, max: 1, ticks: { color: '#8892a8' }, grid: { color: '#243056' } },
                y: { title: { display: true, text: 'True Positive Rate', color: '#8892a8' }, min: 0, max: 1, ticks: { color: '#8892a8' }, grid: { color: '#243056' } },
            },
            plugins: { legend: { labels: { color: '#8892a8', font: { size: 11 } } } },
        },
    });
}

// ===================== BENCHMARKS: PR =====================

let prChart = null;
function showPR(sentiment, el) {
    if (!benchmarkData) return;
    el.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');

    const ctx = document.getElementById('pr-chart').getContext('2d');
    if (prChart) prChart.destroy();

    const datasets = [];
    for (const [key, model] of Object.entries(benchmarkData.roc_pr)) {
        const pr = model.pr[sentiment];
        if (!pr) continue;
        datasets.push({
            label: `${model.name} (AUC=${pr.auc})`,
            data: pr.recall.map((x, i) => ({ x, y: pr.precision[i] })),
            borderColor: MODEL_COLORS[key] || '#888',
            fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2,
        });
    }

    prChart = new Chart(ctx, {
        type: 'scatter', data: { datasets },
        options: {
            responsive: true, showLine: true,
            scales: {
                x: { title: { display: true, text: 'Recall', color: '#8892a8' }, min: 0, max: 1, ticks: { color: '#8892a8' }, grid: { color: '#243056' } },
                y: { title: { display: true, text: 'Precision', color: '#8892a8' }, min: 0, max: 1, ticks: { color: '#8892a8' }, grid: { color: '#243056' } },
            },
            plugins: { legend: { labels: { color: '#8892a8', font: { size: 11 } } } },
        },
    });
}

// ===================== LEARNING CURVES =====================

let lcChart = null;
function renderLearningCurves(data) {
    const ctx = document.getElementById('learning-chart').getContext('2d');
    if (lcChart) lcChart.destroy();

    const datasets = [];
    for (const [key, lc] of Object.entries(data)) {
        const color = MODEL_COLORS[key] || '#888';
        datasets.push({
            label: `${lc.name} (Train)`,
            data: lc.train_sizes.map((x, i) => ({ x, y: lc.train_mean[i] * 100 })),
            borderColor: color, fill: false, tension: 0.3, pointRadius: 3, borderWidth: 2,
        });
        datasets.push({
            label: `${lc.name} (Val)`,
            data: lc.train_sizes.map((x, i) => ({ x, y: lc.val_mean[i] * 100 })),
            borderColor: color, borderDash: [5, 5],
            fill: false, tension: 0.3, pointRadius: 3, borderWidth: 2,
        });
    }

    lcChart = new Chart(ctx, {
        type: 'scatter', data: { datasets },
        options: {
            responsive: true, showLine: true,
            scales: {
                x: { title: { display: true, text: 'Training Samples', color: '#8892a8' }, ticks: { color: '#8892a8' }, grid: { color: '#243056' } },
                y: { title: { display: true, text: 'Accuracy %', color: '#8892a8' }, min: 50, max: 105, ticks: { color: '#8892a8' }, grid: { color: '#243056' } },
            },
            plugins: { legend: { labels: { color: '#8892a8', font: { size: 11 } } } },
        },
    });
}

// ===================== EXPLAINABILITY (GLOBAL) =====================

function showExplainModel(modelKey, el) {
    if (!explainData) return;
    el.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');

    const container = document.getElementById('explain-global');
    const model = explainData[modelKey];
    if (!model) { container.innerHTML = '<p class="subtitle">No data for this model.</p>'; return; }

    if (model.type === 'coefficients' || model.type === 'probability') {
        container.innerHTML = Object.entries(model.classes).map(([cls, data]) => {
            const features = data.positive_features || [];
            const maxW = features.length ? Math.abs(features[0].weight) : 1;
            return `<div class="explain-class-card">
                <h4 class="sentiment-badge sentiment-${cls}" style="font-size:0.8rem">${cls.toUpperCase()}</h4>
                <p class="subtitle">Top features driving "${cls}" predictions:</p>
                ${features.slice(0, 12).map(f => {
                    const pct = Math.min(100, (Math.abs(f.weight) / maxW) * 100);
                    return `<div class="word-bar-row">
                        <span class="word-bar-label">${f.word}</span>
                        <div class="word-bar-fill-bg"><div class="word-bar-fill" style="width:${pct}%;background:${SENT_COLORS[cls] || '#6366f1'}"></div></div>
                        <span class="word-bar-count">${f.weight}</span>
                    </div>`;
                }).join('')}
            </div>`;
        }).join('');
    } else if (model.type === 'importance') {
        const features = model.features || [];
        const maxW = features.length ? features[0].weight : 1;
        container.innerHTML = `<div class="explain-class-card">
            <h4>Global Feature Importance</h4>
            <p class="subtitle">Features ranked by importance across all classes:</p>
            ${features.slice(0, 15).map(f => {
                const pct = Math.min(100, (f.weight / maxW) * 100);
                return `<div class="word-bar-row">
                    <span class="word-bar-label">${f.word}</span>
                    <div class="word-bar-fill-bg"><div class="word-bar-fill" style="width:${pct}%;background:#6366f1"></div></div>
                    <span class="word-bar-count">${f.weight}</span>
                </div>`;
            }).join('')}
        </div>`;
    }
}

// ===================== COMPARE =====================

async function compareTexts() {
    const textA = document.getElementById('compare-a').value.trim();
    const textB = document.getElementById('compare-b').value.trim();
    if (!textA || !textB) return;
    const res = await fetch('/api/compare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_a: textA, text_b: textB }),
    });
    const data = await res.json();
    renderCompareResult('compare-a-result', data.a);
    renderCompareResult('compare-b-result', data.b);
    document.getElementById('compare-results').classList.remove('hidden');
}

function renderCompareResult(id, result) {
    const el = document.getElementById(id);
    const ens = result.predictions.ensemble || {};
    const entities = result.entities || [];
    const lang = result.language;
    let confHTML = '';
    if (ens.confidence) {
        confHTML = ['negative', 'neutral', 'positive'].map(s => `
            <div class="conf-row"><span class="conf-label">${s}</span>
            <div class="conf-bar-bg"><div class="conf-bar ${s}" style="width: ${ens.confidence[s] || 0}%"></div></div>
            <span class="conf-val">${ens.confidence[s] || 0}%</span></div>`).join('');
    }
    const langBadge = lang && lang !== 'en' ? `<span class="lang-badge non-english">${LANG_NAMES[lang] || lang.toUpperCase()}</span>` : '';
    el.innerHTML = `<h4>${result.text}</h4>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
            <span class="sentiment-badge sentiment-${ens.label || ''}">${(ens.label || '').toUpperCase()}</span>
            ${langBadge}
            ${entities.map(e => `<span class="entity-tag">${e.ticker}</span>`).join('')}
        </div>
        <div class="confidence-bars">${confHTML}</div>`;
}

// ===================== NEWS =====================

async function fetchNews() {
    const btn = document.getElementById('news-btn');
    btn.disabled = true; btn.textContent = 'Fetching...';
    try {
        const res = await fetch('/api/news');
        const data = await res.json();
        renderNews(data.articles, data.cached);
    } catch (e) { console.error(e); }
    finally { btn.disabled = false; btn.textContent = 'Fetch Latest News'; }
}

function renderNews(articles, cached) {
    const c = document.getElementById('news-results');
    if (c) c.classList.remove('hidden');
    if (!articles.length) { c.innerHTML = '<p class="subtitle">No articles found.</p>'; return; }
    const cacheNote = cached ? '<p class="subtitle" style="margin-bottom:10px">Showing cached results (auto-refreshed every 30 min)</p>' : '';
    c.innerHTML = cacheNote + articles.map(a => `
        <div class="news-item">
            <div class="news-content">
                <div class="news-title">
                    <a href="${a.link}" target="_blank" rel="noopener">${a.title}</a>
                    ${a.analyzed_body ? '<span style="font-size:0.68rem;background:#10b98122;color:#10b981;border-radius:4px;padding:1px 6px;margin-left:6px;font-weight:600">FULL ARTICLE</span>' : ''}
                </div>
                ${a.snippet && a.analyzed_body ? `<div class="news-snippet">${a.snippet}</div>` : ''}
                <div class="news-meta">
                    <span>${a.source}</span>
                    ${(a.entities || []).map(e => `<span class="entity-tag">${e.ticker}</span>`).join('')}
                </div>
            </div>
            <div class="news-right">
                <span class="sentiment-badge sentiment-${a.sentiment}">${(a.sentiment || '?').toUpperCase()}</span>
            </div>
        </div>`).join('');
}

// ===================== BATCH =====================

function switchBatchTab(tab, el) {
    document.querySelectorAll('.batch-tabs .tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('batch-paste').classList.toggle('hidden', tab !== 'paste');
    document.getElementById('batch-upload').classList.toggle('hidden', tab !== 'upload');
}

async function batchAnalyze() {
    const texts = document.getElementById('batch-input').value.trim().split('\n').filter(t => t.trim());
    if (!texts.length) return;
    const res = await fetch('/api/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texts }) });
    const data = await res.json();
    lastBatchResults = data.results;
    renderBatchResults(data.results);
}

async function uploadCSV() {
    const file = document.getElementById('csv-file').files[0];
    if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    lastBatchResults = data.results;
    renderBatchResults(data.results);
}

function renderBatchResults(results) {
    const c = document.getElementById('batch-results');
    if (c) c.classList.remove('hidden');
    document.getElementById('batch-actions').classList.remove('hidden');
    document.getElementById('batch-count').textContent = `${results.length} results`;
    c.innerHTML = results.map(r => {
        const ens = r.predictions?.ensemble || {};
        const entities = r.entities || [];
        const lang = r.language;
        const langBadge = lang && lang !== 'en' ? `<span class="lang-badge non-english" style="font-size:0.7rem">${lang.toUpperCase()}</span>` : '';
        return `<div class="batch-row">
            <span class="batch-text" title="${r.text}">${r.text}</span>
            ${langBadge}
            <div class="batch-entities">${entities.map(e => `<span class="entity-tag">${e.ticker}</span>`).join('')}</div>
            <span class="sentiment-badge sentiment-${ens.label}">${ens.label || '?'}</span>
        </div>`;
    }).join('');
}

async function exportCSV() {
    if (!lastBatchResults.length) return;
    const res = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ results: lastBatchResults }) });
    const blob = await res.blob();
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'sentiment_results.csv'; a.click();
}

// ===================== STATS & CHARTS =====================

function renderStats(stats) {
    const row = document.getElementById('stats-row');
    const items = [
        { value: stats.train_size, label: 'Training' },
        { value: stats.test_size, label: 'Test' },
        { value: stats.label_distribution.negative, label: 'Negative' },
        { value: stats.label_distribution.neutral, label: 'Neutral' },
        { value: stats.label_distribution.positive, label: 'Positive' },
    ];
    row.innerHTML = items.map(i => `<div class="stat-card"><div class="stat-value">${(i.value || 0).toLocaleString()}</div><div class="stat-label">${i.label}</div></div>`).join('');
}

function renderDistChart(dist) {
    new Chart(document.getElementById('dist-chart').getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{ data: [dist.negative, dist.neutral, dist.positive], backgroundColor: ['#ef4444', '#eab308', '#22c55e'], borderColor: '#131a2b', borderWidth: 3 }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#8892a8', padding: 14, font: { size: 12 } } } } },
    });
}

function showWords(sentiment, el) {
    document.querySelectorAll('.word-tabs .tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    renderWordBars(sentiment);
}

function renderWordBars(sentiment) {
    if (!appData) return;
    const words = appData.top_words[sentiment];
    const c = document.getElementById('word-bars');
    const entries = Object.entries(words);
    const max = Math.max(...entries.map(e => e[1]));
    c.innerHTML = entries.map(([word, count]) => `
        <div class="word-bar-row"><span class="word-bar-label">${word}</span>
        <div class="word-bar-fill-bg"><div class="word-bar-fill" style="width:${(count / max) * 100}%;background:${SENT_COLORS[sentiment]}"></div></div>
        <span class="word-bar-count">${count}</span></div>`).join('');
}

function renderSampleChips(samples) {
    document.getElementById('sample-chips').innerHTML = samples.map(s => {
        const short = s.length > 40 ? s.substring(0, 40) + '...' : s;
        return `<span class="chip" onclick="useSample(this)" data-text="${s.replace(/"/g, '&quot;')}">${short}</span>`;
    }).join('');
}
function useSample(el) { document.getElementById('text-input').value = el.dataset.text; }

// ===================== MODEL CARDS =====================

function renderModelCards(models) {
    const c = document.getElementById('model-cards');
    if (!c) return;
    c.innerHTML = '';
    for (const [key, m] of Object.entries(models)) {
        const card = document.createElement('div');
        card.className = 'model-card';
        const accClass = m.accuracy >= 80 ? 'high' : m.accuracy >= 60 ? 'mid' : 'low';
        const cm = m.confusion_matrix;
        const labels = ['Neg', 'Neu', 'Pos'];
        const metricRows = ['negative', 'neutral', 'positive'].map(cls => {
            const r = m.report?.[cls];
            return r ? `<tr><td>${cls}</td><td>${r.precision}</td><td>${r.recall}</td><td>${r['f1-score']}</td><td>${r.support}</td></tr>` : '';
        }).join('');
        const cvHTML = m.cv ? `<div class="cv-tag">5-Fold CV: ${m.cv.mean}% ± ${m.cv.std}%</div>` : '';
        card.innerHTML = `
            <div class="model-card-top">
                <div>
                    <div class="model-name">${m.name}</div>
                    ${cvHTML}
                </div>
                <div>
                    <div class="accuracy-pill ${accClass}">${m.accuracy}%</div>
                    <div style="font-size:0.7rem;color:var(--text-muted);text-align:right">accuracy</div>
                </div>
            </div>
            <table class="metrics-table"><thead><tr><th></th><th>Prec</th><th>Recall</th><th>F1</th><th>N</th></tr></thead><tbody>${metricRows}</tbody></table>
            <table class="cm-table"><thead><tr><th></th>${labels.map(l => `<th>${l}</th>`).join('')}</tr></thead>
            <tbody>${cm.map((row, i) => `<tr><th>${labels[i]}</th>${row.map((v, j) => `<td class="cm-cell ${i === j ? 'cm-diagonal' : ''}">${v}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
        c.appendChild(card);
    }
}

// ===================== HISTORY =====================

let trendChart = null;
async function loadHistory() {
    const res = await fetch('/api/history');
    const data = await res.json();
    renderTrendChart(data.trend);
    renderHistoryList(data.history);
}

function renderTrendChart(trend) {
    const ctx = document.getElementById('trend-chart').getContext('2d');
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{ label: 'Predictions', data: [trend.negative || 0, trend.neutral || 0, trend.positive || 0], backgroundColor: ['#ef4444', '#eab308', '#22c55e'], borderRadius: 6 }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#8892a8' }, grid: { display: false } },
                y: { ticks: { color: '#8892a8' }, grid: { color: '#243056' }, beginAtZero: true }
            }
        },
    });
}

function renderHistoryList(history) {
    const c = document.getElementById('history-list');
    if (!history.length) { c.innerHTML = '<p class="subtitle">No predictions yet.</p>'; return; }
    c.innerHTML = history.slice(0, 30).map(h => {
        const time = h.timestamp ? new Date(h.timestamp).toLocaleString() : '';
        const ents = Array.isArray(h.entities) ? h.entities : [];
        return `<div class="history-item">
            <span class="sentiment-badge sentiment-${h.ensemble_label}">${h.ensemble_label || '?'}</span>
            <span class="history-text">${h.text}</span>
            ${ents.map(e => `<span class="entity-tag">${e.ticker}</span>`).join('')}
            <span class="history-time">${time}</span>
        </div>`;
    }).join('');
}

// ===================== EVENTS =====================

document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && document.activeElement.id === 'text-input') {
        e.preventDefault(); analyzeText();
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const ua = document.getElementById('upload-area');
    if (!ua) return;
    ua.addEventListener('dragover', e => { e.preventDefault(); ua.style.borderColor = 'var(--accent)'; });
    ua.addEventListener('dragleave', () => { ua.style.borderColor = ''; });
    ua.addEventListener('drop', e => {
        e.preventDefault(); ua.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file) {
            const dt = new DataTransfer(); dt.items.add(file);
            document.getElementById('csv-file').files = dt.files;
            uploadCSV();
        }
    });
});
