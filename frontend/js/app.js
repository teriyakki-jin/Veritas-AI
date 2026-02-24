import { verifyClaim, verifyBatch } from './api.js';

// DOM Elements
const claimInput = document.getElementById('claimInput');
const batchInput = document.getElementById('batchInput');
const singleVerifyBtn = document.getElementById('singleVerifyBtn');
const batchVerifyBtn = document.getElementById('batchVerifyBtn');
const resultsArea = document.getElementById('resultsArea');
const loadingSpinner = document.getElementById('loadingSpinner');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// State
let currentTab = 'single';

// Event Listeners
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

singleVerifyBtn.addEventListener('click', handleSingleVerify);
batchVerifyBtn.addEventListener('click', handleBatchVerify);

// Tab Switching
function switchTab(tabId) {
    currentTab = tabId;
    tabBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabId));
    tabContents.forEach(content => content.classList.toggle('active', content.id === `${tabId}-tab`));
    resultsArea.innerHTML = '';
}

// Single Verification Handler
async function handleSingleVerify() {
    const claim = claimInput.value.trim();
    if (!claim) return showAlert('Please enter a claim to verify.', 'error');

    setLoading(true);
    resultsArea.innerHTML = '';

    try {
        const result = await verifyClaim(claim);
        renderResult(result);
    } catch (error) {
        showAlert(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

// Batch Verification Handler
async function handleBatchVerify() {
    const text = batchInput.value.trim();
    if (!text) return showAlert('Please enter claims to verify.', 'error');

    const claims = text.split('\n').map(c => c.trim()).filter(c => c.length > 0);
    if (claims.length === 0) return showAlert('No valid claims found.', 'error');

    setLoading(true);
    resultsArea.innerHTML = '';

    try {
        const results = await verifyBatch(claims);
        results.forEach(renderResult);
    } catch (error) {
        showAlert(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

// UI Helpers
function setLoading(isLoading) {
    loadingSpinner.style.display = isLoading ? 'block' : 'none';
    singleVerifyBtn.disabled = isLoading;
    batchVerifyBtn.disabled = isLoading;
}

function showAlert(message, type = 'info') {
    const div = document.createElement('div');
    div.className = `error-msg ${type}`;
    div.textContent = message;
    resultsArea.prepend(div);
}

function getVerdictColorClass(verdict) {
    const v = verdict.toUpperCase();
    if (v.includes('TRUE') && !v.includes('HALF')) return 'verdict-true';
    if (v.includes('FALSE') && !v.includes('HALF')) return 'verdict-false';
    return 'verdict-mixed';
}

function highlightText(text, claim) {
    if (!text) return '';
    // Simple highlighting of claim words in text
    const words = claim.split(/\s+/).filter(w => w.length > 3);
    let highlighted = text;
    words.forEach(word => {
        const regex = new RegExp(`(${word})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark>$1</mark>');
    });
    return highlighted;
}

// Render Functions
function renderResult(data) {
    const card = document.createElement('div');
    card.className = 'result-card glass-panel'; // reusing glass-panel style for card

    const verdictClass = getVerdictColorClass(data.verdict);
    const scorePercent = (data.credibility_score * 100).toFixed(1);

    let modelDetailsHtml = '';
    for (const [model, details] of Object.entries(data.model_details)) {
        modelDetailsHtml += `
            <div class="model-chip">
                <strong>${model}</strong>
                <div class="val" style="color: ${getVerdictColorClass(details.predicted_label || '').includes('true') ? 'var(--success-color)' : 'var(--error-color)'}">
                    ${details.predicted_label || '-'}
                </div>
                <div style="font-size:0.8em; margin-top:0.3em; opacity:0.8">
                    Conf: ${(details.credibility_score * 100).toFixed(1)}%
                </div>
            </div>
        `;
    }

    let evidenceHtml = '';
    if (data.evidence && data.evidence.length > 0) {
        evidenceHtml = '<div class="evidence-list"><h4>Evidence</h4>';
        data.evidence.forEach((ev, idx) => {
            evidenceHtml += `
                <div class="evidence-item">
                    <div class="evidence-meta">
                        <span>Doc ID: ${ev.doc_id}</span>
                        <span>Score: ${ev.score.toFixed(4)}</span>
                    </div>
                    <div>${highlightText(ev.snippet, data.claim)}</div>
                </div>
            `;
        });
        evidenceHtml += '</div>';
    }

    card.innerHTML = `
        <div class="result-header">
            <h3 style="max-width: 70%;">${data.claim}</h3>
            <span class="verdict-badge ${verdictClass}">${data.verdict}</span>
        </div>
        
        <div class="score-container">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                <span>Credibility Score</span>
                <strong>${scorePercent}%</strong>
            </div>
            <div class="score-bar-container">
                <div class="score-bar-fill" style="width: ${scorePercent}%"></div>
            </div>
        </div>

        <div class="model-details">
            ${modelDetailsHtml}
        </div>

        ${evidenceHtml}
    `;

    resultsArea.appendChild(card);
}
