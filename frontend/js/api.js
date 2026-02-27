function resolveApiBaseUrl() {
    const configured = window.__APP_CONFIG__?.apiBaseUrl || window.API_BASE_URL || '';
    if (!configured) return '';
    return configured.replace(/\/+$/, '');
}

const API_BASE_URL = resolveApiBaseUrl();

function endpoint(path) {
    return `${API_BASE_URL}${path}`;
}

export async function checkHealth() {
    try {
        const response = await fetch(endpoint('/health'));
        if (!response.ok) throw new Error('Health check failed');
        return await response.json();
    } catch (error) {
        throw error;
    }
}

export async function verifyClaim(claim, topK = 3) {
    try {
        const response = await fetch(endpoint('/verify'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ claim, top_k_evidence: topK })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'Verification failed');
        }

        return await response.json();
    } catch (error) {
        throw error;
    }
}

export async function verifyBatch(claims, topK = 3) {
    try {
        const response = await fetch(endpoint('/verify/batch'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ claims, top_k_evidence: topK })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'Batch verification failed');
        }

        return await response.json();
    } catch (error) {
        throw error;
    }
}

export async function analyzeArticle({ url = '', articleText = '', topK = 3, maxClaims = 5 } = {}) {
    try {
        const payload = {
            top_k_evidence: topK,
            max_claims: maxClaims
        };

        if (url && url.trim()) payload.url = url.trim();
        if (articleText && articleText.trim()) payload.article_text = articleText.trim();

        const response = await fetch(endpoint('/analyze/article'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'Article analysis failed');
        }

        return await response.json();
    } catch (error) {
        throw error;
    }
}
