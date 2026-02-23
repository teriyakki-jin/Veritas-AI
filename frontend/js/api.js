const API_BASE_URL = 'http://127.0.0.1:8000';

export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('Health check failed');
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

export async function verifyClaim(claim, topK = 3) {
    try {
        const response = await fetch(`${API_BASE_URL}/verify`, {
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
        console.error('API Error:', error);
        throw error;
    }
}

export async function verifyBatch(claims, topK = 3) {
    try {
        const response = await fetch(`${API_BASE_URL}/verify/batch`, {
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
        console.error('API Error:', error);
        throw error;
    }
}
