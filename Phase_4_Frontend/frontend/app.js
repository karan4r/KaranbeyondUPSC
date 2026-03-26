document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatHistory = document.getElementById('chat-history');
    const apiKeyInput = document.getElementById('api-key');

    // Restore API key from localStorage
    const savedKey = localStorage.getItem('pw_openai_api_key');
    if (savedKey) {
        apiKeyInput.value = savedKey;
    }

    // Save API key on change
    apiKeyInput.addEventListener('change', (e) => {
        localStorage.setItem('pw_openai_api_key', e.target.value.trim());
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Append user HTML message
        appendMessage('user', query);
        chatInput.value = '';

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            const response = await fetch('http://127.0.0.1:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    api_key: apiKeyInput.value.trim() || null
                })
            });

            if (!response.ok) {
                let errMessage = 'Failed to fetch from the server';
                try {
                    const errStr = await response.text();
                    try {
                        const errData = JSON.parse(errStr);
                        errMessage = errData.detail || JSON.stringify(errData);
                    } catch (e) {
                        errMessage = `Server returned an error (${response.status}): ${errStr.substring(0, 100)}...`;
                    }
                } catch (e) {
                    errMessage = `Server returned status ${response.status}`;
                }
                throw new Error(errMessage);
            }

            const data = await response.json();
            removeMessage(typingId);
            appendMessage('bot', data.answer, data.fallback);
        } catch (error) {
            console.error('API Error:', error);
            removeMessage(typingId);
            appendMessage('bot', `⚠️ **Error connecting to the Backend:** ${error.message}.\\n\\nHave you started the FastAPI server with \`uvicorn main:app --reload\` ?`);
        }
    });

    function appendMessage(role, text, isFallback=false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role} slide-in`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.textContent = role === 'bot' ? '🤖' : '👤';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        let markdownHTML = marked.parse(text);
        
        if (role === 'bot' && isFallback && !apiKeyInput.value.trim()) {
            contentDiv.innerHTML = `<p style="color: #ffb74d; font-size: 0.9em; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1);">⚠️ <i>No API Key found. Returning exact snippets from Vector DB directly:</i></p>` + markdownHTML;
        } else {
            contentDiv.innerHTML = markdownHTML;
        }

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(contentDiv);

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = `message bot slide-in`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.textContent = '🤖';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.style.padding = '12px 20px'; // smaller padding for loader
        contentDiv.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
});
