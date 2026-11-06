const messagesArea = document.getElementById('messagesArea');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');

function formatText(text) {
    if (!text) return '';
    // Escape basic HTML to prevent injection
    const escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Convert bullet points and lines to clean HTML paragraphs
    return escaped
        .split('\n\n')
        .map(para => {
            const lines = para.split('\n').map(line => {
                if (line.trim().startsWith('* ') || line.trim().startsWith('- ') || line.trim().startsWith('• ')) {
                    return `<li>${line.trim().substring(2)}</li>`;
                }
                return line;
            }).join('<br>');

            if (lines.includes('<li>')) {
                return `<ul>${lines.replace(/<br>/g, '')}</ul>`;
            }
            return `<p>${lines}</p>`;
        })
        .join('');
}

function appendMessage(sender, text) {
    const isUser = sender === 'user';
    const messageBubble = document.createElement('div');
    messageBubble.className = `message-bubble ${isUser ? 'user-message' : 'bot-message'}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = isUser ? `<p>${text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>` : formatText(text);

    messageBubble.appendChild(avatar);
    messageBubble.appendChild(content);

    messagesArea.appendChild(messageBubble);
    scrollToBottom();
}

function scrollToBottom() {
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

function showTyping(show) {
    typingIndicator.style.display = show ? 'flex' : 'none';
    if (show) {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
}

async function handleSend(event) {
    if (event) event.preventDefault();

    const query = userInput.value.trim();
    if (!query) return;

    // Append user message
    appendMessage('user', query);
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;
    showTyping(true);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: query })
        });

        const data = await response.json();
        showTyping(false);

        if (response.ok && data.reply) {
            appendMessage('bot', data.reply);
        } else {
            appendMessage('bot', data.error || 'Sorry, something went wrong while fetching the answer.');
        }
    } catch (error) {
        showTyping(false);
        appendMessage('bot', '⚠️ Network error: Could not reach the server. Please try again.');
        console.error('Chat error:', error);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
        scrollToBottom();
    }
}

function sendQuickPrompt(button) {
    userInput.value = button.textContent.trim();
    handleSend();
}

