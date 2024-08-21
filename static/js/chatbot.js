let messageQueue = [];
let isProcessing = false;

function loadChatHistory() {
    const chatMessages = document.getElementById('draft-chat');
    if (!chatMessages) {
        console.error("Chat messages element not found");
        return;
    }
    const history = JSON.parse(sessionStorage.getItem('chatHistory')) || [];
    history.forEach(message => {
        const messageElement = createMessageElement(message.sender, message.content);
        chatMessages.appendChild(messageElement);
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function saveChatHistory(sender, content) {
    const history = JSON.parse(sessionStorage.getItem('chatHistory')) || [];
    history.push({ sender, content });
    sessionStorage.setItem('chatHistory', JSON.stringify(history));
}

function clearChatHistory() {
    sessionStorage.removeItem('chatHistory');
    const chatMessages = document.getElementById('draft-chat');
    if (chatMessages) {
        chatMessages.innerHTML = '';
    }
}
function createMessageElement(sender, content) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}-message`;
    const iconSrc = sender === 'user' ? '/static/images/user-icon.png' : '/static/images/bot-icon.png';
    messageElement.innerHTML = `
        <img src="${iconSrc}" alt="${sender}" class="chat-icon">
        <div class="message-content markdown-content">${marked.parse(content)}</div>
    `;
    return messageElement;
}

function sendDraftMessage() {
    const input = document.getElementById('draft-input');
    if (!input) {
        console.error("Draft input element not found");
        return;
    }
    const message = input.value.trim();
    if (message) {
        addMessageToChat('user', message);
        saveChatHistory('user', message);
        input.value = '';
        messageQueue.push(message);
        processQueue();
    }
}

function processQueue() {
    if (isProcessing || messageQueue.length === 0) return;
    isProcessing = true;
    const message = messageQueue.shift();
    fetchChatbotResponse(message);
}

function fetchChatbotResponse(message) {
    console.log("Fetching chatbot response for:", message);
    const chatMessages = document.getElementById('draft-chat');
    if (!chatMessages) {
        console.error("Chat messages element not found");
        return;
    }
    const responseElement = createMessageElement('bot', '');
    chatMessages.appendChild(responseElement);
    const textElement = responseElement.querySelector('.message-content');

    // Add loading indicator
    textElement.innerHTML = '<div class="loading-indicator">...</div>';

    fetch('/api/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: message }),
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let errorOccurred = false;

        function read() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    if (!errorOccurred) {
                        saveChatHistory('bot', fullResponse);
                    }
                    isProcessing = false;
                    processQueue();
                    return;
                }

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.chunk) {
                                fullResponse += data.chunk + '\n';
                                textElement.innerHTML = marked.parse(fullResponse);
                            } else if (data.references) {
                                const referencesElement = document.createElement('div');
                                referencesElement.className = 'references markdown-content';
                                referencesElement.innerHTML = marked.parse(data.references);
                                textElement.appendChild(referencesElement);
                                fullResponse += '\n\n' + data.references;
                            } else if (data.error) {
                                errorOccurred = true;
                                textElement.innerHTML += `<br><strong>Error:</strong> ${data.error}`;
                                fullResponse += `\n\nError: ${data.error}`;
                            }
                        } catch (e) {
                            console.error("Error parsing JSON:", e);
                            // Don't add the "Unable to parse response" error to the chat
                        }
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                });

                return read();
            });
        }

        return read();
    }).catch(error => {
        console.error("Fetch error:", error);
        textElement.innerHTML = '<strong>Error:</strong> Connection lost. Please try again.';
        isProcessing = false;
        processQueue();
    });
}

function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('draft-chat');
    if (!chatMessages) {
        console.error("Chat messages element not found");
        return;
    }
    const messageElement = createMessageElement(sender, message);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendYoutubeMessage() {
    const input = document.getElementById('youtube-input');
    const message = input.value.trim();
    if (message) {
        const selectedChannels = Array.from(document.querySelectorAll('input[name="youtube-channel"]:checked'))
            .map(checkbox => checkbox.value);

        if (selectedChannels.length === 0) {
            alert('Please select at least one YouTube channel');
            return;
        }

        addMessageToYoutubeChat('user', message);
        input.value = '';
        fetchYoutubeExpertResponse(message, selectedChannels);
    }
}

function addMessageToYoutubeChat(sender, message) {
    const chatMessages = document.getElementById('youtube-chat');
    if (!chatMessages) {
        console.error("YouTube chat messages element not found");
        return;
    }
    const messageElement = createMessageElement(sender, message);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function fetchYoutubeExpertResponse(message, channels) {
    console.log("Fetching YouTube expert response for:", message, "Channels:", channels);
    const chatMessages = document.getElementById('youtube-chat');
    if (!chatMessages) {
        console.error("YouTube chat messages element not found");
        return;
    }
    const responseElement = createMessageElement('bot', '');
    chatMessages.appendChild(responseElement);
    const textElement = responseElement.querySelector('.message-content');

    // Add loading indicator
    textElement.innerHTML = '<div class="loading-indicator">...</div>';

    fetch('/api/youtube_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: message, channels: channels }),
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        function read() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    return;
                }

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.chunk) {
                                fullResponse += data.chunk + '\n';
                                textElement.innerHTML = marked.parse(fullResponse);
                            } else if (data.references) {
                                const referencesElement = document.createElement('div');
                                referencesElement.className = 'references markdown-content';
                                referencesElement.innerHTML = marked.parse(data.references);
                                textElement.appendChild(referencesElement);
                                fullResponse += '\n\n' + data.references;
                            } else if (data.error) {
                                textElement.innerHTML += `<br><strong>Error:</strong> ${data.error}`;
                                fullResponse += `\n\nError: ${data.error}`;
                            }
                        } catch (e) {
                            console.error("Error parsing JSON:", e);
                        }
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                });

                return read();
            });
        }

        return read();
    }).catch(error => {
        console.error("Fetch error:", error);
        textElement.innerHTML = '<strong>Error:</strong> Connection lost. Please try again.';
    });
}

function findSimilarPlayers() {
    const input = document.getElementById('player-input');
    const playerName = input.value.trim();
    if (playerName) {
        fetch('/api/similar_players', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ player: playerName }),
        })
        .then(response => response.json())
        .then(data => {
            displaySimilarPlayers(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('similar-players-results').innerHTML = '<p>Error finding similar players. Please try again.</p>';
        });
        input.value = '';
    }
}

function displaySimilarPlayers(players) {
    const resultsDiv = document.getElementById('similar-players-results');
    resultsDiv.innerHTML = '';

    if (players.length === 0) {
        resultsDiv.innerHTML = '<p>No similar players found.</p>';
        return;
    }

    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th>Player</th>
                <th>Position</th>
                <th>Team</th>
                <th>Similarity Score</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
    `;

    const tbody = table.querySelector('tbody');
    players.forEach(player => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${player.player}</td>
            <td>${player.position}</td>
            <td>${player.team}</td>
            <td>${player.similarity_score.toFixed(2)}</td>
        `;
    });

    resultsDiv.appendChild(table);
}

function findSimilarPlayers() {
    const input = document.getElementById('player-input');
    const playerName = input.value.trim();
    if (playerName) {
        // Show spinner
        document.getElementById('spinner').classList.remove('hidden');
        // Clear previous results
        document.getElementById('similar-players-results').innerHTML = '';

        fetch('/api/similar_players', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ player: playerName }),
        })
        .then(response => response.json())
        .then(data => {
            displaySimilarPlayers(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('similar-players-results').innerHTML = '<p>Error finding similar players. Please try again.</p>';
        })
        .finally(() => {
            // Hide spinner
            document.getElementById('spinner').classList.add('hidden');
        });
        input.value = '';
    }
}

function displaySimilarPlayers(players) {
    const resultsDiv = document.getElementById('similar-players-results');
    resultsDiv.innerHTML = '';

    if (players.length === 0) {
        resultsDiv.innerHTML = '<p>No similar players found.</p>';
        return;
    }

    players.forEach(player => {
        const card = document.createElement('div');
        card.className = 'player-card';
        card.innerHTML = `
            <h3>${player.player}</h3>
            <p><strong>Position:</strong> ${player.position}</p>
            <p><strong>PPG:</strong> ${player.ppr_projection.toFixed(2)}</p>
            <div class="outlook-container">
                <h4>Outlook</h4>
                <div class="outlook-content markdown-content">${marked.parse(player.outlook)}</div>
            </div>
        `;
        resultsDiv.appendChild(card);
    });
}


// Add this function to handle the clear button click
function handleClearChatClick() {
    clearChatHistory();
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    const draftInput = document.getElementById('draft-input');
    const draftSendButton = document.getElementById('draft-send-button');
    const clearChatButton = document.getElementById('clear-chat-button');
    const youtubeSendButton = document.getElementById('youtube-send-button');
    const playerInput = document.getElementById('player-input');
    const playerSearchButton = document.getElementById('player-search-button');

    if (draftInput) {
        draftInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendDraftMessage();
            }
        });
    }

    if (draftSendButton) {
        draftSendButton.addEventListener('click', sendDraftMessage);
    }

    if (youtubeSendButton) {
        youtubeSendButton.addEventListener('click', sendYoutubeMessage);
    }

    if (playerInput) {
        playerInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                findSimilarPlayers();
            }
        });
    }

    if (playerSearchButton) {
        playerSearchButton.addEventListener('click', findSimilarPlayers);
    }


    if (clearChatButton) {
        clearChatButton.addEventListener('click', handleClearChatClick);
    }

    loadChatHistory();
});