// Existing variables
let draftPosition = parseInt(new URLSearchParams(window.location.search).get('position'));
let currentRound = 1;
let currentPick = 1;
let draftBoard = Array(18).fill().map(() => Array(12).fill(null));
let availablePlayers = [];
let recommendations = [];
let isCurrentUserTurn = false;

function saveDraftState() {
    const draftState = {
        draftPosition,
        currentRound,
        currentPick,
        draftBoard,
        availablePlayers
    };
    localStorage.setItem('draftState', JSON.stringify(draftState));
}

function loadDraftState() {
    const savedState = localStorage.getItem('draftState');
    if (savedState) {
        const state = JSON.parse(savedState);
        draftPosition = state.draftPosition;
        currentRound = state.currentRound;
        currentPick = state.currentPick;
        draftBoard = state.draftBoard;
        availablePlayers = state.availablePlayers;
        
        redrawDraftBoard();
        displayAvailablePlayers();
        updateDraftStatus();
    }
}

function redrawDraftBoard() {
    const board = document.getElementById('draft-board');
    board.innerHTML = '';
    for (let i = 0; i < 18; i++) {
        for (let j = 0; j < 12; j++) {
            const cell = document.createElement('div');
            cell.className = 'draft-cell';
            cell.id = `cell-${i}-${j}`;
            const player = draftBoard[i][j];
            if (player) {
                cell.innerHTML = `
                    <div class="pick-number">${i+1}.${j+1}</div>
                    <div class="player-name">${player.player} (${player.pos})</div>
                `;
                cell.classList.add(player.pos);
            } else {
                cell.innerHTML = `
                    <div class="pick-number">${i+1}.${j+1}</div>
                    <div class="player-name">-</div>
                `;
            }
            board.appendChild(cell);
        }
    }
}

function resetDraft() {
    localStorage.removeItem('draftState');
    location.reload();
}

function initializeDraftBoard() {
    const board = document.getElementById('draft-board');
    for (let i = 0; i < 18; i++) {
        for (let j = 0; j < 12; j++) {
            const cell = document.createElement('div');
            cell.className = 'draft-cell';
            cell.id = `cell-${i}-${j}`;
            cell.innerHTML = `
                <div class="pick-number">${i+1}.${j+1}</div>
                <div class="player-name">-</div>
            `;
            board.appendChild(cell);
        }
    }
}

async function fetchAvailablePlayers() {
    try {
        const response = await fetch('/api/available_players');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        availablePlayers = await response.json();
        displayAvailablePlayers();
        updateDraftStatus();
    } catch (error) {
        console.error('Error fetching available players:', error);
        document.getElementById('available-players-table').innerHTML = `<p>Error loading players: ${error.message}</p>`;
    }
}

function scrollToDraftPick(row, col) {
    const cell = document.getElementById(`cell-${row}-${col}`);
    if (cell) {
        cell.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    }
}

function displayAvailablePlayers() {
    const tbody = document.querySelector('#available-players-table tbody');
    tbody.innerHTML = '';
    availablePlayers.forEach(player => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${player.player || 'N/A'}</td>
            <td>${player.pos || 'N/A'}</td>
            <td>${player.ADP !== null ? parseFloat(player.ADP).toFixed(1) : 'N/A'}</td>
            <td>${player.ppr_projection !== null ? parseFloat(player.ppr_projection).toFixed(1) : 'N/A'}</td>
            <td><button onclick="handleDraftPick('${escapePlayerName(player.player)}')">Draft</button></td>
        `;
        tbody.appendChild(row);
    });
}

function escapePlayerName(name) {
    return name.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

async function getRecommendations() {
    if (!isCurrentUserTurn) {
        clearRecommendations();
        return;
    }

    try {
        const pick_number = (currentRound - 1) * 12 + currentPick;
        const response = await fetch('/api/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                team: draftBoard.map(round => round[draftPosition - 1]).filter(Boolean).map(player => player.player),
                available_players: availablePlayers.map(player => player.player),
                round_num: currentRound,
                pick_number: pick_number
            }),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        recommendations = await response.json();
        displayRecommendations();
    } catch (error) {
        console.error('Error getting recommendations:', error);
        document.getElementById('recommendation-cards').innerHTML = `<p>Error loading recommendations: ${error.message}</p>`;
    }
}

function displayRecommendations() {
    const recommendationsDiv = document.getElementById('recommendation-cards');
    recommendationsDiv.innerHTML = '';
    if (isCurrentUserTurn) {
        recommendations.forEach(player => {
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <h3>${player.player} (${player.pos})</h3>
                <p class="reward-score">Reward Score: ${player.q_value.toFixed(2)}</p>
                <p>Projection: ${player.ppr_projection.toFixed(2)}</p>
                <p>ADP: ${player.ADP.toFixed(1)}</p>
                <button class="draft-button" onclick="handleDraftPick('${escapePlayerName(player.player)}')">Draft</button>
            `;
            recommendationsDiv.appendChild(card);
        });
    } else {
        clearRecommendations();
    }
}

function clearRecommendations() {
    const recommendationsDiv = document.getElementById('recommendation-cards');
    recommendationsDiv.innerHTML = '<p>Waiting for your turn...</p>';
}

function handleDraftPick(playerName) {
    const player = availablePlayers.find(p => p.player === playerName);
    if (!player) return;

    const col = getCurrentPickColumn();
    const row = currentRound - 1;
    draftBoard[row][col] = player;
    const cell = document.getElementById(`cell-${row}-${col}`);
    
    cell.innerHTML = `
        <div class="pick-number">${row+1}.${col+1}</div>
        <div class="player-name">${player.player} (${player.pos})</div>
    `;
    cell.className = `draft-cell ${player.pos}`;
    scrollToDraftPick(row, col);
    availablePlayers = availablePlayers.filter(p => p.player !== playerName);
    displayAvailablePlayers();

    currentPick++;
    if (currentPick > 12) {
        currentRound++;
        currentPick = 1;
    }

    updateDraftStatus();
    saveDraftState();
}

function isUserTurn() {
    return getCurrentPickColumn() === draftPosition - 1;
}

function getCurrentPickColumn() {
    if (currentRound % 2 === 1) {
        return currentPick - 1;
    } else {
        return 12 - currentPick;
    }
}

function updateDraftStatus() {
    const cells = document.querySelectorAll('.draft-cell');
    cells.forEach(cell => cell.classList.remove('current-pick'));
    
    const currentCol = getCurrentPickColumn();
    const currentRow = currentRound - 1;
    const currentCell = document.getElementById(`cell-${currentRow}-${currentCol}`);
    if (currentCell) {
        currentCell.classList.add('current-pick');
        scrollToDraftPick(currentRow, currentCol);
    }

    isCurrentUserTurn = isUserTurn();
    if (isCurrentUserTurn) {
        getRecommendations();
    } else {
        clearRecommendations();
    }
}

function setupSearch() {
    const searchInput = document.getElementById('player-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const rows = document.querySelectorAll('#available-players-table tbody tr');
            rows.forEach(row => {
                const playerName = row.querySelector('td').textContent.toLowerCase();
                row.style.display = playerName.includes(searchTerm) ? '' : 'none';
            });
        });
    }
}

window.onload = function() {
    initializeDraftBoard();
    loadDraftState();
    if (availablePlayers.length === 0) {
        fetchAvailablePlayers();
    }
    setupSearch();
    
    document.getElementById('reset-draft').addEventListener('click', resetDraft);
};