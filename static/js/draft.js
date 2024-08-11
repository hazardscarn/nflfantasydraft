console.log('Draft script loaded');

let draftPosition = parseInt(new URLSearchParams(window.location.search).get('position'));
let currentRound = 1;
let currentPick = 1;
let draftBoard = Array(18).fill().map(() => Array(12).fill(null));
let availablePlayers = [];
let recommendations = [];

console.log(`Draft position: ${draftPosition}`);

function initializeDraftBoard() {
    console.log('Initializing draft board');
    const board = document.getElementById('draft-board');
    if (!board) {
        console.error('Draft board element not found');
        return;
    }
    for (let i = 0; i < 18; i++) {
        for (let j = 0; j < 12; j++) {
            const cell = document.createElement('div');
            cell.className = 'draft-cell';
            cell.id = `cell-${i}-${j}`;
            cell.innerHTML = `
                <div>R${i+1} P${j+1}</div>
                <div class="player-name">-</div>
            `;
            board.appendChild(cell);
        }
    }
    console.log('Draft board initialized');
}

async function fetchAvailablePlayers() {
    console.log('Fetching available players');
    try {
        const response = await fetch('/api/available_players');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log(`Fetched ${data.length} players`);
        availablePlayers = data;
        displayAvailablePlayers();
        await getRecommendations();
    } catch (error) {
        console.error('Error fetching available players:', error);
        document.getElementById('available-players-table').innerHTML = `<p>Error loading players: ${error.message}</p>`;
    }
}

function displayAvailablePlayers() {
    console.log('Displaying available players');
    const tbody = document.querySelector('#available-players-table tbody');
    if (!tbody) {
        console.error('Available players table body not found');
        return;
    }
    tbody.innerHTML = '';
    availablePlayers.forEach(player => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${player.player || 'N/A'}</td>
            <td>${player.pos || 'N/A'}</td>
            <td>${player.ADP !== null ? parseFloat(player.ADP).toFixed(1) : 'N/A'}</td>
            <td>${player.ppr_projection !== null ? parseFloat(player.ppr_projection).toFixed(1) : 'N/A'}</td>
            <td><button onclick="handleDraftPick('${player.player}')">Draft</button></td>
        `;
        tbody.appendChild(row);
    });
    console.log(`Displayed ${availablePlayers.length} players`);
}

async function getRecommendations() {
    console.log('Getting recommendations');
    try {
        const response = await fetch('/api/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                team: draftBoard.map(round => round[draftPosition - 1]).filter(Boolean).map(player => player.player),
                available_players: availablePlayers.map(player => player.player),
                round_num: currentRound
            }),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        recommendations = await response.json();
        console.log(`Received ${recommendations.length} recommendations`);
        displayRecommendations();
    } catch (error) {
        console.error('Error getting recommendations:', error);
        document.getElementById('recommendations').innerHTML = `<p>Error loading recommendations: ${error.message}</p>`;
    }
}

function handleDraftPick(playerName) {
    const player = availablePlayers.find(p => p.player === playerName);
    if (!player) return;

    const col = (currentPick - 1) % 12;
    const row = currentRound - 1;
    draftBoard[row][col] = player;
    document.getElementById(`cell-${row}-${col}`).innerHTML = `
        <div>R${row+1} P${col+1}</div>
        <div class="player-name">${player.player} (${player.pos})</div>
    `;

    availablePlayers = availablePlayers.filter(p => p.player !== playerName);
    displayAvailablePlayers();

    if (currentPick % 12 === 0) {
        currentRound++;
    }
    currentPick++;

    updateDraftStatus();

    if ((currentPick - 1) % 12 === draftPosition - 1) {
        getRecommendations();
    }
}



function updateDraftStatus() {
    const cells = document.querySelectorAll('.draft-cell');
    cells.forEach(cell => cell.classList.remove('current-pick'));
    
    const currentCol = (currentPick - 1) % 12;
    const currentRow = currentRound - 1;
    const currentCell = document.getElementById(`cell-${currentRow}-${currentCol}`);
    if (currentCell) {
        currentCell.classList.add('current-pick');
        currentCell.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    console.log('Window loaded');
    initializeDraftBoard();
    fetchAvailablePlayers();
    updateDraftStatus();
    setupSearch();
};