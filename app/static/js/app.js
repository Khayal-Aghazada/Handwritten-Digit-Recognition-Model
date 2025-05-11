const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
let drawing  = false;

// Throttle settings
let lastGuessTime = 0;
const GUESS_INTERVAL = 500; // ms

// Stroke style
ctx.lineWidth   = 20;
ctx.lineCap     = 'round';
ctx.strokeStyle = '#000';

// Fill background white on startup
ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Render into left (0–4) and right (5–9)
function renderResults(probs) {
  // Find the overall top digit
  const entries = Object.entries(probs);
  const bestDigit = entries.reduce((a, b) => a[1] > b[1] ? a : b)[0];

  // Build left list (0–4)
  let leftHTML = '';
  for (let d = 0; d <= 4; d++) {
    const p = probs[d];
    const cls = d.toString() === bestDigit ? 'high' : '';
    leftHTML += `<li class="${cls}">${d}: ${(p*100).toFixed(1)}%</li>`;
  }

  // Build right list (5–9)
  let rightHTML = '';
  for (let d = 5; d <= 9; d++) {
    const p = probs[d];
    const cls = d.toString() === bestDigit ? 'high' : '';
    rightHTML += `<li class="${cls}">${d}: ${(p*100).toFixed(1)}%</li>`;
  }

  document.getElementById('result-left').innerHTML  = leftHTML;
  document.getElementById('result-right').innerHTML = rightHTML;
}

function sendGuess() {
  const dataURL = canvas.toDataURL('image/png');
  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({image: dataURL})
  })
    .then(res => res.json())
    .then(renderResults)
    .catch(console.error);
}

// Drawing events
canvas.addEventListener('pointerdown', e => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();

  // Throttled live guess
  const now = Date.now();
  if (now - lastGuessTime > GUESS_INTERVAL) {
    sendGuess();
    lastGuessTime = now;
  }
});

canvas.addEventListener('pointerup', () => {
  drawing = false;
  sendGuess();  // final guess on lift
});

canvas.addEventListener('pointerleave', () => {
  if (drawing) {
    drawing = false;
    sendGuess();
  }
});

// Clear button
document.getElementById('clear').onclick = () => {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result-left').innerHTML = '';
  document.getElementById('result-right').innerHTML = '';
  lastGuessTime = 0;
};


function renderResults(probs) {
  // Determine overall best digit
  const entries   = Object.entries(probs);
  const bestDigit = entries.reduce((a, b) => a[1] > b[1] ? a : b)[0];

  // Helper to build a list of digits [from d0 to dN]
  function buildList(start, end) {
    let html = '';
    for (let d = start; d <= end; d++) {
      const p    = probs[d];
      const pct  = (p * 100).toFixed(1);
      const cls  = d.toString() === bestDigit ? 'high' : '';
      html += `
        <li class="${cls}">
          ${d}: ${pct}%
          <div class="bar">
            <div class="bar-fill" style="width: ${pct}%"></div>
          </div>
        </li>`;
    }
    return html;
  }

  document.getElementById('result-left').innerHTML  = buildList(0, 4);
  document.getElementById('result-right').innerHTML = buildList(5, 9);
}
