document.getElementById('record-btn').addEventListener('click', async () => {
  const name = document.getElementById('name').value.trim();
  const age = document.getElementById('age').value.trim();
  const status = document.getElementById('status');
  const resultsSection = document.getElementById('results');
  const outputList = document.getElementById('output-list');

  if (!name || !age) {
    alert("Please enter both name and age.");
    return;
  }

  status.textContent = "⏳ Collecting data (audio & video)... Please wait.";

  const formData = new FormData();
  formData.append('name', name);
  formData.append('age', age);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json();

    status.textContent = "✅ Prediction complete!";
    resultsSection.hidden = false;

    outputList.innerHTML = `
      <li><strong>Name:</strong> ${result.name ?? 'N/A'}</li>
      <li><strong>Fused Probability:</strong> ${result.fused_proba?.toFixed(4) ?? 'N/A'}</li>
      <li><strong>Prediction:</strong> ${
        result.prediction === 1 ? "Parkinson's Detected" :
        result.prediction === 0 ? "No Parkinson's" :
        "Undetermined"
      }</li>
      <li><strong>Blink Rate:</strong> ${result.blink_rate ?? 'N/A'} blinks/min</li>
      <li><strong>Audio Probability:</strong> ${result.audio_proba?.toFixed(4) ?? 'N/A'}</li>
      <li><strong>Age & Blink Probability:</strong> ${result.age_proba?.toFixed(4) ?? 'N/A'}</li>
    `;
  } catch (error) {
    console.error(error);
    status.textContent = "❌ Failed to get prediction. Is the backend running?";
    resultsSection.hidden = true;
  }
});
