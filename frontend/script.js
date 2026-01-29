/**
 * Professional mapping for strings. 
 * Required for compatibility with the Pipeline OneHotEncoder.
 */
const binaryFields = {
    rbc: { on: 'Normal', off: 'Abnormal', valOn: 'normal', valOff: 'abnormal' },
    pc: { on: 'Normal', off: 'Abnormal', valOn: 'normal', valOff: 'abnormal' },
    pcc: { on: 'Present', off: 'Not Present', valOn: 'present', valOff: 'notpresent' },
    ba: { on: 'Present', off: 'Not Present', valOn: 'present', valOff: 'notpresent' },
    htn: { on: 'No', off: 'Yes', valOn: 'no', valOff: 'yes' },
    dm: { on: 'No', off: 'Yes', valOn: 'no', valOff: 'yes' },
    cad: { on: 'No', off: 'Yes', valOn: 'no', valOff: 'yes' },
    pe: { on: 'No', off: 'Yes', valOn: 'no', valOff: 'yes' },
    ane: { on: 'No', off: 'Yes', valOn: 'no', valOff: 'yes' },
    appet: { on: 'Good', off: 'Poor', valOn: 'good', valOff: 'poor' }
};

// Initialize Toggle listeners for color and text changes
Object.keys(binaryFields).forEach(id => {
    const input = document.getElementById(id);
    const textSpan = document.getElementById(`${id}-text`);
    if (input) {
        input.addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            const field = binaryFields[id];
            textSpan.textContent = isChecked ? field.on : field.off;
            textSpan.className = `toggle-text ${isChecked ? 'text-green-400' : 'text-red-400'}`;
        });
    }
});

/**
 * Handle Form Submission to Backend API
 */
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const submitBtn = document.getElementById('submitBtn');
    const reportContainer = document.getElementById('reportContainer');
    
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing Clinical Data...';

    const getVal = (id) => parseFloat(document.getElementById(id).value);
    const getToggleVal = (id) => document.getElementById(id).checked ? binaryFields[id].valOn : binaryFields[id].valOff;

    // Construct JSON Payload
    const payload = {
        age: getVal('age'), bp: getVal('bp'), sg: getVal('sg'), al: getVal('al'), su: getVal('su'),
        bgr: getVal('bgr'), bu: getVal('bu'), sc: getVal('sc'), sod: getVal('sod'), pot: getVal('pot'),
        hemo: getVal('hemo'), pcv: getVal('pcv'), wc: getVal('wc'), rc: getVal('rc'),
        rbc: getToggleVal('rbc'), pc: getToggleVal('pc'), pcc: getToggleVal('pcc'), ba: getToggleVal('ba'),
        htn: getToggleVal('htn'), dm: getToggleVal('dm'), cad: getToggleVal('cad'),
        appet: getToggleVal('appet'), pe: getToggleVal('pe'), ane: getToggleVal('ane')
    };

    try {
        // Ensure your FastAPI is running on port 8000
        const response = await fetch('https://ckd-analysis-api.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (response.ok) {
            // Update Report UI
            const header = document.getElementById('predictionHeader');
            header.textContent = result.prediction.toUpperCase();
            header.className = `text-xl p-4 rounded-lg text-center font-bold mb-8 ${result.prediction.includes('Positive') ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300'}`;

            // --- START NEW CODE ---
            const recommendationDiv = document.getElementById('recommendationBox');
            if (recommendationDiv) {
                // We use 'result' because that is the variable name in your code
                recommendationDiv.innerText = result.recommendation; 
                recommendationDiv.classList.remove('hidden'); 
            }
            // --- END NEW CODE ---

            // Map results to the Lab Report table
            document.getElementById('insightTableBody').innerHTML = result.insights.map(item => `
                <tr class="border-b border-gray-700 hover:bg-gray-800 transition">
                    <td class="py-4 text-sm">${item.parameter}</td>
                    <td class="py-4 font-mono">${item.value}</td>
                    <td class="py-4 text-xs text-gray-500">${item.range}</td>
                    <td class="py-4 font-bold text-xs ${item.status === 'Abnormal' ? 'text-red-400' : 'text-green-400'}">${item.status}</td>
                </tr>
            `).join('');

            reportContainer.classList.remove('hidden');
            reportContainer.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (err) {
        alert("Connection Error: Is the Backend API running on http://127.0.0.1:8000?");
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Clinical Data';
    }
});