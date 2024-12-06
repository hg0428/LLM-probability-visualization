// Color and visualization utilities
export function getProbabilityColor(prob) {
	const red = 255 - Math.floor(prob * 255);
	const green = Math.floor(prob * 255);
	return `rgba(${red}, ${green}, 0, 0.5)`;
}

export function getConfidenceLabel(confidence) {
	if (confidence >= 90) return "Very High";
	if (confidence >= 70) return "High";
	if (confidence >= 50) return "Moderate";
	if (confidence >= 30) return "Low";
	return "Very Low";
}

export function getConfidenceColor(confidence) {
	if (confidence >= 90) return "#15803d"; // Green
	if (confidence >= 70) return "#4d7c0f"; // Green-yellow
	if (confidence >= 50) return "#b45309"; // Orange
	if (confidence >= 30) return "#b91c1c"; // Red
	return "#7f1d1d"; // Dark red
}

export function calculateConfidence(tokens) {
	if (!tokens || tokens.length === 0) return 0;
	const probabilities = tokens.map((token) => {
		if (token.probabilities && token.probabilities.length > 0) {
			const chosenProb = token.probabilities.find(
				(p) => p.token === token.token
			);
			return chosenProb ? chosenProb.probability : 0;
		}
		return 0;
	});
	return Math.round(
		(probabilities.reduce((a, b) => a + b, 0) / probabilities.length) * 100
	);
}

// DOM utilities
export function createTokenSpan(token, prob, position, messageIndex, onClick) {
	const tokenSpan = document.createElement("span");
	tokenSpan.textContent = token;
	tokenSpan.className = "token";
	tokenSpan.dataset.position = position;
	if (messageIndex !== undefined) {
		tokenSpan.dataset.messageIndex = messageIndex;
	}
	tokenSpan.onclick = onClick;
	tokenSpan.style.backgroundColor = getProbabilityColor(prob);
	return tokenSpan;
}

export function createConfidenceIndicator(confidence) {
	const confidenceDiv = document.createElement("div");
	confidenceDiv.className = "confidence-indicator";
	confidenceDiv.innerHTML = `
        <div class="confidence-label">Confidence: 
            <span style="color: ${getConfidenceColor(confidence)}">
                ${getConfidenceLabel(confidence)} (${confidence}%)
            </span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%; background-color: ${getConfidenceColor(
		confidence
	)}"></div>
        </div>
    `;
	return confidenceDiv;
}

// Settings utilities
export function setupSettingsToggle(groupName) {
	const toggle = document.getElementById(`toggle-${groupName}`);
	const group = document.getElementById(`${groupName}-group`);

	toggle.addEventListener("change", () => {
		group.classList.toggle("disabled", !toggle.checked);
		if (!toggle.checked) {
			resetGroupValues(groupName);
		}
	});
}

function resetGroupValues(groupName) {
	const defaultValues = {
		temperature: "1.0",
		top_p: "1.0",
		min_p: "0",
		top_k: "0",
		repetition_penalty: "1.0",
		frequency_penalty: "0.0",
		presence_penalty: "0.0",
		dry_allowed_length: "1",
		dry_base: "2",
		dry_multiplier: "3",
		dry_range: "1024",
	};

	const group = document.getElementById(`${groupName}-group`);
	const inputs = group.querySelectorAll('input[type="number"]');
	inputs.forEach((input) => {
		const settingName = input.id.replace("-value", "");
		if (defaultValues[settingName]) {
			input.value = defaultValues[settingName];
		}
	});
}

// Modal utilities
export function showModal(modalId) {
	const modal = document.getElementById(modalId);
	if (modal) {
		modal.style.display = "block";
	}
}

export function closeModal(modalId) {
	const modal = document.getElementById(modalId);
	if (modal) {
		modal.style.display = "none";
	}
}
