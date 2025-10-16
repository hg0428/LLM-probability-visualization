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
const END_TOKEN_MARKERS = ["<|im_end|>", "<|eot_id|>", "</s>", "<|endoftext|>"];

export function getTokenVisualMetadata(token = "", fallback = token) {
	const reference = fallback ?? token ?? "";
	const value = token ?? "";
	const metadata = {
		isInvisible: false,
		placeholder: null,
		ariaLabel: null,
		isEnd: false,
	};
	const isEmpty = value.length === 0;
	const whitespaceOnly = value.length > 0 && value.replace(/\s/g, "").length === 0;
	if (isEmpty || whitespaceOnly) {
		metadata.isInvisible = true;
		const source = reference.length ? reference : value;
		if (source.includes("\n")) {
			const count = Math.max(1, (source.match(/\n/g) || []).length);
			metadata.placeholder = count > 1 ? `↵×${count}` : "↵";
			metadata.ariaLabel = count > 1 ? `${count} newline tokens` : "Newline token";
		} else if (source.includes("\t")) {
			const count = Math.max(1, (source.match(/\t/g) || []).length);
			metadata.placeholder = count > 1 ? `⇥×${count}` : "⇥";
			metadata.ariaLabel = count > 1 ? `${count} tab tokens` : "Tab token";
		} else {
			const count = Math.max(1, source.length);
			metadata.placeholder = count > 1 ? `␠×${count}` : "␠";
			metadata.ariaLabel = count > 1 ? `${count} space tokens` : "Space token";
		}
	}
	const normalized = reference.trim();
	metadata.isEnd = END_TOKEN_MARKERS.some(
		(marker) => normalized === marker || normalized.endsWith(marker)
	);
	return metadata;
}

export function createTokenSpan(
	token,
	prob,
	position,
	messageIndex,
	onClick,
	fallbackToken = token
) {
	const tokenSpan = document.createElement("span");
	tokenSpan.className = "token";
	const metadata = getTokenVisualMetadata(token, fallbackToken);
	if (metadata.isInvisible) {
		tokenSpan.classList.add("token-empty");
		tokenSpan.dataset.placeholder = metadata.placeholder;
		tokenSpan.innerHTML = "&nbsp;";
		tokenSpan.dataset.actualToken = token;
		if (metadata.ariaLabel) {
			tokenSpan.setAttribute("aria-label", metadata.ariaLabel);
		}
	} else {
		tokenSpan.textContent = token;
		if (metadata.ariaLabel) {
			tokenSpan.setAttribute("aria-label", metadata.ariaLabel);
		}
	}
	if (metadata.isEnd) {
		tokenSpan.classList.add("token-end");
	}
	tokenSpan.dataset.position = position;
	if (messageIndex !== undefined) {
		tokenSpan.dataset.messageIndex = messageIndex;
	}
	tokenSpan.onclick = onClick;
	tokenSpan.style.backgroundColor = getProbabilityColor(prob);
	return tokenSpan;
}

export function createConfidenceIndicator(confidence, tokenCount = null) {
	const confidenceDiv = document.createElement("div");
	confidenceDiv.className = "confidence-indicator";
	const tokenCountText = tokenCount !== null ? ` • ${tokenCount} token${tokenCount !== 1 ? 's' : ''}` : '';
	confidenceDiv.innerHTML = `
        <div class="confidence-label">Confidence: 
            <span style="color: ${getConfidenceColor(confidence)}">
                ${getConfidenceLabel(confidence)} (${confidence}%)
            </span>${tokenCountText}
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
