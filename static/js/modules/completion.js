import { state, resetTokenMetrics } from "./state.js";
import { generateText } from "./socket.js";
import { createTokenSpan, showModal, getProbabilityColor, getTokenVisualMetadata } from "./utils.js";

export function initializeCompletionHandlers() {
	const completionDisplay = document.getElementById("completion-display");
	if (completionDisplay) {
		completionDisplay.addEventListener("paste", handlePlainTextPaste);
	}

	document
		.getElementById("complete-button")
		.addEventListener("click", generateCompletion);
}

export function generateCompletion() {
	resetTokenMetrics();
	const completionDisplay = document.getElementById("completion-display");
	const settings = {
		prompt: completionDisplay.innerText,
		mode: "completion",
		...getGenerationSettings(),
	};

	generateText(settings);
}

export function updateCompletionDisplay(chosen, options) {
	const completionDisplay = document.getElementById("completion-display");

	// Generate unique token ID
	const tokenId = Math.floor(Math.random() * 1000000000);
	state.completionTokenOptions[tokenId] = options;

	const chosenToken = options.find(([_, __, isChosen]) => isChosen);
	if (!chosenToken) return;

	const [token, prob, _] = chosenToken;

	// Check if token contains newlines
	if (token.includes("\n")) {
		const parts = token.split("\n");
		parts.forEach((part, i) => {
			if (i > 0) {
				completionDisplay.appendChild(document.createElement("br"));
			}
			if (part.length > 0 || i === 0) {
				const tokenSpan = createTokenSpan(
					part,
					prob,
					undefined,
					undefined,
					() => showCompletionTokenOptions(tokenId, tokenSpan),
					token
				);
				tokenSpan.setAttribute("data-token-id", tokenId);
				completionDisplay.appendChild(tokenSpan);
			}
		});
	} else {
		const tokenSpan = createTokenSpan(
			token,
			prob,
			undefined,
			undefined,
			() => showCompletionTokenOptions(tokenId, tokenSpan),
			token
		);
		tokenSpan.setAttribute("data-token-id", tokenId);
		completionDisplay.appendChild(tokenSpan);
	}
}

function showCompletionTokenOptions(tokenId, tokenSpan) {
	const alternatives = state.completionTokenOptions[tokenId];
	if (!alternatives) return;

	const modalContent = document.getElementById("token-options");
	modalContent.innerHTML = "";

	alternatives
		.sort((a, b) => b[1] - a[1])
		.forEach(([token, prob, isChosen]) => {
			const optionDiv = createCompletionTokenOption(token, prob, isChosen);
			optionDiv.onclick = () => replaceCompletionToken(token, prob, tokenSpan);
			modalContent.appendChild(optionDiv);
		});

	showModal("token-modal");
}

function createCompletionTokenOption(token, prob, isChosen) {
	const optionDiv = document.createElement("div");
	optionDiv.className = `token-option ${isChosen ? "chosen" : ""}`;

	const tokenSpan = document.createElement("span");
	tokenSpan.className = "token-text";
	const metadata = getTokenVisualMetadata(token);
	if (metadata.isInvisible) {
		tokenSpan.classList.add("token-text-empty");
		tokenSpan.textContent = metadata.placeholder;
		if (metadata.ariaLabel) {
			tokenSpan.setAttribute("aria-label", metadata.ariaLabel);
		}
	} else {
		tokenSpan.textContent = token;
	}
	if (metadata.isEnd) {
		tokenSpan.classList.add("token-text-end");
	}

	const probSpan = document.createElement("span");
	probSpan.className = "token-prob";
	probSpan.textContent = (prob * 100).toFixed(2) + "%";

	optionDiv.style.borderLeft = `4px solid ${getProbabilityColor(prob)}`;
	optionDiv.appendChild(tokenSpan);
	optionDiv.appendChild(probSpan);

	return optionDiv;
}

async function replaceCompletionToken(token, prob, tokenSpan) {
	tokenSpan.textContent = token;
	tokenSpan.style.backgroundColor = getProbabilityColor(prob);

	// Remove all elements after the target element
	let nextElement = tokenSpan.nextElementSibling;
	while (nextElement) {
		const next = nextElement.nextElementSibling;
		nextElement.remove();
		nextElement = next;
	}

	await generateCompletion();
}

function getGenerationSettings() {
	// Implementation moved to settings.js
	return window.getSettings();
}

function handlePlainTextPaste(event) {
	event.preventDefault();
	const text = event.clipboardData?.getData("text/plain");
	if (!text) return;

	const selection = window.getSelection();
	if (!selection || selection.rangeCount === 0) {
		const target = event.currentTarget;
		if (target instanceof HTMLElement) {
			target.textContent += text;
		}
		return;
	}

	const range = selection.getRangeAt(0);
	range.deleteContents();
	const textNode = document.createTextNode(text);
	range.insertNode(textNode);

	range.setStartAfter(textNode);
	range.collapse(true);
	selection.removeAllRanges();
	selection.addRange(range);
}
