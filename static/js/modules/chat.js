import { state, resetTokenMetrics } from "./state.js";
import { generateText } from "./socket.js";
import {
	createTokenSpan,
	createConfidenceIndicator,
	showModal,
	getTokenVisualMetadata,
} from "./utils.js";
import { getSettings } from "./settings.js";

export function initializeChatHandlers() {
	document
		.getElementById("send-button")
		.addEventListener("click", generateChat);
	document.getElementById("prompt").addEventListener("keypress", (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			generateChat();
		}
	});
}

export async function generateChat() {
	const prompt = document.getElementById("prompt").value;
	document.getElementById("prompt").value = "";
	if (!prompt.trim()) return;

	resetTokenMetrics();

	// Add user message to chat history
	state.chatHistory.push({
		role: "user",
		content: prompt,
		time: new Date(),
	});

	state.chatHistory.push({
		role: "assistant",
		content: "",
		partial: true,
		tokenSequence: [],
		id: state.chatHistory.length,
		chosenTokens: [],
		time: new Date(),
	});

	updateChatDisplay();

	const settings = {
		chat_history: state.chatHistory,
		mode: "chat",
		...getGenerationSettings(),
	};

	await generateText(settings);
}

function deleteMessage(messageDiv, messageIndex) {
	messageDiv.remove();
	state.chatHistory.splice(messageIndex, 1);
	updateChatDisplay();
}
export function updateChatDisplay(chosen, options, messageId) {
	const chatDisplay = document.getElementById("chat-display");
	chatDisplay.innerHTML = "";

	let hasSystem = getSettings().system_message;
	// Update current message if new token received
	if (chosen && options && messageId !== undefined) {
		console.log("chosen tokens", chosen);
		const currentMessage = state.chatHistory[messageId - (hasSystem ? 1 : 0)];
		console.log(currentMessage);
		if (currentMessage && currentMessage.partial) {
			// Do not include end tokens in the underlying content/chosenTokens to avoid duplication in prompts
			const meta = getTokenVisualMetadata(chosen);
			if (!meta.isEnd) {
				currentMessage.content += chosen;
				currentMessage.chosenTokens.push(chosen);
			}
			currentMessage.tokenSequence.push(options);
			currentMessage.time = new Date();
		}
	}

	state.chatHistory.forEach((message, messageIndex) => {
		const messageDiv = document.createElement("div");
		messageDiv.className = `message ${message.role}`;

		if (message.role === "assistant") {
			const contentWrapper = document.createElement("div");
			contentWrapper.className = "message-content";

			if (message.tokenSequence) {
				message.tokenSequence.forEach((alternatives, position) => {
					if (!Array.isArray(alternatives) || alternatives.length === 0) return;

					const chosenToken = alternatives.find(
						([_, __, isChosen]) => isChosen
					);
					if (!chosenToken) return;

					const [token, prob, _] = chosenToken;

					// Check if token contains newlines
					if (token.includes("\n")) {
						const parts = token.split("\n");
						parts.forEach((part, i) => {
							if (i > 0) {
								contentWrapper.appendChild(document.createElement("br"));
							}
							if (part.length > 0 || i === 0) {
								const tokenSpan = createTokenSpan(
									part,
									prob,
									position,
									messageIndex,
									() => showTokenOptions(message, position, messageIndex),
									token
								);
								contentWrapper.appendChild(tokenSpan);
							}
						});
					} else {
						const tokenSpan = createTokenSpan(
							token,
							prob,
							position,
							messageIndex,
							() => showTokenOptions(message, position, messageIndex),
							token
						);
						contentWrapper.appendChild(tokenSpan);
					}
				});

				messageDiv.appendChild(contentWrapper);

				// Add confidence indicator
				const probabilities = message.tokenSequence.map((alternatives) => {
					const chosenToken = alternatives.find(
						([_, __, isChosen]) => isChosen
					);
					return chosenToken ? chosenToken[1] : 0;
				});

				const avgProb =
					probabilities.reduce((a, b) => a + b, 0) / probabilities.length;
				const confidence = Math.round(avgProb * 100);
				const tokenCount = message.tokenSequence.length;
				messageDiv.appendChild(createConfidenceIndicator(confidence, tokenCount));
			}
		} else {
			messageDiv.textContent = message.content;
		}
		const deleteMessageButton = document.createElement("button");
		deleteMessageButton.className = "delete-message-button";
		deleteMessageButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-trash" viewBox="0 0 16 16">
  <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0z"/>
  <path d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4zM2.5 3h11V2h-11z"/>
</svg>`;
		deleteMessageButton.onclick = () => deleteMessage(messageDiv, messageIndex);
		messageDiv.appendChild(deleteMessageButton);

		chatDisplay.appendChild(messageDiv);
	});

	chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function showTokenOptions(message, position, messageIndex) {
    const alternatives = message.tokenSequence[position];
    if (!alternatives) return;

	const modalContent = document.getElementById("token-options");
	modalContent.innerHTML = "";

	alternatives
		.sort((a, b) => b[1] - a[1])
		.forEach(([token, prob, isChosen]) => {
            const optionDiv = createTokenOption(token, prob, isChosen);
            optionDiv.onclick = () => replaceToken(messageIndex, message, position, token);
            modalContent.appendChild(optionDiv);
        });

	showModal("token-modal");
}

function createTokenOption(token, prob, isChosen) {
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

async function replaceToken(messageIndex, message, position, token) {
	// Delete all messages after the current one
	state.chatHistory = state.chatHistory.slice(0, messageIndex + 1);
	
	// Check if the selected token is an end token
	const metadata = getTokenVisualMetadata(token);
	const isEndToken = metadata.isEnd;
	
	// If it's an end token, mark message as complete; otherwise keep it partial
	message.partial = !isEndToken;
	message.chosenTokens = message.chosenTokens
		.slice(0, position)
		.concat([
			message.tokenSequence[position].find(([t, _, __]) => t === token)[0],
		]);
	message.content = message.chosenTokens.join("");
	message.tokenSequence = message.tokenSequence
		.slice(0, position)
		.concat([
			message.tokenSequence[position].map(([t, p, c]) => [t, p, t === token]),
		]);

	// Only continue generation if the token is not an end token
	if (!isEndToken) {
		const settings = {
			chat_history: state.chatHistory,
			mode: "chat",
			...getGenerationSettings(),
		};
		document.getElementById("token-modal").style.display = "none";
		resetTokenMetrics();
		await generateText(settings);
	} else {
		// Just close the modal and update the display
		document.getElementById("token-modal").style.display = "none";
		updateChatDisplay();
	}
}

function getGenerationSettings() {
	// Implementation moved to settings.js
	return window.getSettings();
}
