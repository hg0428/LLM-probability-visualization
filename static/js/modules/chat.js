import { state, resetTokenMetrics } from "./state.js";
import { generateText } from "./socket.js";
import {
	createTokenSpan,
	createConfidenceIndicator,
	showModal,
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
	state.chatHistory.push({ role: "user", content: prompt });

	state.chatHistory.push({
		role: "assistant",
		content: "",
		partial: true,
		tokenSequence: [],
		id: state.chatHistory.length,
		chosenTokens: [],
	});

	updateChatDisplay();

	const settings = {
		chat_history: state.chatHistory,
		mode: "chat",
		...getGenerationSettings(),
	};

	await generateText(settings);
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
			currentMessage.content += chosen;
			currentMessage.chosenTokens.push(chosen);
			currentMessage.tokenSequence.push(options);
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

					token.split("\n").forEach((tokenText, i) => {
						if (i > 0) {
							contentWrapper.appendChild(document.createElement("br"));
						}
						const tokenSpan = createTokenSpan(
							tokenText,
							prob,
							position,
							messageIndex,
							() => showTokenOptions(message, position)
						);
						contentWrapper.appendChild(tokenSpan);
					});
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
				messageDiv.appendChild(createConfidenceIndicator(confidence));
			}
		} else {
			messageDiv.textContent = message.content;
		}

		chatDisplay.appendChild(messageDiv);
	});

	chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function showTokenOptions(message, position) {
	const alternatives = message.tokenSequence[position];
	if (!alternatives) return;

	const modalContent = document.getElementById("token-options");
	modalContent.innerHTML = "";

	alternatives
		.sort((a, b) => b[1] - a[1])
		.forEach(([token, prob, isChosen]) => {
			const optionDiv = createTokenOption(token, prob, isChosen);
			optionDiv.onclick = () => replaceToken(message, position, token);
			modalContent.appendChild(optionDiv);
		});

	showModal("token-modal");
}

function createTokenOption(token, prob, isChosen) {
	const optionDiv = document.createElement("div");
	optionDiv.className = `token-option ${isChosen ? "chosen" : ""}`;

	const tokenSpan = document.createElement("span");
	tokenSpan.className = "token-text";
	tokenSpan.textContent = token;

	const probSpan = document.createElement("span");
	probSpan.className = "token-prob";
	probSpan.textContent = (prob * 100).toFixed(2) + "%";

	optionDiv.style.borderLeft = `4px solid ${getProbabilityColor(prob)}`;
	optionDiv.appendChild(tokenSpan);
	optionDiv.appendChild(probSpan);

	return optionDiv;
}

async function replaceToken(message, position, token) {
	// Delete all messages after the current one
	state.chatHistory = state.chatHistory.slice(0, message.id + 1);
	message.partial = true;
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

	const settings = {
		chat_history: state.chatHistory,
		mode: "chat",
		...getGenerationSettings(),
	};
	document.getElementById("token-modal").style.display = "none";
	resetTokenMetrics();
	await generateText(settings);
}

function getGenerationSettings() {
	// Implementation moved to settings.js
	return window.getSettings();
}
