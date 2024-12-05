// Global variables
let chatHistory = [];
let lastTokenSequence = null;
let lastResponseTokens = [];
let lastResponseProbs = [];
let selectedPosition = null;
let modelSelect, chatDisplay;
let isResizing = false;
let initialWidth;
let initialX;
let socket = null;
let currentMessageDiv = null;
let completionDisplay = null;
let mode = "chat";
let mainChatContent, mainCompletionContent;
let chatSidebarButton, completionSidebarButton;
let completeButton;
let completionTokenOptions = {};

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
	initializeWebSocket();
	// Send button click
	document
		.getElementById("send-button")
		.addEventListener("click", generateChat);
	modelSelect = document.getElementById("model-select");
	chatDisplay = document.getElementById("chat-display");
	completionDisplay = document.getElementById("completion-display");
	mainChatContent = document.querySelector(".main-chat-content");
	mainCompletionContent = document.querySelector(".main-completion-content");
	chatSidebarButton = document.getElementById("chat-sidebar-button");
	completionSidebarButton = document.getElementById(
		"completion-sidebar-button"
	);
	completeButton = document.getElementById("complete-button");
	completeButton.addEventListener("click", generateCompletion);

	// Enter key in textarea
	document.getElementById("prompt").addEventListener("keypress", (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			generateChat();
		}
	});

	// Close modal when clicking outside
	window.addEventListener("click", (e) => {
		const modal = document.getElementById("token-modal");
		if (e.target === modal) {
			closeTokenModal();
		}
	});

	// Settings panel functionality
	const settingsContainer = document.querySelector(".settings-container");
	const toggleButton = document.getElementById("toggle-settings");
	const dragHandle = document.querySelector(".drag-handle");

	// Toggle settings panel
	toggleButton.addEventListener("click", () => {
		settingsContainer.classList.toggle("closed");
		if (settingsContainer.classList.contains("closed")) {
			settingsContainer.style.width = "0";
		} else {
			settingsContainer.style.width = "300px";
		}
	});

	// Resize functionality
	dragHandle.addEventListener("mousedown", (e) => {
		isResizing = true;
		initialWidth = settingsContainer.offsetWidth;
		initialX = e.clientX;

		document.addEventListener("mousemove", handleMouseMove);
		document.addEventListener("mouseup", () => {
			isResizing = false;
			document.removeEventListener("mousemove", handleMouseMove);
		});
	});

	// Handle mobile toggle
	if (window.innerWidth <= 768) {
		toggleButton.addEventListener("click", () => {
			settingsContainer.classList.toggle("open");
		});
	}

	// Settings group toggles
	setupSettingsToggle("randomness");
	setupSettingsToggle("truncation");
	setupSettingsToggle("xtc");
	setupSettingsToggle("penalties");
	setupSettingsToggle("dry");

	completionSidebarButton.addEventListener("click", (e) => {
		mode = "completion";
		mainChatContent.style.display = "none";
		mainCompletionContent.style.display = "flex";
		chatSidebarButton.classList.remove("active");
		completionSidebarButton.classList.add("active");
	});

	chatSidebarButton.addEventListener("click", (e) => {
		mode = "chat";
		mainChatContent.style.display = "flex";
		mainCompletionContent.style.display = "none";
		chatSidebarButton.classList.add("active");
		completionSidebarButton.classList.remove("active");
	});
});

function setupSettingsToggle(groupName) {
	const toggle = document.getElementById(`toggle-${groupName}`);
	const group = document.getElementById(`${groupName}-group`);

	toggle.addEventListener("change", () => {
		group.classList.toggle("disabled", !toggle.checked);
		if (!toggle.checked) {
			// Reset values when disabled
			const inputs = group.querySelectorAll('input[type="number"]');
			inputs.forEach((input) => {
				switch (input.id) {
					case "temperature-value":
						input.value = "1.0";
						break;
					case "top-p-value":
						input.value = "1.0";
						break;
					case "min-p-value":
					case "top-k-value":
						input.value = "0";
						break;
					case "repetition-penalty-value":
						input.value = "1.0";
						break;
					case "frequency-penalty-value":
						input.value = "0.0";
						break;
					case "presence-penalty-value":
						input.value = "0.0";
						break;
					case "dry-allowed-length":
						input.value = "1";
						break;
					case "dry-base":
						input.value = "2";
						break;
					case "dry-multiplier":
						input.value = "3";
						break;
					case "dry-range":
						input.value = "1024";
						break;
				}
			});
		}
	});
}

function getSettings() {
	const selectedModel = modelSelect.value;
	const randomnessEnabled =
		document.getElementById("toggle-randomness").checked;
	const truncationEnabled =
		document.getElementById("toggle-truncation").checked;
	const penaltiesEnabled = document.getElementById("toggle-penalties").checked;
	const dryEnabled = document.getElementById("toggle-dry").checked;
	const xtcEnabled = document.getElementById("toggle-xtc").checked;
	// Default values when disabled
	let temperature = 1.0;
	let min_p = 0;
	let top_k = 0;
	let top_p = 1.0;
	let repetition_penalty = 1.0;
	let frequency_penalty = 0.0;
	let presence_penalty = 0.0;
	let repeat_last_n = 64;
	let dry_allowed_length = 2;
	let dry_base = 1.2;
	let dry_multiplier = 2;
	let dry_range = 512;
	let xtc_threshold = 1;
	let xtc_probability = 0;

	// Only get values if enabled
	if (truncationEnabled) {
		min_p = parseFloat(document.getElementById("min-p-value").value) || 0;
		top_k = parseInt(document.getElementById("top-k-value").value) || 0;
		top_p = parseFloat(document.getElementById("top-p-value").value) || 1.0;
	}
	if (randomnessEnabled) {
		temperature =
			parseFloat(document.getElementById("temperature-value").value) || 0.7;
	}
	if (xtcEnabled) {
		xtc_threshold =
			parseFloat(document.getElementById("xtc-threshold").value) || 0.2;
		xtc_probability =
			parseFloat(document.getElementById("xtc-probability").value) || 0.5;
	}

	if (penaltiesEnabled) {
		repetition_penalty =
			parseFloat(document.getElementById("repetition-penalty-value").value) ||
			1.0;
		frequency_penalty =
			parseFloat(document.getElementById("frequency-penalty-value").value) ||
			0.0;
		presence_penalty =
			parseFloat(document.getElementById("presence-penalty-value").value) ||
			0.0;
		repeat_last_n =
			parseInt(document.getElementById("repeat-last-n-value").value) || 64;
	}

	if (dryEnabled) {
		dry_allowed_length =
			parseInt(document.getElementById("dry-allowed-length").value) || 1;
		dry_base = parseFloat(document.getElementById("dry-base").value) || 2;
		dry_multiplier =
			parseFloat(document.getElementById("dry-multiplier").value) || 3;
		dry_range = parseInt(document.getElementById("dry-range").value) || 1024;
	}

	const num_show =
		parseInt(document.getElementById("num-show-value").value) || 12;
	const max_new_tokens =
		parseInt(document.getElementById("max-new-tokens-value").value) || 0;

	return {
		model_name: selectedModel,
		randomness_enabled: randomnessEnabled,
		penalties_enabled: penaltiesEnabled,
		xtc_enabled: xtcEnabled,
		truncation_enabled: truncationEnabled,
		dry_enabled: dryEnabled,
		num_show,
		temperature,
		min_p,
		top_k,
		max_new_tokens,
		top_p,
		repetition_penalty,
		frequency_penalty,
		presence_penalty,
		repeat_last_n,
		dry_allowed_length,
		dry_base,
		dry_multiplier,
		dry_range,
		xtc_threshold,
		xtc_probability,
	};
}

async function generateResponse(
	chat_history = null,
	prompt = null,
	mode = "chat"
) {
	socket.emit("stop");
	socket.emit("generate", {
		chat_history: chat_history,
		prompt: prompt,
		mode,
		...getSettings(),
	});
}

// Text Generation
async function generateChat() {
	const prompt = document.getElementById("prompt").value;
	document.getElementById("prompt").value = "";
	if (!prompt.trim()) return;

	// Add user message to chat history
	chatHistory.push({ role: "user", content: prompt });

	messageDiv = document.createElement("div");
	chatHistory.push({
		role: "assistant",
		content: "",
		partial: true,
		tokenSequence: [],
		id: chatHistory.length,
		chosenTokens: [],
	});
	updateChatDisplay();

	await generateResponse(chatHistory, null, "chat");
}
async function generateCompletion() {
	await generateResponse(null, completionDisplay.textContent, "completion");
}

// Chat Display
function updateChatDisplay() {
	chatDisplay.innerHTML = "";

	chatHistory.forEach((message, messageIndex) => {
		const messageDiv = document.createElement("div");
		messageDiv.className = `message ${message.role}`;

		if (message.role === "assistant" && Array.isArray(message.tokenSequence)) {
			message.tokenSequence.forEach((alternatives, position) => {
				if (!Array.isArray(alternatives) || alternatives.length === 0) return;

				const chosenToken = alternatives.find(([_, __, isChosen]) => isChosen);
				if (!chosenToken) return;

				const [token, prob, _] = chosenToken;
				const probColor = getProbabilityColor(prob);
				let i = 0;
				token.split("\n").map((token) => {
					if (i > 0) {
						const br = document.createElement("br");
						messageDiv.appendChild(br);
					}
					const tokenSpan = document.createElement("span");
					tokenSpan.textContent = token;
					tokenSpan.className = "token";
					tokenSpan.dataset.position = position;
					tokenSpan.dataset.messageIndex = messageIndex;
					// Add click handler
					tokenSpan.onclick = () => showTokenOptions(message, position);
					// Add probability-based highlighting
					tokenSpan.style.backgroundColor = probColor;
					messageDiv.appendChild(tokenSpan);
					i++;
				});
			});
		} else {
			messageDiv.textContent = message.content;
		}

		chatDisplay.appendChild(messageDiv);
	});

	chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

// Helper function to get color based on probability
function getProbabilityColor(prob) {
	const red = 255 - Math.floor(prob * 255);
	const green = Math.floor(prob * 255);
	return `rgba(${red}, ${green}, 0, 0.5)`;
}

// Token Options Modal
function showTokenOptions(message, position) {
	const alternatives = message.tokenSequence[position];
	if (!alternatives) return;

	const modal = document.getElementById("token-modal");
	const modalContent = document.getElementById("token-options");
	modalContent.innerHTML = "";

	// Sort alternatives by probability
	alternatives
		.sort((a, b) => b[1] - a[1]) // Sort by probability (descending)
		.forEach(([token, prob, isChosen]) => {
			const optionDiv = document.createElement("div");
			optionDiv.className = `token-option ${isChosen ? "chosen" : ""}`;

			const tokenSpan = document.createElement("span");
			tokenSpan.className = "token-text";
			tokenSpan.textContent = token;

			const probSpan = document.createElement("span");
			probSpan.className = "token-prob";
			probSpan.textContent = (prob * 100).toFixed(2) + "%";

			// Add probability-based highlighting
			const probColor = getProbabilityColor(prob);
			optionDiv.style.borderLeft = `4px solid ${probColor}`;

			optionDiv.appendChild(tokenSpan);
			optionDiv.appendChild(probSpan);

			// Add click handler for token replacement
			optionDiv.onclick = () => {
				replaceToken(message, position, token);
				closeTokenModal();
			};

			modalContent.appendChild(optionDiv);
		});

	modal.style.display = "block";
}
function showTokenOptionsCompletion(token_id, tokenSpan) {
	const alternatives = completionTokenOptions[token_id];
	if (!alternatives) return;

	const modal = document.getElementById("token-modal");
	const modalContent = document.getElementById("token-options");
	modalContent.innerHTML = "";

	// Sort alternatives by probability
	alternatives
		.sort((a, b) => b[1] - a[1]) // Sort by probability (descending)
		.forEach(([token, prob, isChosen]) => {
			const optionDiv = document.createElement("div");
			optionDiv.className = `token-option ${isChosen ? "chosen" : ""}`;

			const tokenOptionSpan = document.createElement("span");
			tokenOptionSpan.className = "token-text";
			tokenOptionSpan.textContent = token;

			const probSpan = document.createElement("span");
			probSpan.className = "token-prob";
			probSpan.textContent = (prob * 100).toFixed(2) + "%";

			// Add probability-based highlighting
			const probColor = getProbabilityColor(prob);
			optionDiv.style.borderLeft = `4px solid ${probColor}`;

			optionDiv.appendChild(tokenOptionSpan);
			optionDiv.appendChild(probSpan);

			// Add click handler for token replacement
			optionDiv.onclick = () => {
				console.log("Option clicked:", token, tokenSpan);
				tokenSpan.textContent = token;
				// Remove all elements after the target element
				let nextElement = tokenSpan.nextElementSibling;
				while (nextElement) {
					const next = nextElement.nextElementSibling;
					nextElement.remove();
					nextElement = next;
				}
				generateCompletion();
				closeTokenModal();
			};

			modalContent.appendChild(optionDiv);
		});

	modal.style.display = "block";
}

function closeTokenModal() {
	const modal = document.getElementById("token-modal");
	modal.style.display = "none";
}

async function replaceToken(message, position, token) {
	// TODO: make it stash past responses.

	// Delete all messages after the current one
	chatHistory = chatHistory.slice(0, message.id + 1);
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
	await generateResponse(chatHistory);
}

function handleMouseMove(e) {
	if (!isResizing) return;

	const settingsContainer = document.querySelector(".settings-container");
	const delta = initialX - e.clientX;
	const newWidth = Math.max(300, Math.min(600, initialWidth + delta));
	settingsContainer.style.width = `${newWidth}px`;
}

function initializeWebSocket() {
	socket = io();

	socket.on("connect", () => {
		console.log("Connected to WebSocket server");
	});

	socket.on("disconnect", () => {
		console.log("Disconnected from WebSocket server");
	});

	socket.on("error", (data) => {
		console.error("Server error:", data.message);
		alert(`Error generating response: ${data.message}`);
	});

	socket.on("token", (data) => {
		console.log("Received token:", data);
		if (data.mode === "chat" && !currentMessageDiv) {
			currentMessageDiv = document.createElement("div");
			currentMessageDiv.className = "message assistant";
			chatDisplay.appendChild(currentMessageDiv);
		}

		const { chosen, options, message_id, token_id } = data;
		if (options && options.length > 0) {
			if (data.mode === "completion") {
				// Generate long random unique id.
				let token_id = Math.floor(Math.random() * 1000000000);
				completionTokenOptions[token_id] = options;
				const chosenToken = options.find(([_, __, isChosen]) => isChosen);
				const [token, prob, _] = chosenToken;
				const probColor = getProbabilityColor(prob);
				let i = 0;
				token.split("\n").map((token) => {
					if (i > 0) {
						const br = document.createElement("br");
						completionDisplay.appendChild(br);
					}
					const tokenSpan = document.createElement("span");
					tokenSpan.textContent = chosen;
					tokenSpan.className = "token";
					tokenSpan.setAttribute("data-token-id", token_id);
					// Add click handler
					tokenSpan.onclick = () =>
						showTokenOptionsCompletion(token_id, tokenSpan);
					// Add probability-based highlighting
					tokenSpan.style.backgroundColor = probColor;
					completionDisplay.appendChild(tokenSpan);
					i++;
				});
			} else if (chatHistory[chatHistory.length - 1].partial) {
				chatHistory[chatHistory.length - 1].content += chosen;
				chatHistory[chatHistory.length - 1].chosenTokens.push(chosen);
				chatHistory[chatHistory.length - 1].tokenSequence.push(options);
			} else {
				chatHistory.push({
					role: "assistant",
					content: chosen,
					partial: true,
					tokenSequence: [options],
					id: chatHistory.length,
					chosenTokens: [chosen],
				});
			}
			updateChatDisplay();
		}
	});
	socket.on("end", () => {
		if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].partial) {
			chatHistory[chatHistory.length - 1].partial = false;
		}
		console.log("end");
		updateChatDisplay();
	});
}
