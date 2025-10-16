import { state, incrementTokenCount, calculateTPS } from "./state.js";
import { updateChatDisplay } from "./chat.js";
import { updateCompletionDisplay } from "./completion.js";
import { getProbabilityColor } from "./utils.js";

let socket = null;

export function initializeWebSocket() {
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

	socket.on("token", handleToken);
	socket.on("end", handleEnd);

	return socket;
}

function handleToken(data) {
	incrementTokenCount();
	updateTpsDisplay();

	const { chosen, options, message_id, token_id, mode } = data;

	if (!options || options.length === 0) return;
	console.log("token");
	if (mode === "completion") {
		updateCompletionDisplay(chosen, options);
	} else {
		updateChatDisplay(chosen, options, message_id);
	}
}

function handleEnd() {
	console.log("Generation completed");
	state.currentMessageDiv = null;
	state.isGenerating = false;

	// Mark the last assistant message as complete (non-partial)
	const history = state.chatHistory;
	for (let i = history.length - 1; i >= 0; i--) {
		if (history[i] && history[i].role === "assistant") {
			history[i].partial = false;
			break;
		}
	}

	// Re-render to ensure UI reflects final state
	updateChatDisplay();

	// Final TPS update
	updateTpsDisplay();
}

function updateTpsDisplay() {
	const tps = calculateTPS();
	const tpsElement = document.getElementById(
		state.mode === "chat" ? "tps-indicator" : "tps-indicator-completion"
	);
	if (tpsElement) {
		tpsElement.textContent = `${tps} tokens/sec`;
	}
}

export function generateText(settings) {
	state.isGenerating = true;
	socket.emit("generate", settings);
}

export function stopGeneration() {
	socket.emit("stop");
	state.isGenerating = false;
}
