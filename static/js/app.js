import { state, updateState } from "./modules/state.js";
import { initializeWebSocket, stopGeneration } from "./modules/socket.js";
import { initializeChatHandlers } from "./modules/chat.js";
import { initializeCompletionHandlers } from "./modules/completion.js";
import { initializeSettings, getSettings } from "./modules/settings.js";
import { closeModal, getProbabilityColor } from "./modules/utils.js";

// Make getSettings available globally for modules
window.getSettings = getSettings;

// Make getProbabilityColor available globally for modules
window.getProbabilityColor = getProbabilityColor;

document.addEventListener("DOMContentLoaded", () => {
	// Initialize WebSocket
	initializeWebSocket();

	// Initialize UI handlers
	initializeUIHandlers();

	// Initialize mode-specific handlers
	initializeChatHandlers();
	initializeCompletionHandlers();

	// Initialize settings
	initializeSettings();
});

function initializeUIHandlers() {
	// Mode switching
	const mainChatContent = document.querySelector(".main-chat-content");
	const mainCompletionContent = document.querySelector(
		".main-completion-content"
	);
	const chatSidebarButton = document.getElementById("chat-sidebar-button");
	const completionSidebarButton = document.getElementById(
		"completion-sidebar-button"
	);

	completionSidebarButton.addEventListener("click", () => {
		updateState("mode", "completion");
		mainChatContent.style.display = "none";
		mainCompletionContent.style.display = "flex";
		chatSidebarButton.classList.remove("active");
		completionSidebarButton.classList.add("active");
	});

	chatSidebarButton.addEventListener("click", () => {
		updateState("mode", "chat");
		mainChatContent.style.display = "flex";
		mainCompletionContent.style.display = "none";
		chatSidebarButton.classList.add("active");
		completionSidebarButton.classList.remove("active");
	});

	// Stop generation buttons
	document
		.getElementById("stop-button")
		.addEventListener("click", stopGeneration);
	document
		.getElementById("stop-button-completion")
		.addEventListener("click", stopGeneration);

	// Modal handling
	window.addEventListener("click", (e) => {
		const modal = document.getElementById("token-modal");
		if (e.target === modal) {
			closeModal("token-modal");
		}
	});

	// Make closeTokenModal available globally for the HTML onclick handler
	window.closeTokenModal = () => closeModal("token-modal");
}
