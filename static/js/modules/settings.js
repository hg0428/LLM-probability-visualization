import { setupSettingsToggle } from "./utils.js";

export function initializeSettings() {
	// Initialize settings panel
	setupSettingsPanel();

	// Initialize all setting group toggles
	setupSettingsToggle("randomness");
	setupSettingsToggle("truncation");
	setupSettingsToggle("xtc");
	setupSettingsToggle("penalties");
	setupSettingsToggle("dry");

	// Initialize highlight toggle
	initializeHighlightToggle();
}

function setupSettingsPanel() {
	const settingsContainer = document.querySelector(".settings-container");
	const resizeHandle = document.querySelector(".resize-handle");
	const toggleButton = document.getElementById("toggle-settings");

	const SETTINGS_MIN_WIDTH = 300;
	const SETTINGS_MAX_WIDTH = 600;
	const SETTINGS_SNAP_THRESHOLD = 200;

	let isResizing = false;
	let initialX;
	let currentWidth;
	let rafId = null;

	// Toggle settings panel
	function toggleSettings(forceState) {
		settingsContainer.classList.add("animating");

		if (typeof forceState === "boolean") {
			settingsContainer.classList.toggle("closed", !forceState);
		} else {
			settingsContainer.classList.toggle("closed");
		}

		if (settingsContainer.classList.contains("closed")) {
			settingsContainer.style.width = "0";
		} else {
			settingsContainer.style.width = `${SETTINGS_MIN_WIDTH}px`;
		}

		setTimeout(() => {
			settingsContainer.classList.remove("animating");
		}, 200);
	}

	toggleButton.addEventListener("click", () => toggleSettings());

	// Resize functionality
	function updateWidth() {
		if (!isResizing) return;
		settingsContainer.style.width = `${currentWidth}px`;
		rafId = requestAnimationFrame(updateWidth);
	}

	function handleMouseMove(e) {
		if (!isResizing) return;
		const delta = e.clientX - initialX;
		const newWidth = currentWidth - delta;
		currentWidth = Math.max(0, Math.min(SETTINGS_MAX_WIDTH, newWidth));
		initialX = e.clientX;
	}

	function handleMouseUp() {
		isResizing = false;
		document.removeEventListener("mousemove", handleMouseMove);
		document.removeEventListener("mouseup", handleMouseUp);

		if (rafId !== null) {
			cancelAnimationFrame(rafId);
			rafId = null;
		}

		settingsContainer.classList.add("animating");

		if (currentWidth < SETTINGS_SNAP_THRESHOLD) {
			toggleSettings(false);
		} else if (currentWidth < SETTINGS_MIN_WIDTH) {
			settingsContainer.style.width = `${SETTINGS_MIN_WIDTH}px`;
		}

		setTimeout(() => {
			settingsContainer.classList.remove("animating");
		}, 200);
	}

	resizeHandle.addEventListener("mousedown", (e) => {
		e.preventDefault();
		isResizing = true;
		const wasClosed = settingsContainer.classList.contains("closed");

		if (wasClosed) {
			settingsContainer.classList.remove("closed", "animating");
			settingsContainer.style.width = "0px";
			currentWidth = 0;
		} else {
			currentWidth = settingsContainer.offsetWidth;
		}

		initialX = e.clientX;
		rafId = requestAnimationFrame(updateWidth);

		document.addEventListener("mousemove", handleMouseMove);
		document.addEventListener("mouseup", handleMouseUp);
	});

	// Handle mobile toggle
	if (window.innerWidth <= 768) {
		toggleButton.addEventListener("click", () => {
			settingsContainer.classList.toggle("open");
		});
	}
}

function initializeHighlightToggle() {
	const highlightToggle = document.getElementById("highlight-toggle");
	const chatDisplay = document.getElementById("chat-display");
	const completionDisplay = document.getElementById("completion-display");

	highlightToggle.addEventListener("change", (e) => {
		if (e.target.checked) {
			chatDisplay.classList.remove("highlights-disabled");
			completionDisplay.classList.remove("highlights-disabled");
		} else {
			chatDisplay.classList.add("highlights-disabled");
			completionDisplay.classList.add("highlights-disabled");
		}
		localStorage.setItem("highlightsEnabled", e.target.checked);
	});

	// Load saved preference
	const highlightsEnabled = localStorage.getItem("highlightsEnabled");
	if (highlightsEnabled !== null) {
		highlightToggle.checked = highlightsEnabled === "true";
		if (!highlightToggle.checked) {
			chatDisplay.classList.add("highlights-disabled");
			completionDisplay.classList.add("highlights-disabled");
		}
	}
}

export function getSettings() {
	const modelSelect = document.getElementById("model-select");

	const settings = {
		model_name: modelSelect.value,
		randomness_enabled: document.getElementById("toggle-randomness").checked,
		truncation_enabled: document.getElementById("toggle-truncation").checked,
		penalties_enabled: document.getElementById("toggle-penalties").checked,
		dry_enabled: document.getElementById("toggle-dry").checked,
		xtc_enabled: document.getElementById("toggle-xtc").checked,
		num_show: parseInt(document.getElementById("num-show-value").value) || 12,
		max_new_tokens:
			parseInt(document.getElementById("max-new-tokens-value").value) || 0,
	};

	// Add conditional settings based on toggles
	if (settings.truncation_enabled) {
		Object.assign(settings, {
			min_p: parseFloat(document.getElementById("min-p-value").value) || 0,
			top_k: parseInt(document.getElementById("top-k-value").value) || 0,
			top_p: parseFloat(document.getElementById("top-p-value").value) || 1.0,
		});
	}

	if (settings.randomness_enabled) {
		settings.temperature =
			parseFloat(document.getElementById("temperature-value").value) || 0.7;
	}

	if (settings.xtc_enabled) {
		Object.assign(settings, {
			xtc_threshold:
				parseFloat(document.getElementById("xtc-threshold").value) || 0.2,
			xtc_probability:
				parseFloat(document.getElementById("xtc-probability").value) || 0.5,
		});
	}

	if (settings.penalties_enabled) {
		Object.assign(settings, {
			repetition_penalty:
				parseFloat(document.getElementById("repetition-penalty-value").value) ||
				1.0,
			frequency_penalty:
				parseFloat(document.getElementById("frequency-penalty-value").value) ||
				0.0,
			presence_penalty:
				parseFloat(document.getElementById("presence-penalty-value").value) ||
				0.0,
			repeat_last_n:
				parseInt(document.getElementById("repeat-last-n-value").value) || 64,
		});
	}

	if (settings.dry_enabled) {
		Object.assign(settings, {
			dry_allowed_length:
				parseInt(document.getElementById("dry-allowed-length").value) || 1,
			dry_base: parseFloat(document.getElementById("dry-base").value) || 2,
			dry_multiplier:
				parseFloat(document.getElementById("dry-multiplier").value) || 3,
			dry_range: parseInt(document.getElementById("dry-range").value) || 1024,
		});
	}

	return settings;
}
