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
	setupSettingsToggle("custom-template");

	// Initialize highlight toggle
	initializeHighlightToggle();

	// Initialize model logit summing
	initializeLogitSumming();
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
	const logitSumEnabled = document.getElementById("toggle-logit-sum").checked;
	let modelName;

	if (logitSumEnabled) {
		// Get models and weights for logit summing
		const modelWeights = getModelWeights();
		if (modelWeights.length > 0) {
			// Create a JSON configuration for the multi-model setup
			modelName = JSON.stringify({
				models: modelWeights,
			});
		} else {
			// Fall back to single model if no models are selected for summing
			modelName = document.getElementById("model-select").value;
		}
	} else {
		// Use the single model selection
		modelName = document.getElementById("model-select").value;
	}

	const settings = {
		model_name: modelName,
		randomness_enabled: document.getElementById("toggle-randomness").checked,
		truncation_enabled: document.getElementById("toggle-truncation").checked,
		penalties_enabled: document.getElementById("toggle-penalties").checked,
		dry_enabled: document.getElementById("toggle-dry").checked,
		xtc_enabled: document.getElementById("toggle-xtc").checked,
		num_show: parseInt(document.getElementById("num-show-value").value) || 12,
		max_new_tokens:
			parseInt(document.getElementById("max-new-tokens-value").value) || 0,
		custom_template_enabled: document.getElementById("toggle-custom-template")
			.value,
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

	if (settings.custom_template_enabled) {
		settings.user_role_name = document.getElementById("user-role-name").value;
		settings.system_role_name =
			document.getElementById("system-role-name").value;
		settings.system_message = document.getElementById("system-message").value;
		settings.assistant_role_name = document.getElementById(
			"assistant-role-name"
		).value;
	}

	return settings;
}

function getModelWeights() {
	// Get all model weight entries
	const modelWeightEntries = document.querySelectorAll(".model-weight-entry");
	const modelWeights = [];

	// Extract model names and weights
	modelWeightEntries.forEach((entry) => {
		const modelSelect = entry.querySelector("select");
		const weightInput = entry.querySelector('input[type="number"]');

		if (modelSelect && weightInput) {
			modelWeights.push({
				name: modelSelect.value,
				weight: parseFloat(weightInput.value) || 1.0,
			});
		}
	});

	return modelWeights;
}

function initializeLogitSumming() {
	const logitSumToggle = document.getElementById("toggle-logit-sum");
	const logitSumGroup = document.getElementById("logit-sum-group");
	const modelSelect = document.getElementById("model-select");
	const modelFamilySelect = document.getElementById("model-family-select");
	const addModelButton = document.getElementById("add-model-button");
	const modelWeightsContainer = document.getElementById(
		"model-weights-container"
	);

	// Initialize the model-weights-header if it doesn't exist
	if (!document.querySelector(".model-weights-header")) {
		const header = document.createElement("div");
		header.className = "model-weights-header";

		const modelHeader = document.createElement("span");
		modelHeader.className = "model-header";
		modelHeader.textContent = "Model";

		const weightHeader = document.createElement("span");
		weightHeader.className = "weight-header";
		weightHeader.textContent = "Weight";

		const actionHeader = document.createElement("span");
		actionHeader.className = "action-header";

		header.appendChild(modelHeader);
		header.appendChild(weightHeader);
		header.appendChild(actionHeader);

		modelWeightsContainer.appendChild(header);
	}

	// Toggle logit summing group visibility
	logitSumToggle.addEventListener("change", function () {
		logitSumGroup.style.display = this.checked ? "block" : "none";
		modelSelect.disabled = this.checked;

		// Add at least one model if none exists or only the header exists
		if (
			this.checked &&
			modelWeightsContainer.querySelectorAll(".model-weight-entry").length === 0
		) {
			addModelWeightEntry();
		}

		// Show a notification about what happened to the main model selector
		if (this.checked) {
			showNotification(
				"Single model selection disabled while logit summing is active"
			);
		} else {
			showNotification("Returned to single model selection mode");
		}
	});

	// Handle model family selection change
	modelFamilySelect.addEventListener("change", function () {
		// Clear existing model entries but keep the header
		const entries = modelWeightsContainer.querySelectorAll(
			".model-weight-entry"
		);
		entries.forEach((entry) => entry.remove());

		// Add a new entry with the selected family
		addModelWeightEntry();

		// Show notification about family change
		showNotification(`Selected model family: ${this.value}`);
	});

	// Add model button
	addModelButton.addEventListener("click", function () {
		addModelWeightEntry();
	});

	// Function to add a new model weight entry
	function addModelWeightEntry() {
		const family = modelFamilySelect.value;

		// Create a new entry div
		const entry = document.createElement("div");
		entry.className = "model-weight-entry setting";

		// Create model selector
		const modelSelector = document.createElement("select");
		modelSelector.setAttribute("aria-label", "Select model");

		// Get models for this family from the server-provided data
		const familyModels = getModelsForFamily(family);

		// Add options for each model in the family
		familyModels.forEach((model) => {
			const option = document.createElement("option");
			option.value = model;
			option.textContent = model;
			modelSelector.appendChild(option);
		});

		// Create weight input
		const weightInput = document.createElement("input");
		weightInput.type = "number";
		weightInput.min = "0.1";
		weightInput.max = "10";
		weightInput.step = "0.1";
		weightInput.value = "1.0";
		weightInput.title =
			"Weight for this model (higher values give more importance)";
		weightInput.setAttribute("aria-label", "Model weight");

		// Create weight label that appears on hover
		const weightLabel = document.createElement("span");
		weightLabel.className = "weight-label";
		weightLabel.textContent = "Weight:";
		weightLabel.style.display = "none";

		// Show label on hover
		weightInput.addEventListener("mouseenter", () => {
			weightLabel.style.display = "inline";
		});

		weightInput.addEventListener("mouseleave", () => {
			weightLabel.style.display = "none";
		});

		// Create remove button
		const removeButton = document.createElement("button");
		removeButton.className = "small-button remove-model-button";
		removeButton.innerHTML = '<i class="fas fa-times"></i>';
		removeButton.title = "Remove this model";
		removeButton.setAttribute("aria-label", "Remove model");

		// Add event listener to remove button
		removeButton.addEventListener("click", function () {
			entry.remove();

			// If no models left, add one back
			if (
				modelWeightsContainer.querySelectorAll(".model-weight-entry").length ===
				0
			) {
				addModelWeightEntry();
			}

			showNotification("Model removed from combination");
		});

		// Add elements to entry
		entry.appendChild(modelSelector);
		entry.appendChild(weightInput);
		entry.appendChild(removeButton);

		// Add entry to container after the header
		modelWeightsContainer.appendChild(entry);

		// Show notification
		const modelCount = modelWeightsContainer.querySelectorAll(
			".model-weight-entry"
		).length;
		if (modelCount > 1) {
			showNotification(`Added model #${modelCount} to combination`);
		} else {
			showNotification("Added first model to combination");
		}
	}

	// Helper function to get models for a family
	function getModelsForFamily(family) {
		// This data should be provided by the server in the template
		const modelFamiliesElement = document.getElementById("model-families-data");
		if (modelFamiliesElement) {
			const modelFamilies = JSON.parse(modelFamiliesElement.textContent);
			return modelFamilies[family] || [];
		}
		return [];
	}

	// Helper function to show a notification
	function showNotification(message) {
		// Check if notification container exists, if not create it
		let notificationContainer = document.getElementById(
			"notification-container"
		);
		if (!notificationContainer) {
			notificationContainer = document.createElement("div");
			notificationContainer.id = "notification-container";
			document.body.appendChild(notificationContainer);
		}

		// Create notification element
		const notification = document.createElement("div");
		notification.className = "notification";
		notification.textContent = message;
		notificationContainer.appendChild(notification);

		// Fade in
		setTimeout(() => {
			notification.classList.add("show");
		}, 10);

		// Fade out and remove after 3 seconds
		setTimeout(() => {
			notification.classList.remove("show");
			setTimeout(() => {
				notification.remove();
			}, 300);
		}, 3000);
	}
}
