// Application state management
export const state = {
	chatHistory: [],
	lastTokenSequence: null,
	lastResponseTokens: [],
	lastResponseProbs: [],
	selectedPosition: null,
	currentMessageDiv: null,
	completionTokenOptions: {},
	tokenStartTime: null,
	tokenCount: 0,
	currentTPS: 0,
	mode: "chat",
	isGenerating: false,
};

// State update functions
export function updateState(key, value) {
	state[key] = value;
}

export function resetTokenMetrics() {
	state.tokenStartTime = null;
	state.tokenCount = 0;
	state.currentTPS = 0;
}

export function incrementTokenCount() {
	if (!state.tokenStartTime) {
		state.tokenStartTime = Date.now();
		state.tokenCount = 0;
	}
	state.tokenCount++;
}

export function calculateTPS() {
	if (state.tokenStartTime && state.tokenCount > 0) {
		const elapsedSeconds = (Date.now() - state.tokenStartTime) / 1000;
		state.currentTPS = (state.tokenCount / elapsedSeconds).toFixed(1);
		return state.currentTPS;
	}
	return 0;
}
