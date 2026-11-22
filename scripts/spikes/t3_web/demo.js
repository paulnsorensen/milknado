"use strict";

const element = (id) => document.getElementById(id);
const show = (id, text) => { element(id).textContent = text; };
const CANCEL_UNCONFIRMED = "Cancellation requested · stop unconfirmed";
const initial = {
  attempt: "A-01",
  runtime: "Running",
  evidence: null,
  accepted: false,
  answered: false,
  notice: "",
};
let state = initial;

function render() {
  const running = state.runtime === "Running";
  const mayFinish = running || state.runtime === CANCEL_UNCONFIRMED;
  const reviewPending = state.evidence !== null && !state.accepted;
  show("attempt", `${state.attempt} · T3`);
  show("runtime", state.runtime);
  show("review", state.accepted ? "Evidence accepted" :
    reviewPending ? "Evidence ready · awaiting review" : "No evidence");
  show("artifact", state.evidence ?? "None");
  show("b-status", state.accepted ? "Ready: A is accepted" : "Blocked: A is not accepted");
  show("dependency", state.accepted ? "B has no prerequisite blocker." : "B waits for acceptance of A.");
  show("question", state.answered ? "User question · answered in fixture" :
    "User question · Which audience? (Fixture answer: new contributors)");
  show("notice", state.notice);
  element("review-task").hidden = !reviewPending;
  element("complete").disabled = !mayFinish;
  element("accept").disabled = !reviewPending;
  element("cancel").disabled = !running;
  element("disconnect").disabled = !mayFinish;
  element("replace").disabled = state.runtime !== "Unknown · disconnected" || state.attempt !== "A-01";
  element("replay").disabled = state.attempt !== "A-02";
  element("answer").disabled = state.answered;
}

function complete(attempt) {
  if (attempt !== state.attempt) {
    return { ...state, notice: `Ignored completion from attempt ${attempt}.` };
  }
  return { ...state, runtime: "Completed", evidence: "onboarding-draft.md (fixture)", notice: "" };
}

const transitions = {
  complete: () => complete(state.attempt),
  accept: () => ({ ...state, accepted: true }),
  cancel: () => ({ ...state, runtime: CANCEL_UNCONFIRMED }),
  disconnect: () => ({ ...state, runtime: "Unknown · disconnected" }),
  replace: () => ({
    ...initial,
    attempt: "A-02",
    answered: state.answered,
    notice: "Loaded a separate fixture with a replacement attempt. No retry or stop occurred.",
  }),
  replay: () => complete("A-01"),
  answer: () => ({ ...state, answered: true }),
  reset: () => initial,
};

for (const [id, transition] of Object.entries(transitions)) {
  element(id).addEventListener("click", () => {
    state = transition();
    render();
  });
}
render();
