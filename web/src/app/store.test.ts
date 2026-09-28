import { beforeEach, describe, expect, it } from "vitest";
import {
  canActOnSelectedRun,
  getState,
  resetStore,
  selectedNodeId,
  setGraphView,
  setSelection,
  setSnapshot,
} from "./store";

const CAPABILITIES = {
  session_input: { available: true, reason: null },
  cancel: { available: true, reason: null },
  force_stop: { available: true, reason: null },
  stop_scheduling: { available: true, reason: null },
  graph_edits: { available: true, reason: null },
  review_decision: { available: true, reason: null },
  git: { available: true, reason: null },
  host_owner: { available: true, reason: null },
  owner: { available: true, run_id: "run-1", node_id: 7 },
};

describe("store", () => {
  beforeEach(resetStore);
  it("maps a selected active run to its node", () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: CAPABILITIES,
      active_runs: [{ run_id: "run-1", node_id: 7 }],
    });
    setSelection("run-1");

    expect(selectedNodeId(getState())).toBe(7);

    setSnapshot({ goal: null, graph: null, capabilities: CAPABILITIES });
    expect(selectedNodeId(getState())).toBe(7);
  });

  it("allows actions only for the selected owner run", () => {
    setSnapshot({ goal: null, graph: null, capabilities: CAPABILITIES });
    setSelection(7);
    expect(canActOnSelectedRun(getState())).toBe(true);

    setSelection(8);
    expect(canActOnSelectedRun(getState())).toBe(false);
  });

  it("round-trips a selection", () => {
    setSelection(7);
    expect(getState().selection).toBe(7);
  });

  it("merges a graph view patch without dropping other fields", () => {
    setGraphView({ zoom: 1.5 });
    setGraphView({ filter: "ready" });
    expect(getState().graphView).toMatchObject({ zoom: 1.5, filter: "ready" });
  });
});
