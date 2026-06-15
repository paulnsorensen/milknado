// Milknado native Workflow node-runner (harness-side, "ultracode").
//
// Runs INSIDE a Claude Code dynamic-Workflow session. It is the DRIVER of the
// native execution backend: milknado is the graph/state backend (SQLite + MCP
// tools), and this script fans out one agent() per ready node, batch by batch,
// against milknado's dependency graph.
//
// It runs ALONGSIDE — never replaces — the subprocess CLI dispatcher. Use this
// only in a Claude Code session in Workflow mode; Codex/opencode have no
// ultracode primitive, so they keep using the subprocess path.
//
// End-to-end verification REQUIRES a live ultracode session and cannot run in an
// autonomous/CI harness. See the spec's acceptance note: drive a 3–5 node graph
// through plan_batches -> claim -> agent loop -> node_verify -> deposit ->
// markTerminal and confirm terminal node status in SQLite.
//
// MCP tools used (all on the `milknado` server):
//   milknado_plan_batches   -> { batches: [{ index, change_ids, depends_on, oversized }, ...] }
//     NOTE: change-id-keyed, NOT node IDs. main() requires batches PRE-RESOLVED to
//     node-id arrays ([[nodeId, ...], ...]); see the resolution note in main().
//   milknado_todo_claim(nodeId) -> structured claim payload:
//     { run_id, node_id, brief, flavor, model, tools, worktree_path,
//       agent_type, loop_mode, max_iterations, max_turns }
//   milknado_node_verify(run_id) -> { ok, feedback }
//   milknado_deposit_result(run_id, payload)
//   milknado_todo_set_status(nodeId, "done")  -> terminal transition (verify-gated)
//
// Default loop is mode B ("redispatch"): a COLD agent() per iteration against the
// node's durable worktree, carrying forward ONLY the verify feedback string —
// never a transcript. Per-flavor "single" (mode A) runs one agent() that loops
// internally, calling milknado_node_verify as a tool until it passes.

async function runWorkerBrief(claim, feedback) {
  // Build the per-iteration brief: the claim's brief plus the prior verify
  // feedback (empty on the first iteration). The worktree carries forward all
  // prior on-disk work; only this feedback string crosses the iteration boundary.
  const fb = feedback ? `\n\n## Verify feedback from the previous attempt\n${feedback}\n` : "";
  return `${claim.brief}${fb}\n\nYour run_id is ${claim.run_id}. Work in ${claim.worktree_path}.`;
}

async function runNodeRedispatch(claim) {
  // Mode B: cold re-dispatch per iteration, bounded per-iteration context.
  let feedback = "";
  for (let i = 0; i < claim.max_iterations; i++) {
    await agent(await runWorkerBrief(claim, feedback), {
      agentType: claim.agent_type,
      model: claim.model,
      tools: claim.tools,
      maxTurns: claim.max_turns,
    });
    const verdict = await milknado_node_verify({ run_id: claim.run_id });
    if (verdict.ok) return true;
    feedback = verdict.feedback;
  }
  return false;
}

async function runNodeSingle(claim) {
  // Mode A: one agent() loops internally, calling milknado_node_verify itself
  // until it passes. Context grows monotonically — opt-in for short flavors only.
  await agent(await runWorkerBrief(claim, ""), {
    agentType: claim.agent_type,
    model: claim.model,
    tools: claim.tools,
    maxTurns: claim.max_turns,
  });
  // The single agent is responsible for reaching ok=True via its own verify
  // calls; re-confirm server-side before declaring the node done.
  const verdict = await milknado_node_verify({ run_id: claim.run_id });
  return verdict.ok;
}

async function runNode(nodeId) {
  const claim = await milknado_todo_claim({ node_id: nodeId }); // creates worktree, no spawn
  const ok =
    claim.loop_mode === "single" ? await runNodeSingle(claim) : await runNodeRedispatch(claim);
  // milknado_deposit_result is the durable result sink; markTerminal is the
  // verify-gated "done" transition — it is REJECTED unless the latest
  // milknado_node_verify(run_id) returned ok=True, so a node that never passed
  // its gates cannot be marked done even if this driver mis-sequences.
  await milknado_deposit_result({
    run_id: claim.run_id,
    payload: ok ? "node completed; gates passed" : "node FAILED to pass gates within max_iterations",
  });
  if (ok) await milknado_todo_set_status({ node_id: nodeId, status: "done" });
  return ok;
}

async function main(args) {
  // args.batchPlan.batches must be PRE-RESOLVED to node-id arrays: each entry an
  // array of ready task node IDs, run in sequence (later batches depend on
  // earlier ones) with the nodes WITHIN a batch run in parallel.
  //
  // milknado_plan_batches does NOT return this shape — it returns change-id-keyed
  // batch objects ({ index, change_ids, depends_on, oversized }), and change_ids
  // are planning-domain identifiers with no server-side mapping to graph node
  // IDs. Resolving them requires the live session's own change-id -> owning-node
  // map (the same one the orchestrator built when it fed `changes` into
  // milknado_plan_batches); there is no MCP tool to do it from inside this runner.
  //
  // DEFERRED to live-session wiring: the orchestrator must produce args.batchPlan
  // by resolving change_ids -> owning node IDs (or a future MCP tool must return
  // node-id batches directly). This runner therefore REQUIRES the pre-resolved
  // node-id shape and fails loud below if handed the raw plan_batches output,
  // rather than crashing obscurely at batch.map on a change-id object.
  const plan = args.batchPlan;
  if (!plan || !Array.isArray(plan.batches)) {
    throw new Error("node-runner: args.batchPlan.batches must be an array of node-id batches");
  }
  for (const batch of plan.batches) {
    if (!Array.isArray(batch) || !batch.every((id) => typeof id === "number")) {
      throw new Error(
        "node-runner: each batch must be an array of numeric node IDs. Got " +
          JSON.stringify(batch) +
          " — this looks like raw milknado_plan_batches output (change-id objects). " +
          "Resolve change_ids -> owning node IDs before passing the plan in (see main() note).",
      );
    }
    await parallel(batch.map((nodeId) => async () => runNode(nodeId)));
  }
}
