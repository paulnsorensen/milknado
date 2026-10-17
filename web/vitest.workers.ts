import { availableParallelism, loadavg } from 'node:os';

/**
 * Size the vitest pool from free cores so concurrent gates share one host.
 *
 * The pool is the free cores, at least 2, capped by TEST_WORKERS_MAX. The cap
 * defaults to vitest's own run-mode default, so the pool only shrinks.
 * os.loadavg() works on Linux and macOS; on Windows it returns zeros.
 */
export function freeCoreWorkers(
  env: NodeJS.ProcessEnv = process.env,
  cpus: number = availableParallelism(),
  load: number = loadavg()[0],
): number {
  const cap =
    env.TEST_WORKERS_MAX === undefined ? Math.max(1, cpus - 1) : Number(env.TEST_WORKERS_MAX);
  if (!Number.isInteger(cap) || cap < 1) {
    throw new Error(`TEST_WORKERS_MAX must be a positive integer, got ${env.TEST_WORKERS_MAX}`);
  }
  return Math.min(cap, Math.max(2, Math.floor(cpus - load)));
}
