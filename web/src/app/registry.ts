// Discovers every `web/src/features/*/index.ts(x)` module, eagerly, in
// sorted path order, and calls its optional `register()` export. A feature
// registers its slots (slots.ts) and actions (actions.ts) from `register()`.
export interface FeatureModule {
  register?: () => void;
}

const modules = import.meta.glob<FeatureModule>('../features/*/index.{ts,tsx}', {
  eager: true,
});

/** The pure ordering-and-invocation step, tested without a real glob result. */
export function registerFeaturesFrom(discovered: Record<string, FeatureModule>): string[] {
  const paths = Object.keys(discovered).sort();
  for (const path of paths) {
    discovered[path].register?.();
  }
  return paths;
}

let registeredPaths: string[] | null = null;

/** Runs `registerFeaturesFrom` at most once; a repeat call returns the first result. */
export function registerFeatures(): string[] {
  registeredPaths ??= registerFeaturesFrom(modules);
  return registeredPaths;
}

/** Test-only reset, mirroring `clearSlots`, so a suite can call `registerFeatures()` again. */
export function resetFeatureRegistration(): void {
  registeredPaths = null;
}
