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

let registered = false;

/** Runs `registerFeaturesFrom` at most once, guarding against a re-import or HMR double-registering every slot and action. */
export function registerFeatures(): string[] {
  if (registered) {
    return [];
  }
  registered = true;
  return registerFeaturesFrom(modules);
}

/** Test-only reset, mirroring `clearSlots`, so a suite can call `registerFeatures()` again. */
export function resetFeatureRegistration(): void {
  registered = false;
}
