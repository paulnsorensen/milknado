const PATH_TITLE_LIMIT = 80;

export function summarizePathTitle(title: string): string {
  const firstSentence =
    title.match(/^[\s\S]*?(?<!e\.g)(?<!i\.e)[.!?](?=\s+[A-ZÀ-ÖØ-Þ]|$)/)?.[0] ??
    title;
  const summary = firstSentence.trim();
  if (summary.length <= PATH_TITLE_LIMIT) {
    return summary;
  }
  return `${summary.slice(0, PATH_TITLE_LIMIT - 1).trimEnd()}…`;
}
