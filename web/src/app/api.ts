// A thin fetch wrapper: 401 redirects to the login page, 409 pushes a notice
// with the domain-supplied reason instead of throwing on the caller.
import { pushNotice } from './store';

async function request<T>(method: string, path: string, body?: unknown): Promise<T | null> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401) {
    window.location.assign('/');
    return null;
  }
  if (response.status === 409) {
    const payload = (await response.json().catch(() => ({}))) as { reason?: string; error?: string };
    pushNotice(payload.reason ?? payload.error ?? 'The server rejected the request.');
    return null;
  }
  if (!response.ok) {
    throw new Error(`${method} ${path} failed with status ${response.status}.`);
  }
  if (response.status === 204) {
    return null;
  }
  return (await response.json()) as T;
}

export function get<T>(path: string): Promise<T | null> {
  return request<T>('GET', path);
}

export function post<T>(path: string, body?: unknown): Promise<T | null> {
  return request<T>('POST', path, body ?? {});
}

export function patch<T>(path: string, body: unknown): Promise<T | null> {
  return request<T>('PATCH', path, body);
}
