import type { FormEvent } from 'react';
import { useEffect, useRef, useState } from 'react';
import { get, post } from '../../app/api';
import { pushNotice, setCoordinatorGraph } from '../../app/store';
import { graphFrom, type Receipt, type SessionSummary, type Snapshot } from './types';

const STORAGE_KEY = 'milknado.coordinator.session';

interface SessionScope {
  id: string;
  active: boolean;
  request: number;
  controllers: Set<AbortController>;
}

function newScope(id: string): SessionScope {
  return { id, active: true, request: 0, controllers: new Set() };
}

function abortSnapshots(scope: SessionScope): void {
  for (const controller of scope.controllers) controller.abort();
  scope.controllers.clear();
}

export function useCoordinatorSession() {
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(STORAGE_KEY) ?? '');
  const scope = useRef<SessionScope>(newScope(sessionId));
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [available, setAvailable] = useState(false);
  const [goal, setGoal] = useState('');
  const [provider, setProvider] = useState<'claude' | 'codex'>('claude');
  const [busy, setBusy] = useState(false);
  const [commandResult, setCommandResult] = useState('');

  useEffect(() => {
    let active = true;
    void fetch('/api/coordinators')
      .then(async (response) => {
        if (response.status === 409) return null;
        if (!response.ok) throw new Error('Coordinator discovery failed.');
        return await response.json() as SessionSummary[];
      })
      .then((items) => { if (active && items) { setSessions(items); setAvailable(true); } })
      .catch(() => { if (active) pushNotice('Failed to discover coordinator sessions.'); });
    return () => { active = false; };
  }, []);

  useEffect(() => () => {
    scope.current.active = false;
    abortSnapshots(scope.current);
  }, []);

  function selectSession(id: string): void {
    if (scope.current.id === id) return;
    scope.current.active = false;
    abortSnapshots(scope.current);
    scope.current = newScope(id);
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
    setSessionId(id);
    setSnapshot(null);
    setCoordinatorGraph(null);
    setCommandResult('');
    setBusy(false);
  }

  async function refresh(currentScope: SessionScope, failureNotice = 'Failed to load coordinator status.'): Promise<void> {
    if (scope.current !== currentScope || !currentScope.active) return;
    abortSnapshots(currentScope);
    const controller = new AbortController();
    currentScope.controllers.add(controller);
    const request = ++currentScope.request;
    try {
      const value = await get<Snapshot>(`/api/coordinators/${encodeURIComponent(currentScope.id)}/snapshot`,
        controller.signal, () => scope.current === currentScope && currentScope.active &&
          request === currentScope.request && !controller.signal.aborted);
      if (scope.current === currentScope && currentScope.active && request === currentScope.request && !controller.signal.aborted && value) {
        setSnapshot(value);
        setCoordinatorGraph(graphFrom(value));
      }
    } catch {
      if (scope.current === currentScope && currentScope.active && !controller.signal.aborted) {
        pushNotice(failureNotice);
      }
    } finally {
      currentScope.controllers.delete(controller);
    }
  }

  useEffect(() => {
    if (!sessionId || !available) return;
    const currentScope = scope.current;
    let timer: number | undefined;
    async function poll(): Promise<void> {
      if (!currentScope.active || scope.current !== currentScope) return;
      if (currentScope.controllers.size === 0) await refresh(currentScope);
      if (currentScope.active && scope.current === currentScope) {
        timer = window.setTimeout(() => void poll(), 2000);
      }
    }
    void poll();
    return () => {
      currentScope.active = false;
      window.clearTimeout(timer);
      abortSnapshots(currentScope);
    };
  }, [sessionId, available]);

  async function send(kind: string, fields: object = {}): Promise<void> {
    const currentScope = scope.current;
    if (!currentScope.id || !currentScope.active) return;
    setBusy(true);
    try {
      const result = await post<Receipt>(`/api/coordinators/${encodeURIComponent(currentScope.id)}/commands`, {
        kind, command_id: crypto.randomUUID(), ...fields,
      }, () => scope.current === currentScope && currentScope.active);
      if (scope.current !== currentScope || !currentScope.active) return;
      if (result && result.status !== 'accepted') pushNotice(String(result.result ?? result.status));
      if (result) setCommandResult(`${kind}: ${result.status} · ${JSON.stringify(result.result)}`);
      await refresh(currentScope, 'Coordinator command failed.');
    } catch {
      if (scope.current === currentScope && currentScope.active) pushNotice('Coordinator command failed.');
    } finally {
      if (scope.current === currentScope && currentScope.active) setBusy(false);
    }
  }

  async function start(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!goal.trim()) return;
    let currentScope = scope.current;
    setBusy(true);
    try {
      const receipt = await post<Receipt>('/api/coordinators/commands', {
        kind: 'start_goal', command_id: crypto.randomUUID(), description: goal.trim(), provider,
      }, () => scope.current === currentScope && currentScope.active);
      if (scope.current !== currentScope || !currentScope.active) return;
      if (receipt?.status !== 'accepted' || !receipt.result || typeof receipt.result !== 'object' || !receipt.result.id) {
        pushNotice(String(receipt?.result ?? 'Goal intake failed.'));
        return;
      }
      selectSession(receipt.result.id);
      currentScope = scope.current;
      const current = await get<SessionSummary[]>('/api/coordinators', undefined,
        () => scope.current === currentScope && currentScope.active);
      if (scope.current === currentScope && currentScope.active && current) setSessions(current);
    } catch {
      if (scope.current === currentScope && currentScope.active) pushNotice('Goal intake failed.');
    } finally {
      if (scope.current === currentScope && currentScope.active) setBusy(false);
    }
  }

  return { available, busy, commandResult, goal, provider, sessionId, sessions, snapshot,
    selectSession, send, setGoal, setProvider, start };
}
