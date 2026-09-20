import { afterEach, describe, expect, it } from 'vitest';
import { getActiveTab, resetTab, setActiveTab, subscribeTab } from './detailTab';

describe('detailTab', () => {
  afterEach(resetTab);

  it('defaults to the session tab', () => {
    expect(getActiveTab()).toBe('session');
  });

  it('switches the active tab and notifies subscribers once', () => {
    let notifications = 0;
    const unsubscribe = subscribeTab(() => {
      notifications += 1;
    });

    setActiveTab('changes');
    setActiveTab('changes');

    expect(getActiveTab()).toBe('changes');
    expect(notifications).toBe(1);
    unsubscribe();
  });
});
