// Sets window.React/window.ReactDOM from the npm React import.
//
// This module has no further imports, so its top-level assignments run
// immediately when it is evaluated. `design-system/index.ts` imports this
// module FIRST, before the vendored classic bundle, so the bundle always
// reads the app's own React instance and never triggers an invalid hook
// call from a duplicate React copy.
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import * as ReactDOMClient from 'react-dom/client';

const reactDom = { ...ReactDOM, ...ReactDOMClient };

declare global {
  interface Window {
    React: typeof React;
    ReactDOM: typeof reactDom;
  }
}

window.React = React;
window.ReactDOM = reactDom;

export { React };
