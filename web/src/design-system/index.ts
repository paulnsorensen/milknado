// The design system loader. Import this module once, before any component
// use, so window.Milknado (and window.Milknado.React) are ready.
import { React } from './globals';
import '../../vendor/milknado/tokens.css';
import '../../vendor/milknado/components/bundle.css';
import '../../vendor/milknado/components/bundle.js';

type MilknadoWithReact = Window['Milknado'] & { React: typeof React };

const milknado = window.Milknado as MilknadoWithReact;
milknado.React = React;

export const Milknado = milknado;
