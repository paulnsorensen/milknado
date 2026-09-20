import './design-system';
import { createRoot } from 'react-dom/client';
import { registerFeatures } from './app/registry';
import { Shell } from './app/Shell';

registerFeatures();

const container = document.getElementById('root');
if (!container) {
  throw new Error('Missing #root element.');
}
createRoot(container).render(<Shell />);
