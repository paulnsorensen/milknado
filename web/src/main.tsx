import './design-system';
import { createRoot } from 'react-dom/client';
import { Shell } from './app/Shell';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Missing #root element.');
}
createRoot(container).render(<Shell />);
