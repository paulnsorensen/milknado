import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { freeCoreWorkers } from './vitest.workers';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    maxWorkers: freeCoreWorkers(),
  },
});
