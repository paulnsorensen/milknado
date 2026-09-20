import { fileURLToPath, URL } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Byte-stable output: Vite's default asset names are content-hashed (no
// timestamps, no per-build salt), so a rebuild of unchanged sources produces
// byte-identical files. The static dir is committed; keep it that way.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL('../src/milknado/web/static', import.meta.url)),
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
});
