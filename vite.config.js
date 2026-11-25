import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
	resolve: {
		// The sign models run on the CPU (WASM) build of the ONNX runtime: its WASM file is
		// 15 MB, against 29 MB for the default build that also carries WebGPU. Node (the
		// tests) keeps the package's own Node build.
		alias: [{ find: /^onnxruntime-web$/, replacement: 'onnxruntime-web/wasm' }],
	},
	build: {
		// Two pages: the app, and the pipeline dashboard (dashboard.html).
		rollupOptions: {
			input: {
				app: resolve(import.meta.dirname, 'index.html'),
				dashboard: resolve(import.meta.dirname, 'dashboard.html'),
			},
		},
	},
});
