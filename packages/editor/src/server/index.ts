import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { assetsRouter } from './routes/assets.js';
import { scenesRouter } from './routes/scenes.js';
import { texturesRouter } from './routes/textures.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = parseInt(process.env.PORT ?? '3001', 10);

// Project root: where scenes and assets live.
// In dev: packages/editor/public/
// Can be overridden via PROJECT_DIR env var.
const PROJECT_DIR = process.env.PROJECT_DIR
    ?? path.resolve(__dirname, '../../public');

const app = express();

app.use(express.json());

// API routes
app.use('/api/assets', assetsRouter(PROJECT_DIR));
app.use('/api/scenes', scenesRouter(PROJECT_DIR));
app.use('/api/textures', texturesRouter(PROJECT_DIR));

// Serve project assets (models, textures) so the engine can fetch them
app.use('/', express.static(PROJECT_DIR));

// In production, serve the built Vue app
const clientDist = path.resolve(__dirname, '../../dist-client');
app.use(express.static(clientDist));

// SPA fallback
app.get('/*path', (_req, res) => {
    res.sendFile(path.join(clientDist, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Skengio Editor server running at http://localhost:${PORT}`);
    console.log(`Project directory: ${PROJECT_DIR}`);
});
