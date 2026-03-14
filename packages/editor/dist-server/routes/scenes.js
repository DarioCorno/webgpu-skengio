import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
export function scenesRouter(projectDir) {
    const router = Router();
    const scenesDir = () => path.join(projectDir, 'scenes');
    // List all scene JSON files
    router.get('/', async (_req, res) => {
        try {
            const files = await listSceneFiles(scenesDir());
            res.json(files);
        }
        catch {
            res.json([]);
        }
    });
    // Read a scene JSON by name (supports subdirs: "animations/gltf-boy")
    router.get('/*name', async (req, res) => {
        const raw = req.params.name;
        const name = Array.isArray(raw) ? raw.join('/') : (raw ?? '');
        const filePath = path.resolve(scenesDir(), name.endsWith('.json') ? name : name + '.json');
        if (!filePath.startsWith(path.resolve(projectDir))) {
            res.status(400).json({ error: 'Invalid path' });
            return;
        }
        try {
            const content = await fs.readFile(filePath, 'utf-8');
            res.json(JSON.parse(content));
        }
        catch {
            res.status(404).json({ error: 'Scene not found' });
        }
    });
    // Save/update a scene JSON
    router.put('/*name', async (req, res) => {
        const raw = req.params.name;
        const name = Array.isArray(raw) ? raw.join('/') : (raw ?? '');
        const filePath = path.resolve(scenesDir(), name.endsWith('.json') ? name : name + '.json');
        if (!filePath.startsWith(path.resolve(projectDir))) {
            res.status(400).json({ error: 'Invalid path' });
            return;
        }
        try {
            // Validate JSON
            const json = JSON.stringify(req.body, null, 2);
            await fs.mkdir(path.dirname(filePath), { recursive: true });
            await fs.writeFile(filePath, json, 'utf-8');
            res.json({ saved: true, path: filePath });
        }
        catch (err) {
            res.status(500).json({ error: 'Failed to save scene' });
        }
    });
    return router;
}
async function listSceneFiles(dir, prefix = '') {
    const results = [];
    let dirents;
    try {
        dirents = await fs.readdir(dir, { withFileTypes: true });
    }
    catch {
        return results;
    }
    for (const d of dirents) {
        const rel = prefix ? `${prefix}/${d.name}` : d.name;
        if (d.isDirectory()) {
            const sub = await listSceneFiles(path.join(dir, d.name), rel);
            results.push(...sub);
        }
        else if (d.name.endsWith('.json')) {
            results.push(rel);
        }
    }
    return results;
}
