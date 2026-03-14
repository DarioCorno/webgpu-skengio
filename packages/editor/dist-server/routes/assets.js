import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import multer from 'multer';
export function assetsRouter(projectDir) {
    const router = Router();
    // List all files under a subdirectory (default: everything)
    router.get('/', async (req, res) => {
        const subDir = req.query.path ?? '';
        const target = path.resolve(projectDir, subDir);
        // Prevent path traversal
        if (!target.startsWith(projectDir)) {
            res.status(400).json({ error: 'Invalid path' });
            return;
        }
        try {
            const entries = await listDir(target, projectDir);
            res.json(entries);
        }
        catch {
            res.json([]);
        }
    });
    // Upload files to a subdirectory
    const upload = multer({ dest: path.join(projectDir, '.tmp-uploads') });
    router.post('/upload', upload.single('file'), async (req, res) => {
        if (!req.file) {
            res.status(400).json({ error: 'No file provided' });
            return;
        }
        const targetDir = req.body.directory ?? 'models';
        const destDir = path.resolve(projectDir, targetDir);
        if (!destDir.startsWith(projectDir)) {
            res.status(400).json({ error: 'Invalid directory' });
            return;
        }
        await fs.mkdir(destDir, { recursive: true });
        const destPath = path.join(destDir, req.file.originalname);
        await fs.rename(req.file.path, destPath);
        const relativePath = path.relative(projectDir, destPath).replace(/\\/g, '/');
        res.json({ path: '/' + relativePath });
    });
    // Delete a file
    router.delete('/*path', async (req, res) => {
        const filePath = path.resolve(projectDir, req.params.path ?? '');
        if (!filePath.startsWith(projectDir)) {
            res.status(400).json({ error: 'Invalid path' });
            return;
        }
        try {
            await fs.unlink(filePath);
            res.json({ deleted: true });
        }
        catch {
            res.status(404).json({ error: 'File not found' });
        }
    });
    return router;
}
async function listDir(dir, root) {
    const entries = [];
    let dirents;
    try {
        dirents = await fs.readdir(dir, { withFileTypes: true });
    }
    catch {
        return entries;
    }
    for (const d of dirents) {
        if (d.name.startsWith('.'))
            continue;
        const full = path.join(dir, d.name);
        const rel = '/' + path.relative(root, full).replace(/\\/g, '/');
        if (d.isDirectory()) {
            const children = await listDir(full, root);
            entries.push({ name: d.name, path: rel, type: 'directory', children });
        }
        else {
            const stat = await fs.stat(full);
            entries.push({ name: d.name, path: rel, type: 'file', size: stat.size });
        }
    }
    return entries;
}
