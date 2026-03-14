import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import multer from 'multer';
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.hdr', '.exr', '.bmp', '.tga']);
export function texturesRouter(projectDir) {
    const router = Router();
    const texturesDir = () => path.join(projectDir, 'textures');
    // Multer storage: save uploaded files into the textures root,
    // preserving original filename (sanitised to prevent path traversal).
    const storage = multer.diskStorage({
        destination: (_req, _file, cb) => {
            const dest = texturesDir();
            fs.mkdir(dest, { recursive: true }).then(() => cb(null, dest), (e) => cb(e, dest));
        },
        filename: (_req, file, cb) => {
            // Strip any directory components to prevent path traversal
            const safe = path.basename(file.originalname);
            cb(null, safe);
        },
    });
    const upload = multer({
        storage,
        fileFilter: (_req, file, cb) => {
            const ext = path.extname(file.originalname).toLowerCase();
            if (IMAGE_EXTS.has(ext)) {
                cb(null, true);
            }
            else {
                cb(new Error(`Unsupported file type: ${ext}`));
            }
        },
        limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB
    });
    // List all texture files.
    // Cubemap folders (inside cubemaps/) are returned as single entries
    // e.g. "cubemaps/nissi" instead of listing individual face files.
    router.get('/', async (_req, res) => {
        try {
            const files = await listTextureFiles(texturesDir());
            res.json(files);
        }
        catch {
            res.json([]);
        }
    });
    // Upload one or more texture files.
    // Accepts multipart/form-data with field name "textures".
    // Optional "folder" text field to place files in a subfolder (e.g. "cubemaps/myEnv").
    router.post('/upload', upload.array('textures', 20), (req, res) => {
        const files = req.files;
        if (!files || files.length === 0) {
            res.status(400).json({ error: 'No files uploaded' });
            return;
        }
        const uploaded = files.map(f => {
            // Return the relative path from the textures root
            const rel = path.relative(texturesDir(), f.path).replace(/\\/g, '/');
            return rel;
        });
        res.json({ uploaded });
    });
    return router;
}
async function listTextureFiles(dir, prefix = '') {
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
            if (prefix === 'cubemaps') {
                // Cubemap subdirectory: treat as a single entry
                results.push(rel);
            }
            else {
                const sub = await listTextureFiles(path.join(dir, d.name), rel);
                results.push(...sub);
            }
        }
        else if (IMAGE_EXTS.has(path.extname(d.name).toLowerCase())) {
            results.push(rel);
        }
    }
    return results;
}
