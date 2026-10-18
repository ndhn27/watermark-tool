// WmServer (the page's DOM-free API client) against a REAL cli/serve.py.
// Run by tests/test_web_client.py, which starts the server and sets WM_TEST_SERVER.
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { clientSource, SERVER } from './harness.mjs';

const sandbox = { fetch, URLSearchParams, AbortController, setTimeout, clearTimeout, Blob, File };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(clientSource(), sandbox);
const { WmServer } = sandbox;

// 1x1 PNG
const PNG = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='), (c) => c.charCodeAt(0));
const bigPng = async () => {
  // a 400x300 flat-ish image so there is room for the mark: build a PNG via the server's own decoder is overkill;
  // instead create a minimal uncompressed-ish PNG with zlib.
  const { deflateSync } = await import('node:zlib');
  const w = 400, h = 300, raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; for (let x = 0; x < w * 3; x++) raw[y * (w * 3 + 1) + 1 + x] = (x * 7 + y * 3) & 255; }
  const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc = (b) => { let c = 0xFFFFFFFF; for (const x of b) c = crcTable[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
  const chunk = (type, data) => { const t = Buffer.from(type), len = Buffer.alloc(4); len.writeUInt32BE(data.length); const c = Buffer.alloc(4); c.writeUInt32BE(crc(Buffer.concat([t, data]))); return Buffer.concat([len, t, data, c]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return new File([Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))])], 'photo.png', { type: 'image/png' });
};

test('baseUrl reads ?server= and trims slashes', () => {
  assert.equal(WmServer.baseUrl(''), '');
  assert.equal(WmServer.baseUrl('?server=http://127.0.0.1:9000/'), 'http://127.0.0.1:9000');
});

test('pageToSpec converts the page percentage opacity to the API alpha', () => {
  const s = WmServer.pageToSpec({ text: 'x', opacityPct: 35, fontSize: 34, angle: 30, spacing: 220 });
  assert.deepEqual({ ...s }, { text: 'x', opacity: 89, font_size: 34, angle: 30, spacing: 220 });
  assert.equal(WmServer.pageToSpec({ text: 'x', opacityPct: 100, fontSize: 1, angle: 0, spacing: 1 }).opacity, 255);
  assert.equal(WmServer.pageToSpec({ text: 'x', opacityPct: 500, fontSize: 1, angle: 0, spacing: 1 }).opacity, 255);
});

test('detect() finds the server and returns its schema', async () => {
  const info = await WmServer.detect(SERVER);
  assert.ok(info, 'server should answer /api/schema');
  assert.ok(info.watermark.length >= 20);
  assert.ok(info.groups.some((g) => g.id === 'invisible'));
  assert.equal(typeof info.capabilities.c2pa, 'boolean');
});

test('detect() returns null when nothing is listening, and for non-schema responses', async () => {
  assert.equal(await WmServer.detect('http://127.0.0.1:9', undefined, 800), null);
  const notFound = async () => ({ ok: false });
  assert.equal(await WmServer.detect('', notFound), null);
  const html = async () => ({ ok: true, json: async () => { throw new SyntaxError('not json'); } });
  assert.equal(await WmServer.detect('', html), null);
  const wrong = async () => ({ ok: true, json: async () => ({ hello: 'world' }) });
  assert.equal(await WmServer.detect('', wrong), null);
});

test('controls(): schema order, minus page-covered, video-only and server-local options', async () => {
  const info = await WmServer.detect(SERVER);
  const names = WmServer.controls(info).map((o) => o.name);
  for (const covered of WmServer.PAGE_COVERED) assert.ok(!names.includes(covered), covered);
  assert.ok(!names.includes('use_shared_memory') && !names.includes('jitter_refresh_seconds'));
  for (const local of ['font_path', 'c2pa_cert', 'c2pa_key', 'face_cascade', 'adaptive_preview']) assert.ok(!names.includes(local), local);
  for (const wanted of ['blend', 'ink', 'invisible', 'invisible_mode', 'ss_key', 'author', 'adaptive', 'seed', 'stamp_id', 'grid_id']) assert.ok(names.includes(wanted), wanted);
  assert.ok(!names.includes('id_value') && !names.includes('ledger'), 'an explicit ID names one file; the ledger is the server\'s');
  assert.ok(names.indexOf('blend') < names.indexOf('invisible'));
});

test('controls() hides C2PA when this install cannot sign', async () => {
  const info = await WmServer.detect(SERVER);
  const noC2pa = { ...info, capabilities: { ...info.capabilities, c2pa: false } };
  const names = WmServer.controls(noC2pa).map((o) => o.name);
  assert.ok(!names.includes('c2pa') && !names.includes('c2pa_title'));
});

test('visible()/cleanExtras(): dependent options only count when their requirements hold', async () => {
  const info = await WmServer.detect(SERVER);
  const ss = info.watermark.find((o) => o.name === 'ss_key');
  assert.equal(WmServer.visible(ss, { invisible: false, invisible_mode: 'spread' }), false);
  assert.equal(WmServer.visible(ss, { invisible: true, invisible_mode: 'lsb' }), false);
  assert.equal(WmServer.visible(ss, { invisible: true, invisible_mode: 'spread' }), true);

  const raw = { blend: 'multiply', ink: '0.3', invisible: false, invisible_mode: 'spread', ss_key: '99', author: '', seed: '7', adaptive: false };
  assert.deepEqual({ ...WmServer.cleanExtras(info, raw) }, { blend: 'multiply', ink: 0.3, seed: 7 });   // ss_key dropped (invisible off), author empty
  const on = { ...raw, invisible: true };
  assert.deepEqual({ ...WmServer.cleanExtras(info, on) },
    { blend: 'multiply', ink: 0.3, seed: 7, invisible: true, invisible_mode: 'spread', ss_key: 99 });
});

test('watermark() round-trips an image through the server and reports notes/keys', async () => {
  const info = await WmServer.detect(SERVER);
  const file = await bigPng();
  const spec = { ...WmServer.pageToSpec({ text: '@me', opacityPct: 35, fontSize: 24, angle: 30, spacing: 120 }),
                 ...WmServer.cleanExtras(info, { blend: 'multiply', invisible: true, invisible_mode: 'spread', seed: '7' }) };
  const { blob, meta } = await WmServer.watermark(SERVER, file, spec, 'png');
  assert.equal(blob.type, 'image/png');
  assert.ok(blob.size > 1000);
  assert.equal(meta.kind, 'image');
  assert.equal(meta.width, 400);
  assert.ok(Number.isInteger(meta.ss_key), 'a spread mark without a key gets one generated and returned');
  assert.ok(meta.notes.some((n) => n.code === 'ss_key_generated'));
});

test('watermark() turns server errors into Errors carrying the server message', async () => {
  const file = await bigPng();
  await assert.rejects(() => WmServer.watermark(SERVER, file, { text: 'x', spacing: 2 }, 'png'),
    (e) => e.status === 413 && /watermark tiles/.test(e.message));
  await assert.rejects(() => WmServer.watermark(SERVER, file, { text: 'x', opacity: 999 }, 'png'),
    (e) => e.status === 422 && /opacity/.test(e.message));
  await assert.rejects(() => WmServer.watermark(SERVER, new File([PNG.slice(0, 8)], 'bad.png'), { text: 'x' }, 'png'),
    (e) => e.status === 415);
});

test('derive_seed_from gives each file its own pattern from one seed (as --batch does)', async () => {
  const file = await bigPng();
  const spec = { text: '@me', seed: 7, spacing: 120 };
  const same1 = (await WmServer.watermark(SERVER, file, spec, 'png')).blob;
  const same2 = (await WmServer.watermark(SERVER, file, spec, 'png')).blob;
  const a = (await WmServer.watermark(SERVER, file, spec, 'png', { derive_seed_from: 'a.png' })).blob;
  const b = (await WmServer.watermark(SERVER, file, spec, 'png', { derive_seed_from: 'b.png' })).blob;
  const bytes = async (x) => Buffer.from(await x.arrayBuffer()).toString('base64');
  assert.equal(await bytes(same1), await bytes(same2), 'same seed, same file -> identical output');
  assert.notEqual(await bytes(a), await bytes(b), 'different file names -> different jitter');
  assert.notEqual(await bytes(a), await bytes(same1));
});
