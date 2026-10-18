// web/index.html booted in jsdom against a REAL cli/serve.py: does server mode appear, build its
// controls from the schema, follow the schema's dependency rules, and stay out of the way when
// there is no server? (jsdom can't decode images, so the actual download click path is covered
// by wm-server.test.mjs at the client-module level, not here.) Skips itself without jsdom.
import test from 'node:test';
import assert from 'node:assert/strict';
import { html, SERVER } from './harness.mjs';

let JSDOM;
try { ({ JSDOM } = await import('jsdom')); } catch { /* not installed */ }
const skip = JSDOM ? false : 'jsdom not installed (npm install in tests/js)';

async function boot(url) {
  const errors = [];
  const dom = new JSDOM(html(), {
    url, runScripts: 'dangerously', pretendToBeVisual: true,
    beforeParse(window) {
      window.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: () => () => ({ width: 0 }), set: () => true });
      window.fetch = (u, o) => fetch(new URL(u, window.location.href), o);
      window.createImageBitmap = async () => ({ width: 1, height: 1 });
      window.addEventListener('error', (e) => errors.push(e.error || e.message));
    },
  });
  await new Promise((r) => setTimeout(r, 400));   // let detect() resolve
  return { dom, doc: dom.window.document, errors };
}

test('with a server: panel appears, controls come from the schema, page boots cleanly', { skip }, async () => {
  const { doc, errors } = await boot(SERVER + '/');
  assert.deepEqual(errors, []);
  assert.equal(doc.getElementById('server-panel').hidden, false);
  const rows = [...doc.querySelectorAll('#srv-options .srv-row')].map((r) => r.querySelector('label').htmlFor);
  for (const id of ['srv-blend', 'srv-ink', 'srv-invisible', 'srv-invisible_mode', 'srv-ss_key', 'srv-author', 'srv-adaptive', 'srv-seed']) {
    assert.ok(rows.includes(id), `${id} should be rendered`);
  }
  assert.ok(!rows.includes('srv-opacity') && !rows.includes('srv-spacing') && !rows.includes('srv-font_path'));
  assert.equal(doc.getElementById('srv-options').hidden, true, 'off until the user opts in');
});

test('dependency rules from the schema drive which rows show', { skip }, async () => {
  const { doc, dom } = await boot(SERVER + '/');
  const row = (n) => doc.getElementById('srv-' + n).closest('.srv-row');
  assert.equal(row('ss_key').hidden, true);
  assert.equal(row('invisible_mode').hidden, true);
  const inv = doc.getElementById('srv-invisible');
  inv.checked = true; inv.dispatchEvent(new dom.window.Event('change'));
  assert.equal(row('invisible_mode').hidden, false);
  assert.equal(row('ss_key').hidden, true, 'still lsb');
  const mode = doc.getElementById('srv-invisible_mode');
  mode.value = 'spread'; mode.dispatchEvent(new dom.window.Event('change'));
  assert.equal(row('ss_key').hidden, false);
  assert.equal(row('ss_alpha').hidden, false);
});

test('opting in shows the options, locks the colour swatches and is honest in the footnote', { skip }, async () => {
  const { doc, dom } = await boot(SERVER + '/');
  const before = doc.getElementById('footnote').textContent;
  assert.match(before, /never uploaded/i);
  const t = doc.getElementById('srv-toggle');
  t.checked = true; t.dispatchEvent(new dom.window.Event('change'));
  assert.equal(doc.getElementById('srv-options').hidden, false);
  assert.equal(doc.getElementById('srv-preview-btn').hidden, false);
  assert.ok(doc.querySelector('.swatches').classList.contains('locked'));
  assert.match(doc.getElementById('footnote').textContent, /sent to .* to be watermarked/i);
  t.checked = false; t.dispatchEvent(new dom.window.Event('change'));
  assert.equal(doc.getElementById('footnote').textContent, before, 'turning it off restores the local-only claim');
  assert.equal(doc.getElementById('srv-options').hidden, true);
});

test('without a server the page is exactly the old static tool', { skip }, async () => {
  const { doc, errors } = await boot('http://127.0.0.1:9/');
  assert.deepEqual(errors, []);
  assert.equal(doc.getElementById('server-panel').hidden, true);
  assert.equal(doc.querySelectorAll('#srv-options .srv-row').length, 0);
  assert.match(doc.getElementById('footnote').textContent, /never uploaded/i);
});
