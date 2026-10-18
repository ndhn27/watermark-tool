// Shared helpers: pull the page's scripts out of web/index.html.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const INDEX = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../web/index.html');
export const html = () => readFileSync(INDEX, 'utf8');

/** The DOM-free client: the <script id="wm-server"> block. */
export function clientSource() {
  const m = html().match(/<script id="wm-server">([\s\S]*?)<\/script>/);
  if (!m) throw new Error('web/index.html has no <script id="wm-server"> block');
  return m[1];
}

export const SERVER = process.env.WM_TEST_SERVER || '';
