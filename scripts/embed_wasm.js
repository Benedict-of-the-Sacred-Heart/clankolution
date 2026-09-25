const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const repoRoot = path.resolve(__dirname, '..');
const wasmDir = path.join(repoRoot, 'wasm', 'clank_wasm');
const wasmPath = path.join(wasmDir, 'target', 'wasm32-unknown-unknown', 'release', 'clank_wasm.wasm');
const htmlPath = path.join(repoRoot, 'clankolution.html');

console.log('Compiling Rust to wasm32-unknown-unknown...');
execSync('RUSTFLAGS="-C link-arg=--allow-undefined" cargo build --target wasm32-unknown-unknown --release --manifest-path Cargo.toml', {
  cwd: wasmDir,
  stdio: 'inherit'
});

const wasmBuffer = fs.readFileSync(wasmPath);
const base64Wasm = wasmBuffer.toString('base64');
console.log(`Compiled Wasm size: ${wasmBuffer.length} bytes (Base64: ${base64Wasm.length} chars)`);

let html = fs.readFileSync(htmlPath, 'utf8');
const needle = /const WASM_B64 = ".*?";/;
const replacement = `const WASM_B64 = "${base64Wasm}";`;

if (needle.test(html)) {
  html = html.replace(needle, replacement);
} else {
  // If not yet present, insert right before the main script logic or at start of script
  html = html.replace('<script>', `<script>\n      ${replacement}\n`);
}

fs.writeFileSync(htmlPath, html, 'utf8');
console.log('Successfully embedded Wasm binary into clankolution.html!');
