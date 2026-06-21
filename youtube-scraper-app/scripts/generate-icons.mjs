// Génère les icônes PNG de la PWA sans dépendance externe.
// Dessine un carré rouge arrondi (style YouTube) avec un triangle "play" blanc.

import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUBLIC = resolve(__dirname, "../public");
mkdirSync(PUBLIC, { recursive: true });

// CRC32 (table-driven).
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, "ascii");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}

function encodePng(size, pixels) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  // raw scanlines : 1 octet filtre (0) par ligne + RGBA
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    pixels.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  return Buffer.concat([
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// Rouge "brand" + triangle play blanc.
const RED = [0xff, 0x00, 0x33];
const WHITE = [0xff, 0xff, 0xff];

function drawIcon(size, { maskable = false } = {}) {
  const px = Buffer.alloc(size * size * 4);
  const radius = maskable ? 0 : size * 0.22;
  // Zone de sécurité maskable : le contenu utile tient dans ~80%.
  const inset = maskable ? size * 0.1 : 0;
  const triCx = size / 2;
  const triHalf = (size - inset * 2) * 0.16;
  const triTop = size / 2 - triHalf;
  const triBot = size / 2 + triHalf;
  const triLeft = size / 2 - triHalf * 0.8;
  const triRight = size / 2 + triHalf * 1.1;

  function inRoundedSquare(x, y) {
    if (maskable) return true;
    const r = radius;
    const minX = r,
      maxX = size - r,
      minY = r,
      maxY = size - r;
    const cx = Math.min(Math.max(x, minX), maxX);
    const cy = Math.min(Math.max(y, minY), maxY);
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
  }

  function inTriangle(x, y) {
    if (y < triTop || y > triBot || x < triLeft) return false;
    // arête droite inclinée (sommet à droite)
    const t = (y - triTop) / (triBot - triTop); // 0..1
    const rightEdge = triLeft + (triRight - triLeft) * (1 - Math.abs(t - 0.5) * 2);
    return x <= rightEdge;
  }

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      if (!inRoundedSquare(x + 0.5, y + 0.5)) {
        px[i + 3] = 0; // transparent
        continue;
      }
      const [r, g, b] = inTriangle(x + 0.5, y + 0.5) ? WHITE : RED;
      px[i] = r;
      px[i + 1] = g;
      px[i + 2] = b;
      px[i + 3] = 255;
    }
  }
  return px;
}

function write(name, size, opts) {
  const png = encodePng(size, drawIcon(size, opts));
  writeFileSync(resolve(PUBLIC, name), png);
  console.log(`✓ ${name} (${png.length} octets)`);
}

write("icon-192x192.png", 192);
write("icon-512x512.png", 512);
write("icon-maskable-512x512.png", 512, { maskable: true });
write("favicon.ico", 64); // PNG renommé .ico (accepté par les navigateurs modernes)
console.log("Icônes générées.");
