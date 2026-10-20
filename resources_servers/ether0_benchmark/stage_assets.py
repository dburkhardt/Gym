"""Restore hash-pinned creator BloomFilter files omitted from its built wheel."""
import hashlib
import importlib.util
import urllib.request
from pathlib import Path
PIN = 'b77e0d8eb06f62c8528da72bff22d4c7fee11371'
HASHES = {
    'fingerprints.bloom': 'f9267f986c512f26131eb0219170c12e07ded9c1574c75233da5e3d5f3f00278',
    'rings.bloom': 'b1b81b5dc5e59d2068742aafb0c217067e951260f56527998e902d37ac177a25',
}
if __name__ == '__main__':
    package = Path(importlib.util.find_spec('ether0').origin).parent
    for name, digest in HASHES.items():
        target = package / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        data = urllib.request.urlopen(f'https://raw.githubusercontent.com/Future-House/ether0/{PIN}/src/ether0/{name}', timeout=90).read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Pinned asset checksum mismatch: {name}')
        target.write_bytes(data)
        print(f'Staged verified {name}')
