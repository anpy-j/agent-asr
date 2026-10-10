"""Build wheel/sdist and hash only artifacts produced for this version."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from agent_asr import API_VERSION, __version__  # noqa: E402

subprocess.run([sys.executable, "-m", "build"], cwd=root, check=True)
dist = root / "dist"
assets = [dist / f"agent_asr-{__version__}-py3-none-any.whl", dist / f"agent_asr-{__version__}.tar.gz"]
installer = dist / "install_component.ps1"
shutil.copyfile(root / "scripts" / "install_component.ps1", installer)
assets.append(installer)
manifest = {
    "component": "agent-asr",
    "version": __version__,
    "api_version": API_VERSION,
    "requires_python": ">=3.11",
    "assets": [],
}
for asset in assets:
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest["assets"].append({"name": asset.name, "sha256": digest, "bytes": asset.stat().st_size})
(dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
assets.append(dist / "manifest.json")
(dist / "SHA256SUMS.txt").write_text(
    "".join(f"{hashlib.sha256(asset.read_bytes()).hexdigest()}  {asset.name}\n" for asset in assets),
    encoding="utf-8",
)
print(f"Built {__version__}; manifest and SHA256SUMS.txt ready")
