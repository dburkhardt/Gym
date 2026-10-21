# SPDX-License-Identifier: Apache-2.0
"""Explicit headless image-observation harness variant; no resumable registry."""

import base64
import copy
import hashlib
import importlib.metadata
import io
import json
import re
import shlex
import textwrap
import uuid
from pathlib import Path, PurePosixPath

from pydantic import Field

from nemo_gym.provider_dispatch import configuration_digest
from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import (
    Terminus2NemoGym,
    Terminus2NemoGymOptions,
)


PREFIX = "[[gym-image:"
TOKEN = re.compile(r"\[\[gym-image:([0-9a-f]{32})\]\]")


class ImageToolInputError(ValueError):
    """A model-supplied image command can be corrected within the same run."""


def reader_script(root, relative, maximum):
    """Open every component without following links; read one bounded descriptor."""
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(x in (".", "..") for x in relative.split("/")):
        raise ImageToolInputError("Image path must be a contained relative file")
    root_parts = PurePosixPath(root).parts[1:]
    body = f"""import os,stat,base64
fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
try:
 for part in {list(root_parts + parts[:-1])!r}:
  child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
  os.close(fd);fd=child
 image=os.open({parts[-1]!r},os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
 try:
  info=os.fstat(image)
  if not stat.S_ISREG(info.st_mode) or info.st_size>{maximum}: raise ValueError('Image is not a bounded regular file')
  data=b''
  while len(data)<={maximum}:
   chunk=os.read(image,min(65536,{maximum}+1-len(data)))
   if not chunk: break
   data+=chunk
  if len(data)>{maximum}: raise ValueError('Image exceeds byte limit')
  print(base64.b64encode(data).decode())
 finally: os.close(image)
finally: os.close(fd)
"""
    return (
        "import errno,sys\ntry:\n"
        + textwrap.indent(body, " ")
        + "\nexcept (ValueError,OSError) as error:\n if isinstance(error,OSError) and error.errno not in (errno.ENOENT,errno.EACCES,errno.ELOOP,errno.ENOTDIR,errno.EISDIR): raise\n print('GYM_IMAGE_INPUT_ERROR:' + type(error).__name__)\n sys.exit(2)\n"
    )


class ImageRegistry:
    def __init__(self, directory, maximum_bytes=4_000_000, maximum_pixels=16_000_000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.maximum_bytes = maximum_bytes
        self.maximum_pixels = maximum_pixels
        self.entries = {}

    def register(self, data, user_turn=0):
        from PIL import Image

        if not data or len(data) > self.maximum_bytes:
            raise ImageToolInputError("Image exceeds byte limit or is empty")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"} or image.width * image.height > self.maximum_pixels:
                    raise ImageToolInputError("Unsupported format or image dimensions")
                mime = Image.MIME[image.format]
                size = list(image.size)
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ImageToolInputError("Invalid image format or dimensions") from error
        self.directory.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        path = self.directory / (identity + ".image")
        with path.open("xb") as stream:
            stream.write(data)
        self.entries[identity] = {
            "user_turn": user_turn,
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mime": mime,
            "size": size,
        }
        (self.directory / "observations.json").write_text(json.dumps(self.entries, indent=2))
        return PREFIX + identity + "]]"

    def project(self, messages):
        result = copy.deepcopy(messages)
        for index, message in enumerate(result):
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, str) or PREFIX not in content:
                continue
            matches = list(TOKEN.finditer(content))
            if not matches or PREFIX in TOKEN.sub("", content):
                raise ValueError("Malformed image observation token")
            parts = [{"type": "text", "text": content}]
            for match in matches:
                entry = self.entries.get(match.group(1))
                if entry is None:
                    raise ValueError("Missing image observation registry; restore unsupported")
                if index != entry["user_turn"] * 2:
                    raise ValueError("Image token is not the registered observation turn")
                path = Path(entry["path"])
                if path.is_symlink():
                    raise ValueError("Image artifact is a symlink")
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError("Image artifact changed")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:" + entry["mime"] + ";base64," + base64.b64encode(data).decode(),
                            "detail": "auto",
                        },
                    }
                )
            message["content"] = parts
        return result


class Terminus2ImagesNemoGymOptions(Terminus2NemoGymOptions):
    image_root: str
    image_reader_python: str = "/opt/corebench-harness/bin/python3"
    image_max_bytes: int = Field(default=4_000_000, gt=0, le=4_000_000)
    image_max_pixels: int = Field(default=16_000_000, gt=0, le=16_000_000)


class Terminus2ImagesNemoGym(Terminus2NemoGym):
    options_model = Terminus2ImagesNemoGymOptions

    def __init__(
        self,
        *args,
        image_root,
        image_reader_python="/opt/corebench-harness/bin/python3",
        image_max_bytes=4_000_000,
        image_max_pixels=16_000_000,
        **kwargs,
    ):
        if kwargs.get("gym_provider_dispatch_guard") is not True:
            raise ValueError("Image variant requires the authoritative dispatch guard")
        if kwargs.get("llm") is not None or kwargs.get("enable_summarize", True) is not False:
            raise ValueError("Image variant requires managed LLM and summarization disabled")
        if not image_root.startswith("/workspace/") or ".." in image_root.split("/"):
            raise ValueError("Image root must be an explicit workspace directory")
        super().__init__(
            *args,
            image_root=image_root,
            image_reader_python=image_reader_python,
            image_max_bytes=image_max_bytes,
            image_max_pixels=image_max_pixels,
            **kwargs,
        )
        self._images = ImageRegistry(self.logs_dir / "image-observations", image_max_bytes, image_max_pixels)
        self._nemo_gym_llm._project_messages = self._images.project
        self._nemo_gym_llm._dispatch_config = configuration_digest(
            {
                "base": self._nemo_gym_llm._dispatch_config,
                "variant": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "root": image_root,
                "reader": image_reader_python,
                "bytes": image_max_bytes,
                "pixels": image_max_pixels,
                "pillow": importlib.metadata.version("Pillow"),
            }
        )
        self._image_environment = None
        self._image_run_started = False

    async def run(self, instruction, environment, context):
        if self._image_run_started:
            raise ValueError("Image harness reuse/restore is unsupported")
        self._image_run_started = True
        self._image_environment = environment
        root_parts = list(PurePosixPath(self.options.image_root).parts[1:])
        check = (
            "import os\nfd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)\ntry:\n for part in "
            + repr(root_parts)
            + ":\n  child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)\n  os.close(fd);fd=child\n print('image-root-ok')\nfinally: os.close(fd)"
        )
        checked = await environment.exec(
            shlex.join([self.options.image_reader_python, "-I", "-c", check]), timeout_sec=30
        )
        if checked.return_code != 0 or (checked.stdout or "").strip() != "image-root-ok":
            raise RuntimeError("Configured image root or reader is unavailable")
        instruction += (
            "\nHarness tool: to view a generated PNG/JPEG, issue a sole command gym_view_image <relative-path> under "
            + self.options.image_root
            + ". It returns an image observation. Do not mix this tool with other commands."
        )
        return await super().run(instruction, environment, context)

    async def _execute_commands(self, commands, session):
        requested = [c for c in commands if c.keystrokes.strip().startswith("gym_view_image")]
        if not requested:
            return await super()._execute_commands(commands, session)
        if self._image_environment is None:
            raise RuntimeError("Image environment is not active")
        try:
            return await self._view_image(commands)
        except ImageToolInputError as error:
            return False, "Image tool error: " + str(error)[:512]

    async def _view_image(self, commands):
        if len(commands) != 1:
            raise ImageToolInputError("Image view requires a sole command")
        try:
            words = shlex.split(commands[0].keystrokes.strip())
        except ValueError as error:
            raise ImageToolInputError("Invalid image-view command quoting") from error
        if len(words) != 2 or words[0] != "gym_view_image":
            raise ImageToolInputError("Invalid image-view command")
        script = reader_script(self.options.image_root, words[1], self.options.image_max_bytes)
        result = await self._image_environment.exec(
            shlex.join([self.options.image_reader_python, "-I", "-c", script]), timeout_sec=30
        )
        if result.return_code == 2 and (result.stdout or "").startswith("GYM_IMAGE_INPUT_ERROR:"):
            raise ImageToolInputError("Image path is missing, unreadable, unsafe, or exceeds limits")
        if result.return_code != 0:
            raise RuntimeError("Image reader infrastructure failed")
        data = base64.b64decode((result.stdout or "").strip(), validate=True)
        return False, self._images.register(data, self._nemo_gym_llm._dispatch_ordinal + 1)
