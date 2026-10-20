# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent native viewer; no inference, no task/reference initialization."""
import json,sys
from pathlib import Path
import napari
from npe2 import PluginManager
source=Path('/source')
sys.path.insert(0,str(source/'src/napari_socket'))
PluginManager.instance().register(str(source/'src/napari_socket/napari_socket/napari.yaml'))
viewer=napari.Viewer(show=True)
from napari_socket._widget import NapariSocketWidget
widget=NapariSocketWidget(viewer)
Path('/workspace/viewer-ready.json').write_text(json.dumps({'native_viewer':True,'preloaded_layers':0,'reference_access':False}))
napari.run()
