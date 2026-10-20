# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-model native manifest/socket/MCP control, with the Qt loop on its main thread."""
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import napari
from npe2 import PluginManager
from qtpy.QtCore import QTimer


def main():
    source, output = [Path(p).resolve() for p in sys.argv[1:3]]
    sys.path.insert(0, str(source / 'src/napari_socket'))
    sys.path.insert(0, str(source / 'src/napari_mcp'))
    PluginManager.instance().register(str(source / 'src/napari_socket/napari_socket/napari.yaml'))
    viewer = napari.Viewer(show=True)
    from napari_socket._widget import NapariSocketWidget
    widget = NapariSocketWidget(viewer)
    done = threading.Event()
    result = {'llm_calls': 0, 'controls': [], 'status': 'running'}

    def controls():
        try:
            from napari_manager import NapariManager
            from napari_mcp_server import build_mcp
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            import os
            manager = NapariManager(timeout=60)
            async def run():
                params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name('mcp_bootstrap.py')), str(source/'src/napari_mcp/napari_mcp_server.py')], env=dict(os.environ))
                async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    result['tool_names'] = [t.name for t in tools.tools]
                    sequence = [('clear_all_layers', {})]
                    if os.environ.get('SCIVIS_FULL_CONTROL') == '1':
                        for channel, color in [(0, 'red'), (1, 'green')]:
                            name = f'dataset_002_ch{channel}'
                            sequence += [('open_file', {'file_path': f'/workspace/data/dataset_002/{name}.tif'}), ('list_layers', {}), ('set_colormap', {'layer_name': name, 'colormap': color}), ('set_blending', {'layer_name': name, 'blending': 'additive'})]
                        sequence += [('set_view_mode', {'mode':'3D'}), ('set_timestep', {'timestep':14})]
                    sequence += [('list_layers', {}), ('get_screenshot', {})]
                    for name, args in sequence:
                        response = await session.call_tool(name, args)
                        serialized = response.model_dump(mode='json')
                        # Retain image bytes separately, not a large JSON screenshot string.
                        for item in serialized.get('content', []):
                            if item.get('type') == 'image':
                                import base64
                                image = base64.b64decode(item.pop('data'))
                                (output.parent / 'mcp-screenshot.png').write_bytes(image)
                                item['image_bytes'] = len(image)
                        result['controls'].append({'tool': name, 'response': serialized})
                        assert not response.isError, serialized
                        assert not any(getattr(c, 'text', '').startswith('❌') for c in response.content)
                    if os.environ.get('SCIVIS_ISOLATION_CONTROL') == '1':
                        denied = ['/workspace/GS/dataset_002_1.png', '/var/run/docker.sock', '/home/ubuntu/projects/skill-eval/.secrets/inference-nvidia.key', '/source/benchmark/eval_cases/napari/eval_visualization_tasks.yaml']
                        for path in denied:
                            response = await session.call_tool('open_file', {'file_path': path})
                            text = '\n'.join(getattr(c, 'text', '') for c in response.content)
                            assert 'File not found' in text, (path, text)
                            result['controls'].append({'denied_path': path, 'response': text})
                        response = await session.call_tool('save_text', {'filename': '/workspace/write-control.txt', 'content': 'native writable path control'})
                        assert Path('/workspace/write-control.txt').read_text() == 'native writable path control'
                        result['controls'].append({'writable_workspace': True})
                    if os.environ.get('SCIVIS_FILESYSTEM_MCP'):
                        fs = StdioServerParameters(command=os.environ['SCIVIS_NODE'], args=[os.environ['SCIVIS_FILESYSTEM_MCP'], '/workspace'], env=dict(os.environ))
                        async with stdio_client(fs) as (fr, fw), ClientSession(fr, fw) as filesystem:
                            await filesystem.initialize()
                            tools = await filesystem.list_tools()
                            result['filesystem_tool_names'] = [t.name for t in tools.tools]
                            for path in ['/workspace/GS/dataset_002_1.png', '/source/benchmark/eval_cases/napari/eval_visualization_tasks.yaml', '/home/ubuntu/projects/skill-eval/.secrets/inference-nvidia.key', '/var/run/docker.sock']:
                                reply = await filesystem.call_tool('read_text_file', {'path': path})
                                assert reply.isError, reply
                                result['controls'].append({'filesystem_denied':path,'response':reply.model_dump(mode='json')})
                            reply = await filesystem.call_tool('read_text_file', {'path':'/workspace/write-control.txt'})
                            assert not reply.isError
                            assert any('native writable path control' in getattr(c,'text','') for c in reply.content)
                            result['controls'].append({'filesystem_public_output_read':True})
                    import resource
                    result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    result['full_assets'] = os.environ.get('SCIVIS_FULL_CONTROL') == '1'
                    result['layers'] = [{'name':l.name,'shape':list(l.data.shape),'dtype':str(l.data.dtype),'colormap':l.colormap.name,'blending':str(l.blending)} for l in viewer.layers]
                    result['current_step'] = list(viewer.dims.current_step)
                    result['ndisplay'] = viewer.dims.ndisplay
                    if result['full_assets']:
                        assert len(viewer.layers)==2
                        assert all(tuple(l.data.shape)==(35,100,320,525) for l in viewer.layers)
                        assert viewer.dims.current_step[0]==14 and viewer.dims.ndisplay==3
                    result['status'] = 'passed'
            asyncio.run(run())
        except Exception as exc:
            result['status'] = 'failed'
            result['error'] = repr(exc)
            import traceback
            result['traceback'] = traceback.format_exc()
        finally:
            output.write_text(json.dumps(result, indent=2))
            done.set()

    threading.Thread(target=controls, daemon=True).start()
    timer = QTimer()
    timer.timeout.connect(lambda: viewer.close() if done.is_set() else None)
    timer.start(100)
    napari.run()
    widget._stop()
    if result['status'] != 'passed':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
