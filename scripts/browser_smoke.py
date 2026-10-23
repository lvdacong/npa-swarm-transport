"""Small actual WebGPU check. No browser-mode menu and no Canvas substitution."""
import argparse
import json
from pathlib import Path
import threading
import time
import torch
from playwright.sync_api import sync_playwright
from npa_swarm.config import EnvConfig
from npa_swarm.server import create_server


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='reports/browser')
    p.add_argument('--executable')
    p.add_argument('--hardware',action='store_true',help='do not enable software graphics flags')
    p.add_argument('--headed',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    server=create_server(EnvConfig(),untrained=True,stochastic=True,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    result={'scope':'untrained 900-particle WebGPU integration, not learned transport','physical_gpu_validated':False}
    try:
        with sync_playwright() as pw:
            flags=[] if args.hardware else ['--enable-unsafe-webgpu','--use-angle=swiftshader','--use-vulkan=swiftshader','--enable-features=Vulkan']
            browser=pw.chromium.launch(executable_path=args.executable,headless=not args.headed,args=flags)
            page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_address[1]}')
            page.wait_for_function('window.swarmDiagnostics?.().particles===900 && window.swarmDiagnostics().renderedFrames>0',timeout=20000)
            result['initial']=page.evaluate('swarmDiagnostics()')
            assert result['initial']['backend']=='WebGPU'
            assert page.locator('select,h1').count()==0
            assert page.locator('#diagnostics').is_hidden()
            page.screenshot(path=str(out/'viewer.png'))
            from PIL import Image
            import numpy as np
            a=np.array(Image.open(out/'viewer.png').convert('RGB'))
            teal=(a[...,1]>120)&(a[...,1]>a[...,0]*1.2)&(a[...,2]>a[...,0]*1.1)
            orange=(a[...,0]>150)&(a[...,0]>a[...,1]*1.1)&(a[...,1]>a[...,2]*1.15)
            result['scene_pixels']={'teal':int(teal.sum()),'orange':int(orange.sum())}
            assert teal.sum()>300 and orange.sum()>300
            page.click('#play');page.wait_for_function('swarmDiagnostics().step>=15',timeout=20000)
            page.click('#play');page.wait_for_function('swarmDiagnostics().paused')
            page.wait_for_timeout(250)
            first=page.evaluate('swarmDiagnostics()');page.wait_for_timeout(350);second=page.evaluate('swarmDiagnostics()')
            assert second['step']==first['step'] and second['uploads']==first['uploads']
            assert second['renderedFrames']==first['renderedFrames']
            result['paused']=second
            page.mouse.move(720,450);page.mouse.wheel(0,-400)
            page.wait_for_function('swarmDiagnostics().camera.zoom>1.2')
            page.click('#fit');page.wait_for_function('swarmDiagnostics().camera.zoom===1')
            # Select a real particle using the displayed positions, not a fake UI hook.
            payload=page.evaluate('async()=>Array.from(new Float32Array(await(await fetch("/api/snapshot")).arrayBuffer(),128,2))')
            scale=900/(2*10*1.08)
            page.mouse.click(720+payload[0]*scale,450-payload[1]*scale)
            page.wait_for_function('swarmDiagnostics().selected>=0')
            result['selected']=page.evaluate('swarmDiagnostics().selected')
            page.click('#reset');page.wait_for_function('swarmDiagnostics().step===0')
            page.click('#speed');page.wait_for_function('document.querySelector("#speed").textContent==="8×"')
            page.click('#speed')  # 4 -> 8 -> unpaced, actual rate measured separately
            page.wait_for_function('document.querySelector("#speed").textContent==="∞"')
            page.click('#play');page.wait_for_function('swarmDiagnostics().step>=20',timeout=20000)
            page.click('#play');page.wait_for_function('swarmDiagnostics().paused')
            page.click('#details');assert page.locator('#diagnostics').is_visible()
            result['final']=page.evaluate('swarmDiagnostics()');assert not result['final']['errors'] and not errors
            page.click('#details')
            page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.screenshot(path=str(out/'viewer-mobile.png'))
            result['page_errors']=errors
            adapter=result['initial']['adapter'].lower()
            result['software_adapter']=any(s in adapter for s in ('swiftshader','llvmpipe','lavapipe','software'))
            if args.hardware:assert not result['software_adapter'],'Hardware requested but software adapter observed'
            result['physical_gpu_validated']=bool(args.hardware and not result['software_adapter'])
            # Explicit unsupported-device path must report, not silently select Canvas.
            unsupported=browser.new_page()
            unsupported.add_init_script('Object.defineProperty(navigator,"gpu",{value:undefined})')
            unsupported.goto(f'http://127.0.0.1:{server.server_address[1]}')
            unsupported.wait_for_selector('#scene-error',state='visible')
            assert unsupported.evaluate('swarmDiagnostics().backend') is None
            result['unsupported_webgpu_error']=unsupported.locator('#scene-error').inner_text()
            result['status']='PASS';browser.close()
    except Exception as e:
        result['status']='FAIL';result['error']=repr(e)
        raise
    finally:
        (out/'browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        server.shutdown();server.server_close();thread.join(5)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
