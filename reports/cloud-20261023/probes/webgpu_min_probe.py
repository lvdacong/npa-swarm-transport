import json, os, threading, time
from http.server import HTTPServer,BaseHTTPRequestHandler
from playwright.sync_api import sync_playwright
from pathlib import Path
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  body=b'<html><canvas id="c" width="256" height="256"></canvas></html>'
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def log_message(self,*a):pass
s=HTTPServer(('127.0.0.1',0),H);threading.Thread(target=s.serve_forever,daemon=True).start()
js='''async () => {
 const r = window.result = {frames:0, errors:[]};
 const gpu=window.savedGPU=navigator.gpu;
 const adapter=window.savedAdapter=await gpu.requestAdapter();
 if(!adapter)return {...r,noAdapter:true};
 r.adapter={...adapter.info,description:adapter.info.description};
 const device=window.savedDevice=await adapter.requestDevice();
 device.lost.then(i=>r.errors.push({lost:i.reason,message:i.message}));
 device.addEventListener('uncapturederror',e=>r.errors.push(e.error.message));
 const canvas=document.getElementById('c'),ctx=canvas.getContext('webgpu'),format=gpu.getPreferredCanvasFormat();
 ctx.configure({device,format});
 const module=device.createShaderModule({code:`
 @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
  let p=array<vec2f,3>(vec2f(0.,.8),vec2f(-.8,-.8),vec2f(.8,-.8));return vec4f(p[i],0.,1.);
 }
 @fragment fn fs()->@location(0)vec4f {return vec4f(.2,.8,.4,1.);}`});
 const pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format}]}});
 for(let i=0;i<90;i++) {
  if(r.errors.length)break;
  const enc=device.createCommandEncoder(),pass=enc.beginRenderPass({colorAttachments:[{view:ctx.getCurrentTexture().createView(),loadOp:'clear',clearValue:[0,0,0,1],storeOp:'store'}]});
  pass.setPipeline(pipeline);pass.draw(3);pass.end();device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();r.frames++;
  await new Promise(requestAnimationFrame);
 }
 return r;
}'''
variants=[('baseline',['--use-angle=swiftshader','--enable-unsafe-swiftshader']),('explicit-vulkan',['--use-angle=swiftshader','--enable-unsafe-swiftshader','--enable-features=Vulkan','--use-vulkan=swiftshader','--disable-vulkan-surface']),('angle-vulkan',['--use-angle=vulkan','--enable-features=Vulkan','--use-vulkan=swiftshader','--disable-vulkan-surface'])]
results=[]
with sync_playwright() as p:
 for name,extra in variants:
  row={'variant':name,'args':extra};t=time.monotonic()
  try:
   b=p.chromium.launch(headless=True,executable_path=os.environ['SWARM_BROWSER_EXECUTABLE'],args=['--enable-unsafe-webgpu','--no-sandbox',*extra])
   page=b.new_page();row['console']=[];page.on('console',lambda m:row['console'].append(m.text));page.goto(f'http://127.0.0.1:{s.server_port}');page.set_default_timeout(20000)
   row['result']=page.evaluate(js)
   page.screenshot(path=f'/workspace/scratch/d5c88b93317d/browser-runtime/triangle-{name}.png')
   b.close()
  except Exception as e: row['error']=str(e)
  row['seconds']=round(time.monotonic()-t,3);results.append(row);print(json.dumps(row),flush=True)
Path('/workspace/scratch/d5c88b93317d/npa-swarm-transport/reports/cloud-20261023/webgpu-minimal-probe.json').write_text(json.dumps(results,indent=2))
s.shutdown()
