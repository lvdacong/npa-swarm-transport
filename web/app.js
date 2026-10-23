const $ = id => document.getElementById(id);
// Retain the exact GPU instance used to request the adapter through async setup.
const webgpu = navigator.gpu;
const params = new URLSearchParams(location.search);
const state = {frame:null, previous:null, renderer:null, backend:'', selected:-1,
  camera:{x:0,y:0,zoom:1}, status:null, paused:true, received:0, interval:100,
  dirty:true, visible:!document.hidden, snapshots:0, draws:0, renderedFrames:0, lastMeter:performance.now(),
  errors:[], request:null, stopped:false, requestGeneration:0, initialDistance:null};
const fmt = (n,d=2) => Number.isFinite(n) ? n.toFixed(d) : '—';
const error = text => { state.errors.push(String(text)); $('scene-error').textContent=text; $('scene-error').hidden=false; };
function parseSnapshot(buffer) {
  if(buffer.byteLength<128) throw new Error('快照头不完整');
  const u = new Uint32Array(buffer,0,8), f = new Float32Array(buffer,32,24);
  if(u[0]!==0x3153504e || u[1]!==1 || u[3]!==128 || u[2]<1 || buffer.byteLength!==128+u[2]*32)
    throw new Error('快照格式不匹配');
  return {buffer,n:u[2],sequence:u[4],flags:u[5],step:u[6],revision:u[7],f,
    data:new Float32Array(buffer,128,u[2]*8)};
}
function transform(canvas) {
  const aspect=canvas.width/canvas.height;
  const half=(state.frame?.f[7]||10)*1.08/(state.camera.zoom*Math.min(aspect,1));
  return {aspect,half,sx:1/(half*aspect),sy:1/half};
}
function resizeCanvas(canvas) {
  const dpr=Math.min(devicePixelRatio||1,2), r=canvas.getBoundingClientRect();
  const w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));
  if(canvas.width!==w||canvas.height!==h) {canvas.width=w;canvas.height=h;state.dirty=true;}
}

class CanvasRenderer {
  constructor(canvas) {this.canvas=canvas;this.ctx=canvas.getContext('2d',{alpha:false});this.frame=null;this.previous=null;this.name='Canvas 2D';this.adapter='CPU 绘制回退';}
  setFrame(frame) {this.previous=this.frame?.n===frame.n?this.frame:frame;this.frame=frame;}
  render(blend) {
    const {ctx:c,canvas,frame:f}=this;if(!f)return;
    resizeCanvas(canvas);const t=transform(canvas),w=canvas.width,h=canvas.height;
    const scale=h/(2*t.half),cx=w/2-state.camera.x*scale,cy=h/2+state.camera.y*scale;
    c.fillStyle='#0d1b1e';c.fillRect(0,0,w,h);
    const xy=(x,y)=>[cx+x*scale,cy-y*scale];
    c.strokeStyle='#193231';c.lineWidth=1;c.beginPath();
    const a=f.f[7];for(let k=Math.ceil(-a);k<=a;k++){let p=xy(k,-a),q=xy(k,a);c.moveTo(...p);c.lineTo(...q);p=xy(-a,k);q=xy(a,k);c.moveTo(...p);c.lineTo(...q);}c.stroke();
    c.strokeStyle='#34544e';const tl=xy(-a,a);c.strokeRect(tl[0],tl[1],2*a*scale,2*a*scale);
    const disk=(x,y,r,fill,stroke)=>{const p=xy(x,y);c.beginPath();c.arc(p[0],p[1],r*scale,0,Math.PI*2);if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.lineWidth=1.5;c.stroke();}};
    disk(f.f[2],f.f[3],f.f[5],'#15372a','#69d99b');
    const old=this.previous?.f||f.f;
    disk(old[0]+(f.f[0]-old[0])*blend,old[1]+(f.f[1]-old[1])*blend,f.f[4],'#dca66b','#f9cf96');
    const data=f.data,prev=this.previous?.data||data;
    if(state.selected>=0&&state.selected<f.n){const k=state.selected*8;disk(data[k],data[k+1],f.f[8],'#18454b44','#4f96a3');disk(data[k],data[k+1],f.f[9],null,'#2b5257');}
    c.fillStyle='#87dfc3';c.beginPath();
    const r=Math.max(.8,f.f[6]*scale);
    for(let i=0;i<f.n;i++){const k=i*8,x=prev[k]+(data[k]-prev[k])*blend,y=prev[k+1]+(data[k+1]-prev[k+1])*blend,p=xy(x,y);c.moveTo(p[0]+r,p[1]);c.arc(p[0],p[1],r,0,Math.PI*2);}c.fill();
    if(state.selected>=0&&state.selected<f.n){const k=state.selected*8;disk(data[k],data[k+1],Math.max(f.f[6]*2,2/scale),'#f1f7ed',null);}
  }
  dispose(){}
}

const shader = `
struct Particle { p:vec2f, v:vec2f, signal:vec4f };
struct Uniforms { view:vec4f, scene:vec4f, radii:vec4f, pick:vec4f, misc:vec4f, camera:vec4f };
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> previous:array<Particle>;
@group(0) @binding(2) var<storage,read> current:array<Particle>;
struct VOut { @builtin(position) pos:vec4f, @location(0) local:vec2f, @location(1) color:vec4f,
 @location(2) @interpolate(flat) role:u32 };
@vertex fn vs(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->VOut {
 let corners=array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
 let q=corners[vertex];var p=vec2f(0);var r=1.0;var role=instance;var color=vec4f(.5,.88,.76,1);
 if(instance==0u) {r=u.misc.x;color=vec4f(.2,.34,.30,1);}
 else if(instance==1u) {p=u.scene.zw;r=u.radii.y;color=vec4f(.41,.85,.61,1);}
 else if(instance==2u) {p=u.scene.xy;r=u.radii.x;color=vec4f(.86,.65,.42,1);}
 else if(instance==3u||instance==4u) {p=u.pick.xy;r=select(u.radii.w,u.misc.z,instance==4u);color=vec4f(.28,.60,.66,select(1.,.4,instance==4u));if(u.pick.w<.5){r=0.0;}}
 else {let i=instance-5u;let a=previous[i];let b=current[i];p=mix(a.p,b.p,u.view.z);r=max(u.radii.z,u.misc.y);
  color=mix(vec4f(.42,.73,.65,1),vec4f(.59,.97,.82,1),clamp(length(b.v)*2.0,0.,1.));
  if(u32(u.pick.z)==i&&u.pick.w>.5){color=vec4f(.95,.99,.93,1);r*=1.65;}role=5u;}
 var out:VOut;out.pos=vec4f((p+q*r-u.camera.xy)*u.view.xy,0,1);out.local=q;out.color=color;out.role=role;return out;
}
@fragment fn fs(v:VOut)->@location(0) vec4f {
 let localWidth=fwidth(v.local);let d=length(v.local);let radiusWidth=fwidth(d);
 if(v.role==0u){let world=v.local*u.misc.x;let grid=abs(fract(world+.5)-.5);let aw=max(localWidth*u.misc.x,vec2f(.0001));
  let line=1.-min(min(grid.x/aw.x,grid.y/aw.y),1.);let border=1.-smoothstep(.993,1.,max(abs(v.local.x),abs(v.local.y)));
  return vec4f(v.color.rgb,max(line*.22,1.-border));}
 if(d>1.){discard;}let aa=max(radiusWidth,.005);let edge=1.-smoothstep(1.-aa,1.,d);
 if(v.role==1u||v.role==3u||v.role==4u){let ring=smoothstep(.96-aa,.96,d);return vec4f(v.color.rgb,edge*mix(.055,.85,ring)*v.color.a);}
 if(v.role==2u){let light=.85+.15*(1.-d);return vec4f(v.color.rgb*light,edge);}
 return vec4f(v.color.rgb,edge);
}`;

class GPURenderer {
  constructor(canvas,adapter,device,onLost) {
    this.canvas=canvas;this.device=device;this.gpu=webgpu;this.adapterHandle=adapter;this.name='WebGPU';this.frame=null;this.previous=null;
    const info=adapter.info||{};this.adapter=[info.vendor,info.architecture,info.device,info.description].filter(Boolean).join(' · ')||'WebGPU adapter (名称不可用)';
    this.context=canvas.getContext('webgpu');this.format=webgpu.getPreferredCanvasFormat();
    this.context.configure({device,format:this.format,alphaMode:'opaque'});
    this.uniform=device.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.uniformData=new Float32Array(24);this.buffers=[];this.disposed=false;this.lastQuery=0;this.queryPending=false;this.inFlight=0;
    device.lost.then(info=>{if(!this.disposed)onLost(`WebGPU device lost: ${info.reason}`);});
    device.addEventListener('uncapturederror',e=>{if(!this.disposed)onLost(e.error.message);});
  }
  async init(){
    const module=this.device.createShaderModule({code:shader});
    const compilation=await module.getCompilationInfo();
    const errors=compilation.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(e=>e.message).join('; '));
    this.pipeline=await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'triangle-list'}});
    if(this.device.features.has('timestamp-query')){
      this.query=this.device.createQuerySet({type:'timestamp',count:2});
      this.resolve=this.device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      this.read=this.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    }
  }
  setFrame(frame){this.pendingFrame=frame;}
  uploadFrame(frame){
    const bytes=frame.data.byteLength;const fresh=!this.frame||this.frame.n!==frame.n;
    this.previous=fresh?frame:this.frame;this.frame=frame;
    if(fresh){for(const b of this.buffers)b.destroy();this.buffers=[0,1].map(()=>this.device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));this.index=0;this.device.queue.writeBuffer(this.buffers[1],0,frame.data);}
    else this.index=1-this.index;
    this.device.queue.writeBuffer(this.buffers[this.index],0,frame.data);
    this.bind=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniform}},{binding:1,resource:{buffer:this.buffers[1-this.index]}},{binding:2,resource:{buffer:this.buffers[this.index]}}]});
  }
  render(blend,now){
    if(this.disposed||this.inFlight>=2)return false;
    if(this.pendingFrame){this.uploadFrame(this.pendingFrame);this.pendingFrame=null;}
    if(!this.frame)return false;resizeCanvas(this.canvas);
    const f=this.frame,t=transform(this.canvas),u=this.uniformData,old=this.previous.f;
    u.set([t.sx,t.sy,blend,f.n],0);u.set([old[0]+(f.f[0]-old[0])*blend,old[1]+(f.f[1]-old[1])*blend,f.f[2],f.f[3]],4);
    u.set([f.f[4],f.f[5],f.f[6],f.f[8]],8);
    const selected=state.selected>=0&&state.selected<f.n,k=state.selected*8;
    u.set([selected?f.data[k]:0,selected?f.data[k+1]:0,Math.max(state.selected,0),selected?1:0],12);
    u.set([f.f[7],1.0/(this.canvas.height*t.sy),f.f[9],0],16);u.set([state.camera.x,state.camera.y,0,0],20);
    this.device.queue.writeBuffer(this.uniform,0,u);
    const encoder=this.device.createCommandEncoder();const timed=this.query&&!this.queryPending&&now-this.lastQuery>1000;
    const descriptor={colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:.051,g:.106,b:.118,a:1},loadOp:'clear',storeOp:'store'}]};
    if(timed)descriptor.timestampWrites={querySet:this.query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1};
    const pass=encoder.beginRenderPass(descriptor);pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bind);pass.draw(6,f.n+5);pass.end();
    if(timed){encoder.resolveQuerySet(this.query,0,2,this.resolve,0);encoder.copyBufferToBuffer(this.resolve,0,this.read,0,16);this.queryPending=true;this.lastQuery=now;}
    this.device.queue.submit([encoder.finish()]);
    // Coalesce display work when a slow adapter falls behind; never grow a frame queue.
    this.inFlight++;
    this.device.queue.onSubmittedWorkDone().catch(()=>{}).finally(()=>{this.inFlight--;});
    if(timed)this.read.mapAsync(GPUMapMode.READ).then(()=>{const ticks=new BigUint64Array(this.read.getMappedRange());const ms=Number(ticks[1]-ticks[0])/1e6;this.read.unmap();if(!this.disposed)$('gpu-ms').textContent=fmt(ms)+' ms';}).catch(()=>{}).finally(()=>{this.queryPending=false;});
    return true;
  }
  dispose(){this.disposed=true;for(const b of this.buffers)b.destroy();this.uniform.destroy();this.query?.destroy();this.resolve?.destroy();this.read?.destroy();this.device.destroy();}
}

let rendererGeneration=0;
async function chooseRenderer(mode='auto') {
  const generation=++rendererGeneration;
  state.renderer?.dispose();state.renderer=null;
  const old=$('scene'),canvas=old.cloneNode(false);old.replaceWith(canvas);resizeCanvas(canvas);
  let renderer;
  try {
    if(mode==='canvas'||!webgpu)throw new Error(mode==='canvas'?'用户选择 Canvas':'浏览器未提供 WebGPU');
    const adapter=await webgpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw new Error('没有 WebGPU adapter');
    const info=adapter.info||{},name=Object.values(info).join(' ')+[info.vendor,info.architecture,info.device,info.description].join(' ');
    const softwareAdapter=adapter.isFallbackAdapter||/swiftshader|llvmpipe|lavapipe|software|basic render/i.test(name);
    if(mode==='auto'&&softwareAdapter)throw new Error('自动模式避开软件 WebGPU');
    const requiredFeatures=!softwareAdapter&&adapter.features.has('timestamp-query')?['timestamp-query']:[];
    const device=await adapter.requestDevice({requiredFeatures});
    renderer=new GPURenderer(canvas,adapter,device,message=>{if(generation!==rendererGeneration)return;state.errors.push(message);console.error(message);chooseRenderer('canvas');});
    await renderer.init();
  } catch(e) {
    renderer?.dispose();
    const replacement=canvas.cloneNode(false);canvas.replaceWith(replacement);renderer=new CanvasRenderer(replacement);
    renderer.adapter=`${renderer.adapter} · ${e.message}`;
    $('gpu-ms').textContent='未采集';
  }
  if(generation!==rendererGeneration){renderer.dispose();return;}
  state.renderer=renderer;state.backend=renderer.name;$('backend').textContent=renderer.name;$('adapter').textContent=renderer.adapter;
  if(state.frame)renderer.setFrame(state.frame);state.dirty=true;
}

function updateMetrics(){
  const f=state.frame;if(!f)return;
  state.paused=Boolean(f.flags&1);
  $('count').textContent=f.n.toLocaleString();$('steps').textContent=f.step.toLocaleString();$('distance').textContent=fmt(f.f[13])+' m';
  $('contacts').textContent=Math.round(f.f[14]).toLocaleString();$('object-speed').textContent=fmt(f.f[15],3)+' m/s';
  $('largest').textContent=f.f[10]<0?'尚未采样':fmt(f.f[10]*100,1)+'%';
  $('connected').textContent=f.f[11]<0?'尚未采样':f.f[11]>.5?'连通':'已断开';
  $('bridge').textContent=f.f[19]===0?'尚未发现两端':f.f[12]<0?'尚未采样':f.f[12]>.5?'保持':'未建立';
  $('step-ms').textContent=fmt(f.f[16],1)+' ms';
  const finished=Boolean(f.flags&6);
  $('play').textContent=finished?'已结束':state.paused?'▶ 开始':'Ⅱ 暂停';$('play').disabled=finished;$('reset').disabled=false;
  $('run-state').textContent=f.flags&2?'搬运完成':f.flags&4?'时间用尽':state.paused?'已暂停':'运行中';
  $('scene-label').textContent=`${state.paused?'PAUSED':'LIVE'} · STEP ${f.step}`;
  if(state.initialDistance===null||f.step===0)state.initialDistance=f.f[13];
  const tolerance=f.f[5]-f.f[4],range=Math.max(state.initialDistance-tolerance,.001);
  $('progress-fill').style.width=Math.max(0,Math.min(100,(state.initialDistance-f.f[13])/range*100))+'%';
  if(state.selected>=0&&state.selected<f.n){const k=state.selected*8;$('selection').textContent=`粒子 #${state.selected} · ${Math.round(f.data[k+5])} 个邻居\n速度 ${fmt(Math.hypot(f.data[k+2],f.data[k+3]),3)} m/s · 记忆 RMS ${fmt(f.data[k+4],3)}`;}
}
async function pollStatus(){
  try {
    const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const s=await response.json();state.status=s;
    $('connection').textContent='本机已连接';$('connection-dot').classList.add('online');
    $('device').textContent=`仿真 / 策略设备：${s.device}`;$('stage-badge').textContent=s.config.stage==='search'?'SEARCH · 正式任务':'CONTACT · 热身任务';
    const ck=s.checkpoint,trained=ck.has_trained_weights??Number(ck.updates)>0;
    $('training-notice').textContent=trained?`已加载 ${ck.name} · 本段 ${ck.updates} 次更新${ck.lineage?.kind==='warmstart'?' · 继承已训练权重':''}。加载权重不代表已学会搬运，实际质量以独立评价为准。`:'未训练策略 · 这是随机初始化网络的工程诊断，不能作为学会探索、连通或搬运的证据。';
    $('training-notice').classList.toggle('trained',trained);
    $('checkpoint').textContent=`策略：${ck.name}${ck.checkpoint_sha256?' · '+ck.checkpoint_sha256.slice(0,12):''} · ${s.policy_mode==='sampled_seeded'?'抽样动作':'均值动作'}`;
    if(s.error)error(`仿真已暂停：${s.error}`);
  } catch(e) {$('connection').textContent='连接中断';$('connection-dot').classList.remove('online');}
  if(!state.stopped)setTimeout(pollStatus,2000);
}
async function pollSnapshot(){
  if(state.stopped)return;
  if(!state.visible){setTimeout(pollSnapshot,250);return;}
  const controller=new AbortController();state.request=controller;
  try {
    const headers={};if(state.frame)headers['If-None-Match']=`"${state.frame.sequence}"`;
    const response=await fetch('/api/snapshot',{headers,cache:'no-store',signal:controller.signal});
    if(response.status!==304){if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const frame=parseSnapshot(await response.arrayBuffer());
      if(!state.frame||frame.sequence!==state.frame.sequence){
        const now=performance.now();state.interval=Math.min(500,Math.max(16,now-state.received));state.received=now;
        if(!state.frame||frame.step!==state.frame.step)state.snapshots++;
        state.previous=state.frame;state.frame=frame;state.renderer?.setFrame(frame);state.dirty=true;updateMetrics();
      }
    }
  } catch(e){if(e.name!=='AbortError')state.errors.push(e.message);}
  finally{if(state.request===controller)state.request=null;if(!state.stopped)setTimeout(pollSnapshot,40);}
}
async function command(command,value){
  try{const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,value})});const result=await response.json();if(!response.ok)throw new Error(result.error||`HTTP ${response.status}`);if(command==='reset'){$('scene-error').hidden=true;state.initialDistance=null;}return result;}
  catch(e){error(e.message);}
}
function animate(now){
  if(state.stopped)return;
  const blend=Math.min(1,(now-state.received)/state.interval);
  if(state.visible&&state.renderer&&state.frame&&(state.dirty||blend<1)){
    const submitted=state.renderer.render(blend,now)!==false;
    if(submitted){state.draws++;state.renderedFrames++;state.dirty=false;}else state.dirty=true;
  }
  if(now-state.lastMeter>=1000){const elapsed=(now-state.lastMeter)/1000;$('snapshot-rate').textContent=fmt(state.snapshots/elapsed,1);$('draw-rate').textContent=fmt(state.draws/elapsed,1);state.snapshots=0;state.draws=0;state.lastMeter=now;}
  requestAnimationFrame(animate);
}
$('play').onclick=()=>command(state.paused?'play':'pause');$('reset').onclick=()=>command('reset');$('speed').onchange=e=>command('speed',Number(e.target.value));
$('fit').onclick=()=>{state.camera={x:0,y:0,zoom:1};state.dirty=true;};
$('renderer').onchange=e=>chooseRenderer(e.target.value);
$('stage').addEventListener('wheel',e=>{e.preventDefault();state.camera.zoom=Math.max(.5,Math.min(10,state.camera.zoom*Math.exp(-e.deltaY*.001)));state.dirty=true;},{passive:false});
let pointer=null;
$('stage').addEventListener('pointerdown',e=>{if(e.target.tagName!=='CANVAS')return;pointer={x:e.clientX,y:e.clientY,cx:state.camera.x,cy:state.camera.y,moved:false};e.target.setPointerCapture(e.pointerId);});
$('stage').addEventListener('pointermove',e=>{if(!pointer)return;const canvas=$('scene'),t=transform(canvas),r=canvas.getBoundingClientRect();const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;if(Math.hypot(dx,dy)>4)pointer.moved=true;if(pointer.moved){state.camera.x=pointer.cx-dx*2*t.half*t.aspect/r.width;state.camera.y=pointer.cy+dy*2*t.half/r.height;state.dirty=true;}});
$('stage').addEventListener('pointerup',e=>{if(!pointer)return;const moved=pointer.moved;pointer=null;if(moved||!state.frame)return;const canvas=$('scene'),r=canvas.getBoundingClientRect(),t=transform(canvas);const x=(2*(e.clientX-r.left)/r.width-1)*t.half*t.aspect+state.camera.x,y=(1-2*(e.clientY-r.top)/r.height)*t.half+state.camera.y;
  let best=-1,distance=(state.frame.f[8]*2)**2;for(let i=0;i<state.frame.n;i++){const k=i*8,d=(state.frame.data[k]-x)**2+(state.frame.data[k+1]-y)**2;if(d<distance){distance=d;best=i;}}
  state.selected=best;state.dirty=true;if(best<0)$('selection').textContent='单击一个粒子，查看其感知范围与邻居数量。';updateMetrics();});
$('stage').addEventListener('pointercancel',()=>{pointer=null;});
new ResizeObserver(()=>{state.dirty=true;}).observe($('stage'));
document.addEventListener('visibilitychange',()=>{state.visible=!document.hidden;if(!state.visible)state.request?.abort();else state.dirty=true;});
window.addEventListener('pagehide',()=>{state.stopped=true;state.request?.abort();state.renderer?.dispose();});
window.__swarm={get diagnostics(){return {backend:state.backend,adapter:state.renderer?.adapter,sequence:state.frame?.sequence,step:state.frame?.step,particles:state.frame?.n,selected:state.selected,paused:state.paused,errors:[...state.errors],snapshotBytes:state.frame?.buffer.byteLength,visible:state.visible,renderedFrames:state.renderedFrames};},parseSnapshot};
const initialMode=['canvas','webgpu'].includes(params.get('renderer'))?params.get('renderer'):'auto';$('renderer').value=initialMode;
await chooseRenderer(initialMode);pollStatus();pollSnapshot();requestAnimationFrame(animate);
