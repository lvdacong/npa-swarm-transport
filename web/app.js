const $=id=>document.getElementById(id), gpu=navigator.gpu;
const state={frame:null,previous:null,renderer:null,status:null,selected:-1,paused:true,
  camera:{x:0,y:0,zoom:1},received:0,interval:100,dirty:true,raf:0,visible:!document.hidden,
  errors:[],draws:0,snapshots:0,renderedFrames:0,lastMeter:performance.now(),stopped:false};
const fmt=(v,d=1)=>Number.isFinite(v)?v.toFixed(d):'—';
function fail(message){message=String(message);if(state.errors.at(-1)!==message){state.errors.push(message);if(state.errors.length>20)state.errors.shift();}$('scene-error').textContent=message;$('scene-error').hidden=false;}
function parseSnapshot(buffer){
  if(buffer.byteLength<128)throw Error('快照不完整');
  const u=new Uint32Array(buffer,0,8),f=new Float32Array(buffer,32,24);
  if(u[0]!==0x3153504e||u[1]!==1||u[3]!==128||buffer.byteLength!==128+u[2]*32)throw Error('快照格式不匹配');
  return {buffer,n:u[2],sequence:u[4],flags:u[5],step:u[6],revision:u[7],f,data:new Float32Array(buffer,128,u[2]*8)};
}
function transform(){const c=$('scene'),aspect=c.width/c.height,half=(state.frame?.f[7]||10)*1.08/(state.camera.zoom*Math.min(aspect,1));return {sx:1/(half*aspect),sy:1/half};}
function resize(){const c=$('scene'),r=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(c.width!==w||c.height!==h){c.width=w;c.height=h;state.dirty=true;}}
// Orthogonal visual semantics: cargo=core, goal=ring, contact=corona,
 // direct sensing=glint, disconnected=opacity. All read existing snapshot data.
 // Shader-only rendering; no extra passes, buffers, or policy inputs.
const shader=`
struct Particle {p:vec2f,v:vec2f,signal:vec4f};
struct Uniforms {view:vec4f,scene:vec4f,radii:vec4f,pick:vec4f,misc:vec4f,camera:vec4f};
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> previous:array<Particle>;
@group(0) @binding(2) var<storage,read> current:array<Particle>;
struct Out {
  @builtin(position) pos:vec4f,
  @location(0) local:vec2f,
  @location(1) color:vec4f,
  @location(2) @interpolate(flat) role:u32,
  @location(3) @interpolate(flat) info:vec4f
};
@vertex fn vs(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32)->Out {
  let corners=array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));
  let q=corners[vi];var p=vec2f(0.);var r=1.;var role=ii;
  var color=vec4f(.55,.85,.76,1.);var info=vec4f(0.);
  if(ii==0u){r=u.misc.x;color=vec4f(.23,.36,.39,1.);}
  else if(ii==1u){p=u.scene.zw;r=u.radii.y;color=vec4f(.62,.60,.96,1.);}
  else if(ii==2u){p=u.scene.xy;r=u.radii.x;color=vec4f(.98,.68,.33,1.);}
  else if(ii==3u||ii==4u){p=u.pick.xy;r=select(u.radii.w,u.misc.z,ii==4u);
    color=vec4f(.38,.71,.79,select(1.,.4,ii==4u));if(u.pick.w<.5){r=0.;}}
  else {
    let i=ii-5u;
    let a=previous[i];let b=current[i];
    p=mix(a.p,b.p,u.view.z);
    info=b.signal;
    let touch=(u32(round(info.w))&8u)!=0u;
    r=max(u.radii.z,u.misc.y)*select(1.46,2.18,touch);
    role=5u;
  }
  var o:Out;
  o.pos=vec4f((p+q*r-u.camera.xy)*u.view.xy,0.,1.);
  o.local=q;o.color=color;o.role=role;o.info=info;
  return o;
}
@fragment fn fs(v:Out)->@location(0) vec4f {
  let d=length(v.local);
  let dw=max(fwidth(d),.008);
  let w=fwidth(v.local);
  if(v.role==0u){
    let world=v.local*u.misc.x;
    let grid=abs(fract(world*.5+.5)-.5);
    let aw=max(w*u.misc.x*.5,vec2f(.0001));
    let line=1.-min(min(grid.x/aw.x,grid.y/aw.y),1.);
    let border=smoothstep(.992,1.,max(abs(v.local.x),abs(v.local.y)));
    return vec4f(v.color.rgb,max(line*.13,border*.65));
  }
  if(d>1.){discard;}
  let aa=max(dw*.5,.006);
  if(v.role==1u||v.role==3u||v.role==4u){
    let edge=1.-smoothstep(1.-aa,1.,d);
    let ring=smoothstep(.96-aa,.96,d);
    return vec4f(v.color.rgb,edge*mix(.055,.9,ring)*v.color.a);
  }
  if(v.role==2u){
    let edge=1.-smoothstep(1.-aa,1.,d);
    return vec4f(v.color.rgb*(.85+.15*(1.-d*d)),edge);
  }
  // Orthogonal layers: cargo confidence = core fill; goal = outer
  // violet outline; physical contact = independent corona; detached = alpha.
  let flags=u32(round(v.info.w));
  let touch=(flags&8u)!=0u;
  let live=(flags&1u)!=0u;
  let direct=(flags&6u)!=0u;
  let bodyR=select(1./1.46,1./2.18,touch);
  let cargo=clamp(v.info.x,0.,1.);
  let goal=clamp(v.info.y,0.,1.);
  let signedPush=clamp(v.info.z,-1.,1.);
  let coreColor=mix(vec3f(.36,.66,.69),vec3f(.98,.68,.33),sqrt(cargo)*.94);
  let core=1.-smoothstep(bodyR-aa*.8,bodyR+aa*.55,d);
  let ringDist=abs(d-(bodyR+.145));
  let band=max(.058,dw*.40);
  let outline=(1.-smoothstep(band-aa*.50,band+aa*.55,ringDist))*pow(goal,.65)*.98;
  var corona=0.;var haloColor=vec3f(.79,.83,.85);
  if(touch){
    let power=clamp(abs(signedPush),0.,1.);
    haloColor=select(vec3f(1.,.43,.48),vec3f(1.,.83,.46),signedPush>=.045);
    if(power<.045){haloColor=vec3f(.7,.8,.82);}
    let shoulder=smoothstep(bodyR+.20,bodyR+.34,d);
    let falloff=1.-smoothstep(.85,1.02,d);
    var shape=1.;
    if(signedPush<-.045){
      let angle=atan2(v.local.y,v.local.x+1e-6);
      shape=.16+.84*smoothstep(-.25,.35,sin(angle*4.));
    }
    corona=shoulder*falloff*shape*(.24+.42*power);
  }
  let coreLayer=core;
  let rimLayer=outline*(1.-coreLayer);
  let haloLayer=corona*(1.-coreLayer)*(1.-outline);
  var alpha=coreLayer+rimLayer+haloLayer;
  var rgb=(coreColor*coreLayer+vec3f(.64,.60,.99)*rimLayer+haloColor*haloLayer)/max(alpha,1e-5);
  if(direct){
    let glint=length(v.local-vec2f(-bodyR*.27,-bodyR*.26));
    let sparkle=(1.-smoothstep(.06,.16,glint))*(.74+.16*sin(u.view.w*2.2));
    rgb=mix(rgb,vec3f(1.,.98,.93),sparkle*coreLayer);
  }
  if(!live){alpha*=.34;}
  return vec4f(rgb,alpha);
}
`;
class Renderer{
  constructor(adapter,device){
    this.device=device;this.canvas=$('scene');this.context=this.canvas.getContext('webgpu');this.disposed=false;this.inFlight=0;
    this.format=gpu.getPreferredCanvasFormat();this.context.configure({device,format:this.format,alphaMode:'opaque'});
    const i=adapter.info||{};this.adapter=[i.vendor,i.architecture,i.device,i.description].filter(Boolean).join(' · ')||'WebGPU';
    this.uniform=device.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.uniformData=new Float32Array(24);
    this.buffers=[];this.lastQuery=0;this.queryPending=false;this.uploads=0;
    device.lost.then(info=>{if(!this.disposed){fail(`WebGPU 设备已断开（${info.reason}），请刷新页面。`);this.dispose();$('play').disabled=true;command({command:'pause'});}});
    device.addEventListener('uncapturederror',e=>{if(this.disposed)return;fail(e.error.message);this.dispose();$('play').disabled=true;command({command:'pause'});});
  }
  async init(){
    const module=this.device.createShaderModule({code:shader}),compilation=await module.getCompilationInfo();
    const errors=compilation.messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(e=>e.message).join('; '));
    this.pipeline=await this.device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vs'},
      fragment:{module,entryPoint:'fs',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'triangle-list'}});
    if(this.device.features.has('timestamp-query')){
      this.query=this.device.createQuerySet({type:'timestamp',count:2});
      this.resolve=this.device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      this.read=this.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    }
  }
  upload(frame){
    const fresh=!this.frame||this.frame.n!==frame.n,reset=fresh||frame.step<this.frame.step;
    if(fresh){
      for(const b of this.buffers)b.destroy();
      this.buffers=[0,1].map(()=>this.device.createBuffer({size:frame.data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}));
      this.bindings=[0,1].map(i=>this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:this.uniform}},{binding:1,resource:{buffer:this.buffers[1-i]}},{binding:2,resource:{buffer:this.buffers[i]}}]}));
      this.index=0;
    }else this.index=1-this.index;
    if(reset)this.device.queue.writeBuffer(this.buffers[1-this.index],0,frame.data);
    this.previous=reset?frame:this.frame;this.frame=frame;
    this.device.queue.writeBuffer(this.buffers[this.index],0,frame.data);this.uploads++;
  }
  render(blend,now){
    if(this.disposed||this.inFlight>=2)return false;
    const frame=state.frame;if(!frame)return false;
    if(this.frame?.sequence!==frame.sequence)this.upload(frame);
    resize();const t=transform(),f=this.frame,old=this.previous.f,u=this.uniformData;
    u.set([t.sx,t.sy,blend,now*.001],0);u.set([old[0]+(f.f[0]-old[0])*blend,old[1]+(f.f[1]-old[1])*blend,f.f[2],f.f[3]],4);
    u.set([f.f[4],f.f[5],f.f[6],f.f[8]],8);const pick=state.selected>=0&&state.selected<f.n,k=state.selected*8;
    u.set([pick?f.data[k]:0,pick?f.data[k+1]:0,Math.max(0,state.selected),pick?1:0],12);
    u.set([f.f[7],1.15/(this.canvas.height*t.sy),f.f[9],0],16);u.set([state.camera.x,state.camera.y,0,0],20);
    this.device.queue.writeBuffer(this.uniform,0,u);
    const encoder=this.device.createCommandEncoder();const timed=this.query&&!this.queryPending&&!$('diagnostics').hidden&&now-this.lastQuery>1000;
    const desc={colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:.043,g:.078,b:.094,a:1},loadOp:'clear',storeOp:'store'}]};
    if(timed)desc.timestampWrites={querySet:this.query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1};
    const pass=encoder.beginRenderPass(desc);pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindings[this.index]);pass.draw(6,f.n+5);pass.end();
    if(timed){encoder.resolveQuerySet(this.query,0,2,this.resolve,0);encoder.copyBufferToBuffer(this.resolve,0,this.read,0,16);this.queryPending=true;this.lastQuery=now;}
    this.device.queue.submit([encoder.finish()]);this.inFlight++;
    this.device.queue.onSubmittedWorkDone().catch(()=>{}).finally(()=>{this.inFlight--;});
    if(timed)this.read.mapAsync(GPUMapMode.READ).then(()=>{const ticks=new BigUint64Array(this.read.getMappedRange());$('gpu-ms').textContent=fmt(Number(ticks[1]-ticks[0])/1e6,3)+' ms';this.read.unmap();}).catch(()=>{}).finally(()=>{this.queryPending=false;});
    return true;
  }
  dispose(){if(this.disposed)return;this.disposed=true;for(const b of this.buffers)b.destroy();this.uniform.destroy();this.query?.destroy();this.resolve?.destroy();this.read?.destroy();this.device.destroy();}
}
function requestDraw(){state.dirty=true;if(!state.raf&&state.visible&&!state.stopped)state.raf=requestAnimationFrame(draw);}
function draw(now){
  state.raf=0;if(state.stopped||!state.visible||state.renderer?.disposed)return;
  const blend=state.paused?1:Math.min(1,(now-state.received)/Math.max(16,state.interval));
  if(state.dirty||blend<1){if(state.renderer?.render(blend,now)){state.draws++;state.renderedFrames++;state.dirty=false;}}
  if(now-state.lastMeter>=1000){const dt=(now-state.lastMeter)/1000;$('state-rate').textContent=fmt(state.snapshots/dt);$('draw-rate').textContent=fmt(state.draws/dt);state.snapshots=state.draws=0;state.lastMeter=now;}
  if(state.dirty||blend<1)state.raf=requestAnimationFrame(draw);
}
function updateHud(){
  const f=state.frame;if(!f)return;
  state.paused=!!(f.flags&1);$('play-text').textContent=state.paused?'播放':'暂停';$('play-icon').textContent=state.paused?'▶':'Ⅱ';$('play').setAttribute('aria-label',state.paused?'播放':'暂停');
  $('clock').textContent=fmt(f.step*f.f[18])+' s';$('population').textContent=f.n+' 粒子';
  $('found-object').classList.toggle('found',f.f[21]>.5);$('found-goal').classList.toggle('found',f.f[22]>.5);
  $('found-object').title=f.f[21]>.5?'已发现重物':'尚未发现重物';$('found-goal').title=f.f[22]>.5?'已发现目的地':'尚未发现目的地';
  $('distance').textContent='剩余 '+fmt(Math.max(0,f.f[13]-(f.f[5]-f.f[4])));
  $('connected').textContent='连通 '+fmt(f.f[10]*100,1)+'%';$('connected').className=f.f[11]>.5?'good':'bad';
  $('connected').title='最大连通分量占比；灰蓝粒子已离开主群。没有删除或自动重连。';
  $('bridge').textContent=f.f[19]<.5?'待发现两端':f.f[12]>.5?'两端已接通':'两端未接通';$('bridge').className=f.f[12]>.5?'good':'';
  $('bridge').title='同一粒子连通分量须分别进入两端的通信范围；看见不等于接通。';
  $('step-ms').textContent=fmt(f.f[16],2)+' ms';$('actual-speed').textContent=fmt(f.f[20],2)+'×';$('endpoint-gap').textContent=fmt(f.f[23],2);
  const terminal=f.flags&6;$('terminal').hidden=!terminal;$('terminal').textContent=(f.flags&2)?'已送达':'本轮结束 · 未送达';
  $('play').disabled=!!terminal||!state.renderer||state.renderer.disposed;
  if(state.selected>=f.n)state.selected=-1;
  const k=state.selected*8;$('selection').hidden=state.selected<0;
  if(state.selected>=0){
    const particleFlags=Math.round(f.data[k+7]);
    const hints=[];
    if(particleFlags&2)hints.push('直视重物');
    if(particleFlags&4)hints.push('直视目的地');
    if(particleFlags&8)hints.push(f.data[k+6]>.04?'正向施力':f.data[k+6]<-.04?'反向施力':'接触重物');
    if(!(particleFlags&1))hints.push('离群');
    if(particleFlags&16)hints.push('两端通路');
    $('selection').textContent=`#${state.selected} · 重物信息 ${Math.round(f.data[k+4]*100)}% · 目的地信息 ${Math.round(f.data[k+5]*100)}%`+(hints.length?' · '+hints.join(' · '):'');
  }
}
let etag='',pollTimer,requestController,pollGeneration=0,commandChain=Promise.resolve();
function restartPolling(){const generation=++pollGeneration;clearTimeout(pollTimer);requestController?.abort();if(state.visible&&!state.stopped)pollTimer=setTimeout(()=>poll(generation),10);}
async function poll(generation=pollGeneration){
  if(state.stopped||!state.visible||generation!==pollGeneration)return;
  requestController=new AbortController();
  try{
    const response=await fetch('/api/snapshot',{headers:etag?{'If-None-Match':etag}:{},signal:requestController.signal});
    if(response.status===200){const f=parseSnapshot(await response.arrayBuffer());if(generation!==pollGeneration)return;etag=response.headers.get('ETag')||'';
      if(!state.frame||f.sequence!==state.frame.sequence){const now=performance.now();state.interval=Math.min(500,Math.max(16,now-state.received));state.previous=state.frame;state.frame=f;state.received=now;state.snapshots++;updateHud();requestDraw();if(f.flags&8)status().catch(e=>fail(e.message));}}
    else if(response.status!==304)throw Error('状态服务不可用');
  }catch(e){if(e.name!=='AbortError'&&generation===pollGeneration)fail(e.message);}
  if(!state.stopped&&state.visible&&generation===pollGeneration)pollTimer=setTimeout(()=>poll(generation),state.paused?150:33);
}
async function status(){
  const r=await fetch('/api/status');if(!r.ok)throw Error('模拟服务不可用');state.status=await r.json();
  $('sim-device').textContent=state.status.device;$('model-tag').hidden=!!state.status.checkpoint.has_trained_weights;
  $('policy-mode').textContent=state.status.policy_mode==='sampled_seeded'?'固定种子采样':'确定性';
  $('speed').textContent=state.status.speed===0?'∞':state.status.speed+'×';
  if(state.status.error)fail(state.status.error);
}
function command(body){
  commandChain=commandChain.then(async()=>{
    const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(typeof body==='function'?body():body)});
    const result=await r.json();if(!r.ok)throw Error(result.error||'控制失败');
    await status();restartPolling();
  }).catch(e=>fail(e.message));return commandChain;
}
$('play').onclick=()=>command(()=>({command:state.status?.paused?'play':'pause'}));
$('reset').onclick=()=>{state.selected=-1;command({command:'reset'});};
$('speed').onclick=()=>command(()=>{const values=[1,4,8,0],old=state.status?.speed??4;return {command:'speed',value:values[(values.indexOf(old)+1)%values.length]};});
$('fit').onclick=()=>{state.camera={x:0,y:0,zoom:1};requestDraw();};
$('details').onclick=()=>{const panel=$('diagnostics');panel.hidden=!panel.hidden;$('details').setAttribute('aria-expanded',String(!panel.hidden));status().catch(e=>fail(e.message));requestDraw();};
let pointer=null;
const canvas=$('scene');
canvas.addEventListener('wheel',e=>{e.preventDefault();state.camera.zoom=Math.min(12,Math.max(.65,state.camera.zoom*Math.exp(-e.deltaY*.001)));requestDraw();},{passive:false});
canvas.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!pointer)return;const r=canvas.getBoundingClientRect(),t=transform();state.camera.x-=(e.clientX-pointer.x)*2/(r.width*t.sx);state.camera.y+=(e.clientY-pointer.y)*2/(r.height*t.sy);pointer.x=e.clientX;pointer.y=e.clientY;requestDraw();});
canvas.addEventListener('pointerup',e=>{
  if(pointer&&Math.hypot(e.clientX-pointer.sx,e.clientY-pointer.sy)<5&&state.frame){const r=canvas.getBoundingClientRect(),t=transform(),x=((e.clientX-r.left)/r.width*2-1)/t.sx+state.camera.x,y=(1-(e.clientY-r.top)/r.height*2)/t.sy+state.camera.y;let best=Infinity,selected=-1;
    for(let i=0;i<state.frame.n;i++){const k=i*8,d=Math.hypot(state.frame.data[k]-x,state.frame.data[k+1]-y);if(d<best){best=d;selected=i;}}
    state.selected=best<20/(r.height*t.sy)?selected:-1;updateHud();requestDraw();}
  pointer=null;
});
canvas.addEventListener('pointercancel',()=>{pointer=null;});
window.addEventListener('resize',()=>{resize();requestDraw();});
document.addEventListener('visibilitychange',()=>{state.visible=!document.hidden;restartPolling();if(state.visible){state.received=performance.now();requestDraw();}});
window.addEventListener('pagehide',()=>{state.stopped=true;clearTimeout(pollTimer);requestController?.abort();cancelAnimationFrame(state.raf);state.renderer?.dispose();});
// Read-only test/diagnostic entrypoint. Does not advance physics or alter policy.
window.swarmDiagnostics=()=>({backend:state.renderer&&!state.renderer.disposed?'WebGPU':null,adapter:state.renderer?.adapter,
  sequence:state.frame?.sequence,step:state.frame?.step,particles:state.frame?.n,selected:state.selected,paused:state.paused,
  errors:[...state.errors],visible:state.visible,renderedFrames:state.renderedFrames,uploads:state.renderer?.uploads,
  actualSpeed:state.frame?.f[20],snapshotBytes:state.frame?.buffer.byteLength,camera:{...state.camera}});
async function start(){
  resize();if(!gpu)throw Error('此浏览器未提供 WebGPU。请使用启用硬件加速的 Chrome / Edge，并通过 localhost 打开。');
  const adapter=await gpu.requestAdapter({powerPreference:'low-power'});if(!adapter)throw Error('没有可用的 WebGPU 设备，请检查浏览器硬件加速。');
  const info=adapter.info||{},name=[info.vendor,info.architecture,info.description].join(' '),software=/swiftshader|llvmpipe|lavapipe|software/i.test(name)||adapter.isFallbackAdapter;
  const device=await adapter.requestDevice({requiredFeatures:!software&&adapter.features.has('timestamp-query')?['timestamp-query']:[]});
  const renderer=new Renderer(adapter,device);state.renderer=renderer;try{await renderer.init();}catch(e){renderer.dispose();throw e;}$('adapter').textContent=renderer.adapter;
  await status();$('reset').disabled=$('speed').disabled=false;poll();
}
start().catch(e=>fail(e.message));
