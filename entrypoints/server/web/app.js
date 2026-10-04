'use strict';
const $ = id => document.getElementById(id);
const video = $('inputVideo'), canvas = $('paint'), ctx = canvas.getContext('2d');
let urls = {}, drawing = false, lastPoint = null, painted = false, uploading = false;
function message(text) { $('message').textContent = text; }
function replaceURL(key, file) {
  if (urls[key]) URL.revokeObjectURL(urls[key]);
  urls[key] = file ? URL.createObjectURL(file) : '';
  return urls[key];
}
$('videoFile').onchange = () => { video.src = replaceURL('video', $('videoFile').files[0]); painted = false; };
video.onloadedmetadata = () => { canvas.width = video.videoWidth; canvas.height = video.videoHeight; ctx.clearRect(0, 0, canvas.width, canvas.height); };
$('maskFile').onchange = () => {
  const file = $('maskFile').files[0], png = file && (file.type === 'image/png' || file.name.toLowerCase().endsWith('.png'));
  $('maskImage').hidden = !png; $('maskVideo').hidden = !file || png;
  const target = png ? $('maskImage') : $('maskVideo'); target.src = replaceURL('mask', file);
};
function maskMode() {
  const paint = $('maskMode').value === 'paint';
  $('paintControls').hidden = !paint; $('uploadControls').hidden = paint;
  canvas.hidden = !paint; $('maskFile').required = !paint;
  video.controls = !paint;
  canvas.style.pointerEvents = paint ? 'auto' : 'none';
  if (paint) video.pause();
}
$('maskMode').onchange = maskMode; maskMode();
$('seek').oninput = () => { if(Number.isFinite(video.duration)) video.currentTime=Number($('seek').value)*video.duration; };
function point(event) { const box = canvas.getBoundingClientRect(); return [(event.clientX-box.left)*canvas.width/box.width, (event.clientY-box.top)*canvas.height/box.height]; }
function stroke(p) {
  ctx.globalCompositeOperation = $('eraser').checked ? 'destination-out' : 'source-over';
  ctx.strokeStyle = 'white'; ctx.fillStyle = 'white'; ctx.lineWidth = Number($('brush').value); ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(...(lastPoint || p)); ctx.lineTo(...p); ctx.stroke();
  if (!lastPoint) { ctx.beginPath(); ctx.arc(...p, ctx.lineWidth/2, 0, Math.PI*2); ctx.fill(); }
  lastPoint = p; painted = true;
}
canvas.onpointerdown = e => { if (!video.videoWidth) return; drawing = true; lastPoint = null; canvas.setPointerCapture(e.pointerId); stroke(point(e)); };
canvas.onpointermove = e => { if (drawing) stroke(point(e)); };
canvas.onpointerup = canvas.onpointercancel = () => { drawing = false; lastPoint = null; };
$('clear').onclick = () => { ctx.clearRect(0,0,canvas.width,canvas.height); painted = false; };
const notes = {
  quality: '使用 121 帧窗口，缓存关闭。显存预算与硬件加速由服务启动配置决定。',
  memory: '使用 65 帧窗口，缓存关闭；缩短窗口会改变结果和窗口衔接。服务仍需启用卸载与显存预算。',
  speed: '使用 TeaCache、mask 逐帧局部保护与 0.1 阈值。这是近似策略，请检查擦除区域与闪烁。'
};
$('preset').onchange = () => { $('presetNote').textContent = notes[$('preset').value]; };
async function request(url, options) {
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw Error(data.error?.message || `HTTP ${response.status}`);
  return data;
}
function upload(form) {
  return new Promise((resolve,reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('POST', '/v1/videos');
    xhr.upload.onprogress = e => { if(e.lengthComputable) $('uploadProgress').value = 100*e.loaded/e.total; };
    xhr.onerror = () => reject(Error('上传连接中断，请刷新任务列表确认是否已创建任务。'));
    xhr.onload = () => { try { const data = JSON.parse(xhr.responseText); if(xhr.status >= 200 && xhr.status < 300) resolve(data); else reject(Error(data.error?.message || `HTTP ${xhr.status}`)); } catch { reject(Error('服务返回了无法解析的响应。')); } };
    xhr.send(form);
  });
}
$('form').onsubmit = async e => {
  e.preventDefault(); if (uploading) return;
  uploading = true; $('submit').disabled = true;
  try {
    const file = $('videoFile').files[0]; if (!file) throw Error('请先选择视频。');
    let mask = $('maskFile').files[0];
    if ($('maskMode').value === 'paint') {
      if (!painted || !canvas.width) throw Error('请先在画面上绘制擦除区域。');
      const flat = document.createElement('canvas'); flat.width=canvas.width; flat.height=canvas.height;
      const c = flat.getContext('2d'); c.fillStyle='black'; c.fillRect(0,0,flat.width,flat.height); c.drawImage(canvas,0,0);
      mask = await new Promise(resolve => flat.toBlob(resolve, 'image/png'));
    }
    if (!mask) throw Error('请选择 mask 文件。');
    const mode = $('preset').value;
    const params = {prompt: $('prompt').value, seed: Number($('seed').value), num_inference_steps: Number($('steps').value), strength: Number($('strength').value), streaming_cache_dtype:'uint8', infer_len: mode === 'memory' ? 65 : 121, transformer_cache_mode:mode === 'speed' ? 'teacache' : 'off', cache_probe_metric:'mask_frame_max', teacache_threshold:0.1};
    const form = new FormData(); form.append('video',file); form.append('mask',mask,mask.name || 'paint.png'); form.append('parameters',JSON.stringify(params));
    $('uploadProgress').hidden=false; $('uploadProgress').value=0; message('正在上传并准备输入…');
    const task = await upload(form); message(`任务已提交：${task.id}`); await refresh();
  } catch(error) { message(error.message); }
  finally { uploading=false; $('submit').disabled=false; $('uploadProgress').hidden=true; }
};
const phases = {queued:'等待开始',preparing:'准备输入',processing:'擦除中',finalizing:'生成结果',terminal:'已结束'};
const statuses = {queued:'排队中',running:'处理中',completed:'已完成',failed:'失败',cancelled:'已取消'};
let refreshing=false;
async function refresh() {
  if (refreshing) return; refreshing=true;
  try {
    const {data} = await request('/v1/videos?limit=20'); $('taskError').textContent='';
    // Update existing cards so polling does not reset result video playback.
    const ids = new Set(data.map(t=>t.id));
    for(const child of [...$('tasks').children]) if(!ids.has(child.dataset.id)) child.remove();
    let previous=null;
    for(const task of data) {
      let card = [...$('tasks').children].find(c=>c.dataset.id===task.id);
      if(!card) { card=document.createElement('div'); card.className='task'; card.dataset.id=task.id; card.innerHTML='<div class="taskTop"><span></span><button type="button">取消任务</button></div><progress max="100"></progress><p></p><div class="result"></div>'; if(previous) previous.after(card); else $('tasks').prepend(card); }
      previous=card;
      card.querySelector('span').textContent=`${statuses[task.status]} · ${task.progress}% · ${task.id}`;
      card.querySelector('progress').value=task.progress;
      card.querySelector('p').textContent=task.error?.message || (task.queue_position != null ? `队列位置：${task.queue_position}` : `${phases[task.phase] || task.phase}${task.window_count ? ` · 窗口 ${task.window_index == null ? 0 : task.window_index + 1} / ${task.window_count}` : ''}`);
      const button=card.querySelector('button'); button.hidden=!['queued','running'].includes(task.status);
      button.onclick=async()=>{button.disabled=true;try{await request(`/v1/videos/${encodeURIComponent(task.id)}`,{method:'DELETE'});await refresh();}catch(e){$('taskError').textContent=e.message;}finally{button.disabled=false;}};
      const result=card.querySelector('.result');
      if(task.status==='completed' && !result.children.length) {
        const source=task.content_url || task.url; if(!source) continue;
        const url=new URL(source,location.href); if(!['http:','https:'].includes(url.protocol)) continue;
        const a=document.createElement('a'); a.href=url.href; a.textContent='下载结果'; a.download=`${task.id}.mp4`; a.rel='noopener'; result.append(a);
        const v=document.createElement('video');v.controls=true;v.preload='none';v.src=url.href;result.append(v);
      }
    }
    if(!data.length) $('taskError').textContent='暂无任务。';
  } catch(error) {$('taskError').textContent=error.message;} finally{refreshing=false;}
}
$('refresh').onclick=refresh;
async function health() { try{await request('/health');$('health').textContent='服务就绪';}catch{$('health').textContent='服务尚未就绪';} }
async function capabilities() {
  try {
    const info=await request('/server_info'), modes=info.effective_acceleration?.transformer_cache?.supported_modes;
    if(modes && !modes.includes('teacache')) {
      $('preset').querySelector('[value="speed"]').disabled=true;
      if($('preset').value==='speed') { $('preset').value='quality'; $('preset').onchange(); }
    }
  } catch { /* Readiness polling reports an unavailable service. */ }
}
capabilities();refresh();health();setInterval(()=>{if(!document.hidden){refresh();health();}},2500);
