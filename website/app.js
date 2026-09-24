const $=id=>document.getElementById(id);
let items=[],current=null,playing=false,raf=0,pendingSeek=0;
const src=$('source'),res=$('result');
const names={source:'机器人原视频',result:'本例默认结果',optimized:'扩大掩码 · A',alternate:'扩大掩码 · B',optimizedRaw:'候选 A · 原始输出',alternateRaw:'候选 B · 原始输出',optimizedMask:'扩大的编辑区域',baseline:'原版 1.3B',larger:'14B 对照',baselineRaw:'原版 1.3B · 原始',largerRaw:'14B · 原始',mask:'原编辑区域',reference:'人手参考图'};
function pause(){src.pause();res.pause();playing=false;$('play').textContent='▶ 播放';cancelAnimationFrame(raf)}
function duration(){return Number.isFinite(src.duration)?src.duration:(current?.duration||0)}
function update(){const t=src.currentTime||0,d=duration();$('seek').value=d?t/d:0;$('clock').textContent=`${t.toFixed(2)} / ${d.toFixed(2)} s`;if(playing){if(!res.hidden&&res.readyState>=2&&Math.abs(res.currentTime-t)>.10)res.currentTime=Math.min(t,res.duration||t);raf=requestAnimationFrame(update)}}
function seek(t){const target=Math.max(0,Math.min(duration(),t));if(src.readyState>=1)src.currentTime=target;if(res.src&&res.readyState>=1)res.currentTime=Math.min(target,res.duration||target);update()}
async function play(){if(!current?.source)return;if(playing)return pause();if(src.currentTime>=duration()-.03)seek(0);try{await src.play();playing=true;$('play').textContent='Ⅱ 暂停';if(!res.hidden&&res.src){res.currentTime=src.currentTime;res.play().catch(()=>{});}update()}catch(e){showError('视频暂时无法播放，请检查文件是否已加载。')}}
function showError(message){$('load-error').hidden=false;$('load-error').textContent=message}
function showView(){
 pause();res.removeAttribute('src');res.load();res.hidden=true;$('reference').hidden=true;$('empty').hidden=true;
 const kind=$('view').value,file=current?.[kind];$('download').hidden=!file;
 if(file){$('download').href=file;$('download').textContent=kind==='reference'?'下载当前参考图 ↗':kind.toLowerCase().includes('mask')?'下载当前编辑区域 ↗':'下载当前生成视频 ↗'}
 $('result-title').textContent=kind==='result'?(current?.resultLabel||names.result):names[kind];
 $('variant-note').textContent=current?.variantNotes?.[kind]||current?.variantNotes?.default||'';
 if(!file){$('empty').hidden=false;return}
 if(kind==='reference'){$('reference').src=file;$('reference').hidden=false}else{res.src=file;res.hidden=false;res.onloadedmetadata=()=>{res.currentTime=Math.min(src.currentTime,res.duration);res.playbackRate=Number($('speed').value)}}
}
function showLeftView(){pause();pendingSeek=src.currentTime||0;const kind=$('left-view').value;$('left-title').textContent=names[kind];src.poster=kind==='source'?(current.poster||''):'';src.src=current[kind];src.load()}
function downloadLink(id,file){const el=$(id);if(!el)return;el.hidden=!file;if(file)el.href=file;else el.removeAttribute('href')}
function select(i){
 pause();$('load-error').hidden=true;current=items[i];
 $('compare-models').disabled=!current.baseline||!current.larger;$('compare-edit').disabled=!current.baseline||!current.optimized;$('compare-seeds').disabled=!current.optimized||!current.alternate;
 document.querySelectorAll('.sample').forEach((e,j)=>{e.classList.toggle('active',i===j);e.setAttribute('aria-current',i===j?'true':'false')});
 $('sample-index').textContent=`SCENE ${String(i+1).padStart(2,'0')} / ${String(items.length).padStart(2,'0')}`;
 $('title').textContent=current.title;$('instruction').textContent=current.instruction;$('status').textContent=current.status||'待核验';$('source-spec').textContent=current.sourceSpec||'';$('review').textContent=current.review||'';
 $('checks').replaceChildren(...(current.checks||[]).map(x=>{const li=document.createElement('li');li.textContent=x;return li}));
 $('facts').replaceChildren();for(const [k,v] of Object.entries(current.facts||{})){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;$('facts').append(dt,dd)}
 pendingSeek=0;$('left-view').value='source';$('left-title').textContent=names.source;
 for(const option of $('left-view').options)option.disabled=!current[option.value];
 src.src=current.source;src.poster=current.poster||'';src.load();
 downloadLink('native-download',current.native);downloadLink('pair-download',current.paired);downloadLink('labels',current.labels);
 $('seek').value=0;$('view').value='result';for(const option of $('view').options)option.disabled=!current[option.value];
 $('zoom').value='1';resetFocus();showView();update();
}
function compare(left,right){$('left-view').value=left;showLeftView();$('view').value=right;showView()}
function zoom(){document.querySelectorAll('.screen video,.screen img').forEach(el=>{el.style.transform=`scale(${$('zoom').value})`});document.querySelectorAll('.screen').forEach(el=>el.classList.toggle('zoomed',Number($('zoom').value)>1))}
function setFocus(x,y){document.querySelectorAll('.screen video,.screen img').forEach(el=>el.style.transformOrigin=`${x}% ${y}%`)}
function resetFocus(){setFocus(50,50);zoom()}
document.querySelectorAll('.screen').forEach(el=>el.addEventListener('click',e=>{if(Number($('zoom').value)===1)return;const b=el.getBoundingClientRect();setFocus(Math.max(0,Math.min(100,(e.clientX-b.left)/b.width*100)),Math.max(0,Math.min(100,(e.clientY-b.top)/b.height*100)))}));
function setup(){
 $('zoom').onchange=zoom;$('reset-focus').onclick=resetFocus;
 $('compare-models').onclick=()=>compare('baseline','larger');$('compare-edit').onclick=()=>compare('baseline','optimized');$('compare-seeds').onclick=()=>compare('optimized','alternate');
 $('play').onclick=play;$('previous').onclick=()=>{pause();seek(src.currentTime-1/(current?.displayFps||16))};$('next').onclick=()=>{pause();seek(src.currentTime+1/(current?.displayFps||16))};
 $('seek').oninput=()=>{pause();seek(Number($('seek').value)*duration())};$('view').onchange=showView;$('left-view').onchange=showLeftView;$('speed').onchange=()=>{src.playbackRate=res.playbackRate=Number($('speed').value)};
 src.onloadedmetadata=()=>{src.playbackRate=Number($('speed').value);seek(Math.min(pendingSeek,duration()))};src.onended=()=>{pause();update()};src.onseeked=update;
 res.onerror=()=>{if(res.getAttribute('src'))showError('右侧视频加载失败，可切换其他对照内容或重新选择样例。')};
 document.addEventListener('keydown',e=>{if(/INPUT|SELECT|BUTTON/.test(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();play()}if(e.code==='ArrowRight'){e.preventDefault();$('next').click()}if(e.code==='ArrowLeft'){e.preventDefault();$('previous').click()}});
}
setup();
fetch('examples.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(data=>{
 items=data.examples;$('count').textContent=items.length;
 const clips=items.reduce((n,x)=>n+['baseline','larger','optimized','alternate'].filter(k=>x[k]).length,0);
 $('gallery-stats').textContent=`${items.length} 个场景 · ${clips} 个生成版本 · 同步逐帧检查`;
 $('samples').replaceChildren(...items.map((x,i)=>{const b=document.createElement('button');b.className='sample';const im=document.createElement('img');im.src=x.poster;im.alt='';const title=document.createElement('span');title.textContent=`${String(i+1).padStart(2,'0')}  ${x.title}`;const sub=document.createElement('small');sub.textContent=x.sourceLabel||'BRIDGE · 机器人操作';b.append(im,title,sub);b.onclick=()=>select(i);return b}));if(items.length)select(Math.max(0,items.findIndex(x=>x.id===new URLSearchParams(location.search).get("scene"))))
}).catch(()=>{showError('样例清单加载失败，请稍后刷新。');$('title').textContent='样例暂不可用'});
