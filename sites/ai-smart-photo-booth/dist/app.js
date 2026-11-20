import { FaceTracker } from "./ar/faceTracker.js?v=20261120-2";
import { chooseFilter, isUsableProfile, manualProfile } from "./ar/filterPolicy.js?v=20261120-3";
import { GENDER_FILTERS } from "./ar/filters/expressionMatrix.js";
import { FilterRenderer } from "./ar/filterRenderer.js";
import { createFusionState, updateExpressionFusion } from "./ar/expressionFusion.js?v=20261120-2";
import { getFrameLayout } from "./frameLayouts.js";
import { AmbientEffects } from "./ar/ambientEffects.js?v=20261119-2";

const THEMES = {
  cool: { name:"Ngầu", icon:"⚡", className:"theme-cool", filterClass:"filter-cool", desc:"Tông xanh điện, tương phản rõ và thần thái tự tin.", color:"#46d7d2", caption:"OWN THE MOMENT", canvasFilter:"contrast(1.16) saturate(1.28) hue-rotate(8deg)", stickers:["⚡","✦","◆"], challenges:[{emotion:"neutral",emoji:"😎",text:"Giữ ánh mắt thật ngầu — bình thản nhé!"},{emotion:"surprise",emoji:"🤩",text:"Bật mode ngôi sao — ngạc nhiên một chút!"},{emotion:"happy",emoji:"😁",text:"Chốt bằng một nụ cười đầy tự tin!"}] },
  birthday: { name:"Sinh nhật", icon:"🎈", className:"theme-birthday", filterClass:"filter-birthday", desc:"Màu ấm rực rỡ, bóng bay và confetti vui vẻ.", color:"#ff7166", caption:"HAPPY BIRTHDAY!", canvasFilter:"brightness(1.06) saturate(1.38) contrast(1.04)", stickers:["🎈","★","🎉"], challenges:[{emotion:"happy",emoji:"🥳",text:"Cười thật tươi như đang thổi nến!"},{emotion:"surprise",emoji:"🎁",text:"Biểu cảm khi mở món quà bất ngờ!"},{emotion:"happy",emoji:"🎉",text:"Cười lớn — bữa tiệc bắt đầu rồi!"}] },
  horror: { name:"Kinh dị", icon:"👻", className:"theme-horror", filterClass:"filter-horror", desc:"Khói tím, ánh đỏ và hiệu ứng huyền bí kiểu Halloween.", color:"#9b4dff", caption:"SCARY NIGHT", canvasFilter:"contrast(1.22) saturate(.82) brightness(.86) hue-rotate(8deg)", stickers:["👻","🕸","⚡"], challenges:[{emotion:"neutral",emoji:"🧛",text:"Giữ gương mặt bí ẩn trong màn đêm!"},{emotion:"surprise",emoji:"👻",text:"Biểu cảm khi bóng ma bất ngờ xuất hiện!"},{emotion:"angry",emoji:"😈",text:"Thử ánh mắt dữ dội kiểu điện ảnh!"}] },
  christmas: { name:"Giáng sinh", icon:"🎄", className:"theme-christmas", filterClass:"filter-christmas", desc:"Tuyết trắng, sắc đỏ xanh và ánh đèn mùa lễ hội.", color:"#24a66a", caption:"MERRY CHRISTMAS", canvasFilter:"brightness(1.06) saturate(1.2) contrast(1.04)", stickers:["🎄","❄","🎁"], challenges:[{emotion:"happy",emoji:"🎅",text:"Cười thật tươi để đón Giáng sinh!"},{emotion:"surprise",emoji:"🎁",text:"Biểu cảm khi mở món quà bất ngờ!"},{emotion:"neutral",emoji:"❄️",text:"Giữ vẻ bình yên giữa trời tuyết nhé!"}] }
};
const THEME_IDS=["cool","birthday","horror","christmas"];

const EMOTIONS={angry:"Tức giận",happy:"Vui",neutral:"Bình thản",sad:"Buồn",surprise:"Ngạc nhiên"};
const FILTER_EXPRESSIONS=["happy","surprise","neutral","sad","angry"];
const GENDERS={male:"Nam",female:"Nữ"};
const EMPTY={detected:false,hasFace:false,emotion:null,probability:0,ready:false,gender:null,genderProbability:0,status:null,quality:{light:0,sharp:0,size:0,angle:0},message:"Hãy bật camera"};
const app={screen:"welcome",filterChoice:"",selectedTheme:"cool",effectTheme:"cool",userChoice:null,shotIndex:0,shots:[],frameBackground:{color:null,dataUrl:null,image:null},cameraStream:null,analysisRunId:0,analysisBusy:false,aiState:{...EMPTY},apiOnline:false,arTracker:null,lastFace:null,fusionState:createFusionState(),fusionResult:null,modelToken:0};
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)], wait=ms=>new Promise(r=>setTimeout(r,ms));
const ambientEffects=new AmbientEffects($("#ambientCanvas"));
const labelEmotion=e=>EMOTIONS[e]||"";

function suggestPackage(face=app.aiState){
  const theme=app.userChoice&&THEME_IDS.includes(app.userChoice)?app.userChoice:"cool";
  return {theme,reason:"Filter sẽ đổi theo nét mặt của bạn khi chụp"};
}

function applyFilter(theme){
  const stage=$("#cameraStage"); if(!stage)return;
  stage.className=`camera-stage ${theme.filterClass}`;
}

function applyTheme(key,reason=""){
  if(!THEMES[key])return;
  app.selectedTheme=key;app.effectTheme=key; const t=THEMES[key]; applyFilter(t);
  ambientEffects.setTheme(key);
  $("#recommendCard").className=`recommend-card ${t.className}`;
  $("#recommendName").textContent=t.name; $("#recommendDesc").textContent=t.desc;
  $("#recommendReason").textContent=reason||suggestPackage().reason; $("#previewCaption").textContent=t.caption;
  $("#challengePreview").innerHTML=FILTER_EXPRESSIONS.map(expression=>`<span>${labelEmotion(expression)}</span>`).join("");
  $("#framePreview").style.boxShadow=`0 18px 40px ${t.color}35`;
  renderThemeGrid();applyFrameBackground();if(app.screen==="capture")renderCapture();
}

function renderThemeGrid(){
  const grid=$("#themeGrid"); if(!grid)return;
  grid.innerHTML=THEME_IDS.map(key=>[key,THEMES[key]]).map(([key,t])=>`<button class="theme-tile ${key===app.selectedTheme?"selected":""}" data-theme="${key}"><span class="theme-icon" style="background:${t.color}22;color:${t.color}">${t.icon}</span><b>${t.name}</b><small>${t.desc}</small></button>`).join("");
  grid.querySelectorAll("[data-theme]").forEach(btn=>btn.addEventListener("click",()=>{app.userChoice=btn.dataset.theme;applyTheme(btn.dataset.theme,"Bạn đã tự chọn phong cách khung ảnh này");}));
}

function refreshSuggestion(){const s=suggestPackage();applyTheme(s.theme,s.reason);}
function goTo(screen){
  if(screen!=="capture"&&app.cameraStream) shutdownCamera(false);
  app.screen=screen; $$(".screen").forEach(e=>e.classList.toggle("active-screen",e.id===screen));
  const steps={welcome:1,themes:2,capture:3,results:4}; $$(".step").forEach(e=>e.classList.toggle("active",+e.dataset.step===steps[screen]));
  if(screen==="themes")refreshSuggestion();
  if(screen==="capture"){renderCapture();if(!app.cameraStream)setTimeout(toggleCamera,50);}
  if(screen==="results")renderResults(); window.scrollTo({top:0,behavior:"smooth"});
}

function manualFilterName(){const [gender,expression]=app.filterChoice.split(":");return GENDER_FILTERS[gender]?.[expression]?.name||"";}

function renderArFrame(face,time){
  app.lastFace=face;
  if(app.filterChoice){
    const profile=manualProfile(app.filterChoice,face,app.selectedTheme);
    if(profile.ready!==app.aiState.ready){app.aiState={...EMPTY,...profile,message:profile.ready?"Đang dùng filter tự chọn":"Không thấy khuôn mặt"};renderQuality();}
    app.filterRenderer?.setSelection(chooseFilter(profile),time);
    app.filterRenderer?.draw(face,time,profile.detected);
    $("#faceGuide").style.opacity=face?"0":"1";
    return;
  }
  const fused=updateExpressionFusion(app.fusionState,{probs:app.aiState.probs||{},modelToken:app.modelToken},time);
  app.fusionState=fused.state;app.fusionResult=fused.result;app.aiState.expression=fused.result.label;app.aiState.emotion=fused.result.label;app.aiState.fusedProbability=fused.result.score;
  const profile={...app.aiState,themeId:app.selectedTheme,expression:fused.result.label};
  const visible=isUsableProfile(profile);
  app.filterRenderer?.setSelection(chooseFilter(profile),time);
  app.filterRenderer?.draw(face,time,visible);
  $("#faceGuide").style.opacity=face&&visible?"0":"1";
}

async function resetServerSession(){
  try{await fetch("/api/session/start",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});}catch{}
}

async function startArTracking(){
  try{
    app.arTracker ||= new FaceTracker($("#cameraVideo"),$("#arCanvas"),{ema:.42,maxFps:30});
    await app.arTracker.init();
    app.filterRenderer ||= new FilterRenderer($("#arCanvas"));
    app.arTracker.start(renderArFrame);
  }catch(error){
    console.error(error);toast("Không tải được filter. Hãy tải lại trang.");
  }
}

function renderCapture(){
  const t=THEMES[app.selectedTheme], challenge=t.challenges[app.shotIndex]||t.challenges[0]; applyFilter(THEMES[app.effectTheme||app.selectedTheme]);
  $("#activeThemeBadge").innerHTML=`<span>${t.icon}</span><div><b>${t.name}</b><small>${t.desc}</small></div>`;
  $("#challengeNumber").textContent=`${Math.min(app.shotIndex+1,3)}/3`; $("#progressBar").style.width=`${Math.min(app.shotIndex+1,3)/3*100}%`;
  $("#challengeEmoji").textContent=challenge.emoji; $("#challengeText").textContent=challenge.text;
  $("#shotList").innerHTML=t.challenges.map((c,i)=>`<div class="shot-item ${i===app.shotIndex?"active":""} ${i<app.shotIndex?"done":""}"><i>${i<app.shotIndex?"✓":i+1}</i><span>${c.text}</span><b>${c.emoji}</b></div>`).join(""); renderQuality();
}

function renderQuality(){
  const s=app.aiState;
  const m=$("#qualityMessage"),title=document.createElement("b"),detail=document.createElement("span");
  m.className=`quality-message ${s.ready?"good":"bad"}`;title.textContent=s.ready?"✓ Sẵn sàng chụp":"Chưa chụp được";detail.textContent=s.ready?(app.filterChoice?`Filter: ${manualFilterName()}`:"Bấm Chụp ngay khi bạn thích"):(s.message||"Hãy nhìn vào camera");m.replaceChildren(title,detail);
  $("#faceGuide").style.borderColor=s.ready?"#42d7da":"#ff9b62"; $("#captureNow").disabled=!s.ready||!app.cameraStream;
}

async function checkApi(){try{const r=await fetch("/api/health",{cache:"no-store"});if(!r.ok)throw 0;app.apiOnline=!!(await r.json()).ok;}catch{app.apiOnline=false;toast("Chưa kết nối được máy chủ. Hãy chạy lại run.bat.");}}
function frameData(){const v=$("#cameraVideo");if(!v.videoWidth)return null;const scale=Math.min(1,480/v.videoWidth),c=document.createElement("canvas");c.width=Math.round(v.videoWidth*scale);c.height=Math.round(v.videoHeight*scale);c.getContext("2d").drawImage(v,0,0,c.width,c.height);return c.toDataURL("image/jpeg",.7);}

async function analyzeFrame(runId=app.analysisRunId){
  if(!app.cameraStream||app.analysisBusy)return;const image=frameData();if(!image)return;app.analysisBusy=true;
  try{const r=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image})});const d=await r.json();if(!r.ok||!d.ok)throw new Error(d.error||"Máy chủ không phản hồi");app.apiOnline=true;
    if(runId!==app.analysisRunId)return;
    app.modelToken+=1;
    if(!d.detected)app.aiState={...EMPTY,probs:d.probs||{},latencyMs:d.latencyMs,modelEmotion:null,message:d.message||"Không tìm thấy khuôn mặt"};
    else app.aiState={...d,modelEmotion:d.emotion,expression:app.fusionResult?.label||"neutral",hasFace:true,quality:d.quality||EMPTY.quality,message:d.message||"Đang phân tích"};
  }catch(e){app.apiOnline=false;app.aiState={...EMPTY,message:"Mất kết nối, đang thử lại"};}
  finally{app.analysisBusy=false;renderQuality();}
}
async function startAnalysis(){
  stopAnalysis();if(app.filterChoice)return;const runId=app.analysisRunId;
  while(app.cameraStream&&runId===app.analysisRunId){const started=performance.now();await analyzeFrame(runId);const remaining=250-(performance.now()-started);if(remaining>0)await wait(remaining);}
}
function stopAnalysis(){app.analysisRunId+=1;}

// Mặt hoạt hình chỉ có các điểm mốc mà filter dùng (mũi 1, trán 10, đuôi mắt 33/263, gò má 50/280).
function previewFace(w,h){
  const u=w*.22,cx=w/2,cy=h*.6,p=(x,y)=>({x:cx+x*u,y:cy+y*u}),landmarks=Array.from({length:478},()=>p(0,0));
  Object.assign(landmarks,{1:p(0,0),10:p(0,-1.45),33:p(-.5,-.45),263:p(.5,-.45),50:p(-.45,.25),280:p(.45,.25)});
  return {landmarks,eyeDistance:u,roll:0,yaw:0,pitch:0,width:w,height:h,center:p(0,.2)};
}
function drawFilterPreview(canvas,filter,auto=false){
  const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,face=previewFace(w,h),u=face.eyeDistance,n=face.landmarks[1];
  ctx.fillStyle="#2a2342";ctx.fillRect(0,0,w,h);
  ctx.fillStyle="#f3c9a6";ctx.beginPath();ctx.ellipse(n.x,n.y-u*.25,u*1.05,u*1.35,0,0,Math.PI*2);ctx.fill();
  ctx.fillStyle="#3b2b25";for(const side of [-1,1]){ctx.beginPath();ctx.arc(n.x+side*u*.42,n.y-u*.45,u*.09,0,Math.PI*2);ctx.fill();}
  ctx.strokeStyle="#3b2b25";ctx.lineWidth=u*.07;ctx.lineCap="round";ctx.beginPath();ctx.arc(n.x,n.y+u*.2,u*.32,.2*Math.PI,.8*Math.PI);ctx.stroke();
  if(filter)filter.draw(ctx,face,1600);
  if(auto){ctx.font=`${Math.round(h*.2)}px system-ui`;ctx.textAlign="center";ctx.fillText("😁 😮 😢",w/2,h*.95);}
}
function renderFilterChoices(){
  const options=[["","Tự đổi theo nét mặt",null],...Object.entries(GENDERS).flatMap(([gender,label])=>[[null,label],...FILTER_EXPRESSIONS.map(expression=>[`${gender}:${expression}`,GENDER_FILTERS[gender][expression].name,GENDER_FILTERS[gender][expression]])])];
  const box=$("#filterChoices");
  box.innerHTML=options.map(([value,name])=>value===null?`<span class="filter-group-label">${name}</span>`:`<button type="button" class="filter-option ${value?"":"auto"}" data-filter="${value}" aria-pressed="false"><canvas width="192" height="150" aria-hidden="true"></canvas><b>${name}</b></button>`).join("");
  const filters=options.filter(([value])=>value!==null);
  box.querySelectorAll(".filter-option").forEach((button,i)=>{drawFilterPreview(button.querySelector("canvas"),filters[i][2],!filters[i][0]);button.addEventListener("click",()=>setFilterChoice(button.dataset.filter));});
  markFilterChoice();
}
function markFilterChoice(){$$("#filterChoices .filter-option").forEach(button=>{const active=button.dataset.filter===app.filterChoice;button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active));});}

// Chọn filter bằng tay thì ngừng gửi ảnh đi nhận diện; chọn lại "Tự động" thì nhận diện tiếp.
function setFilterChoice(value){
  app.filterChoice=value;app.aiState={...EMPTY,message:app.cameraStream?"Đang chờ khuôn mặt":EMPTY.message};app.fusionState=createFusionState();app.fusionResult=null;
  if(value)stopAnalysis();else if(app.cameraStream)void startAnalysis();
  markFilterChoice();renderQuality();
}
function shutdownCamera(resetState=true){
  stopAnalysis();
  app.arTracker?.stop();
  ambientEffects.stop();
  if(app.cameraStream)app.cameraStream.getTracks().forEach(t=>t.stop());
  app.cameraStream=null;
  const video=$("#cameraVideo");video.srcObject=null;video.style.display="none";$("#faceGuide").style.opacity="1";
  $("#mockScene").style.display="block";$("#cameraBtn").textContent="Bật camera";$("#cameraStatus").innerHTML="<i></i> Camera đang tắt";
  if(resetState)app.aiState={...EMPTY};
  renderQuality();
}

async function toggleCamera(){
  if(app.cameraStream){shutdownCamera();return;}
  try{app.cameraStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"},audio:false});const v=$("#cameraVideo");v.srcObject=app.cameraStream;await v.play();v.style.display="block";$("#mockScene").style.display="none";$("#cameraBtn").textContent="Tắt camera";$("#cameraStatus").innerHTML="<i style='background:#42d7da'></i> Camera đang bật";app.fusionState=createFusionState();app.fusionResult=null;app.modelToken=0;ambientEffects.setTheme(app.selectedTheme);ambientEffects.start();await resetServerSession();startAnalysis();startArTracking();}catch{app.cameraStream=null;toast("Không mở được camera. Hãy cho phép trình duyệt sử dụng webcam.");}
}

function loadImage(src){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=src;});}
async function captureFrame(){
  const v=$("#cameraVideo");if(!v.videoWidth)return null;
  const t=THEMES[app.effectTheme||app.selectedTheme],c=document.createElement("canvas"),sourceAspect=v.videoWidth/v.videoHeight;
  c.width=1280;c.height=Math.round(c.width/sourceAspect);
  const x=c.getContext("2d");
  x.save();x.translate(c.width,0);x.scale(-1,1);x.filter=t.canvasFilter;x.drawImage(v,0,0,v.videoWidth,v.videoHeight,0,0,c.width,c.height);x.restore();x.filter="none";
  const arCanvas=$("#arCanvas");if(arCanvas.width&&arCanvas.height)x.drawImage(arCanvas,0,0,arCanvas.width,arCanvas.height,0,0,c.width,c.height);
  const ambient=$("#ambientCanvas");if(ambient.width&&ambient.height)x.drawImage(ambient,0,0,ambient.width,ambient.height,0,0,c.width,c.height);
  return c.toDataURL("image/png");
}

async function runCountdown(){const s=app.aiState;if(!s.ready||!app.cameraStream){toast(s.message||"Hãy nhìn vào camera");return;}$("#captureNow").disabled=true;for(const n of [3,2,1]){$("#countdown").textContent=n;await wait(650)}$("#countdown").textContent="";$("#flash").classList.add("fire");const image=await captureFrame();await wait(380);$("#flash").classList.remove("fire");const challenge=THEMES[app.selectedTheme].challenges[app.shotIndex];app.shots.push({image,theme:app.selectedTheme,challenge:{...challenge},ai:{...s},time:new Date()});app.shotIndex++;if(app.shotIndex>=3)goTo("results");else{renderCapture();toast("Đã chụp và giữ filter! Tiếp tục ảnh tiếp theo ✦");}}

function renderResults(){stopAnalysis();const themeId=app.effectTheme||app.selectedTheme,t=THEMES[themeId],strip=$("#finalStrip");strip.className=`final-strip frame-${themeId}`;strip.dataset.title=t.caption;strip.innerHTML=app.shots.map((s,i)=>`<article class="result-shot"><button class="retake-one" data-retake="${i}" title="Chụp lại ảnh ${i+1}">↻</button><img src="${s.image}" alt="Ảnh Photo Booth ${i+1}"></article>`).join("");applyFrameBackground();$("#summaryTheme").innerHTML=`<span>${t.icon}</span><div><b>${t.name}</b><small>${t.caption}</small></div>`;$$('[data-retake]').forEach(b=>b.addEventListener("click",()=>{app.shotIndex=+b.dataset.retake;app.shots=app.shots.slice(0,app.shotIndex);goTo("capture");}));}
function drawCover(ctx,img,x,y,w,h){const s=Math.max(w/img.width,h/img.height),sw=w/s,sh=h/s;ctx.drawImage(img,(img.width-sw)/2,(img.height-sh)/2,sw,sh,x,y,w,h);}
function drawContain(ctx,img,x,y,w,h){const s=Math.min(w/img.width,h/img.height),dw=img.width*s,dh=img.height*s;ctx.drawImage(img,x+(w-dw)/2,y+(h-dh)/2,dw,dh);}
function applyFrameBackground(){const themeId=app.effectTheme||app.selectedTheme,layout=getFrameLayout(themeId),strip=$("#finalStrip"),preview=$("#frameBgPreview"),color=app.frameBackground.color||layout.background,image=app.frameBackground.dataUrl?`url("${app.frameBackground.dataUrl}")`:"";if(strip){strip.style.backgroundColor=color;strip.style.backgroundImage=image;strip.style.backgroundSize=image?"cover":"";strip.style.backgroundPosition=image?"center":"";}if(preview){preview.style.backgroundColor=color;preview.style.backgroundImage=image;preview.style.backgroundSize=image?"cover":"";preview.style.backgroundPosition=image?"center":"";}}
function drawFrameBackground(ctx,layout){
  ctx.fillStyle=app.frameBackground.color||layout.background;ctx.fillRect(0,0,layout.width,layout.height);
  if(app.frameBackground.image)drawCover(ctx,app.frameBackground.image,0,0,layout.width,layout.height);
  ctx.save();ctx.globalAlpha=.16;ctx.strokeStyle=layout.accent;ctx.lineWidth=3;
  if(layout.id==="cool"){for(let x=0;x<layout.width;x+=75){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,layout.height);ctx.stroke();}for(let y=0;y<layout.height;y+=75){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(layout.width,y);ctx.stroke();}}
  else if(layout.id==="birthday"){for(let i=0;i<70;i++){ctx.fillStyle=i%2?layout.accent:"#ffffff";ctx.fillRect((i*137)%layout.width,(i*83)%layout.height,12+(i%3)*7,5+(i%2)*5);}}
  else if(layout.id==="horror"){for(let r=80;r<700;r+=70){ctx.beginPath();ctx.arc(0,0,r,0,Math.PI*.5);ctx.stroke();ctx.beginPath();ctx.arc(layout.width,0,r,Math.PI*.5,Math.PI);ctx.stroke();}}
  else {for(let i=0;i<90;i++){const px=(i*149)%layout.width,py=(i*89)%layout.height;ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(px,py,2+(i%4),0,Math.PI*2);ctx.fill();}}
  ctx.restore();
}
function drawFramedPhoto(ctx,img,photo,layout){
  ctx.save();ctx.translate(photo.x+photo.w/2,photo.y+photo.h/2);ctx.rotate(photo.rotation||0);ctx.shadowColor="rgba(0,0,0,.3)";ctx.shadowBlur=22;ctx.fillStyle=layout.id==="cool"?layout.accent:"#fffaf0";ctx.fillRect(-photo.w/2-12,-photo.h/2-12,photo.w+24,photo.h+24);ctx.shadowColor="transparent";ctx.beginPath();ctx.roundRect(-photo.w/2,-photo.h/2,photo.w,photo.h,layout.id==="christmas"?28:12);ctx.clip();ctx.fillStyle="#171320";ctx.fillRect(-photo.w/2,-photo.h/2,photo.w,photo.h);drawContain(ctx,img,-photo.w/2,-photo.h/2,photo.w,photo.h);ctx.restore();
}
function drawFrameTitle(ctx,layout){ctx.save();ctx.textAlign="center";ctx.font=`900 ${layout.id==="christmas"?58:48}px Arial, sans-serif`;ctx.fillStyle=layout.accent;ctx.shadowColor="rgba(0,0,0,.25)";ctx.shadowBlur=10;ctx.fillText(layout.title,layout.width/2,layout.height-42);ctx.restore();}
async function downloadStrip(){if(!app.shots.length){toast("Chưa có ảnh để tải.");return;}const themeId=app.effectTheme||app.selectedTheme,layout=getFrameLayout(themeId),c=document.createElement("canvas");c.width=layout.width;c.height=layout.height;const x=c.getContext("2d"),imgs=await Promise.all(app.shots.map(s=>new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=s.image;})));drawFrameBackground(x,layout);layout.photos.forEach((photo,i)=>drawFramedPhoto(x,imgs[i%imgs.length],photo,layout));drawFrameTitle(x,layout);const a=document.createElement("a");a.download=`ai-photo-booth-${themeId}-${Date.now()}.png`;a.href=c.toDataURL("image/png");a.click();}
function resetSession(){app.shots=[];app.shotIndex=0;app.userChoice=null;app.frameBackground={color:null,dataUrl:null,image:null};app.selectedTheme="cool";app.effectTheme="cool";app.fusionState=createFusionState();app.fusionResult=null;resetServerSession();refreshSuggestion();}
function toast(msg){const e=$("#toast");e.textContent=msg;e.classList.add("show");clearTimeout(toast.t);toast.t=setTimeout(()=>e.classList.remove("show"),2800);}
function setFrameBackgroundColor(color){app.frameBackground={color,dataUrl:null,image:null};applyFrameBackground();$("#frameBgColor").value=color;$("#frameBgFeedback").textContent="Đã đổi màu nền khung.";}
async function setFrameBackgroundImage(file){
  if(!file)return;
  if(!["image/png","image/jpeg","image/webp"].includes(file.type)||file.size>8*1024*1024){$("#frameBgFeedback").textContent="Chỉ nhận PNG/JPG/WebP tối đa 8 MB.";return;}
  const dataUrl=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});
  const image=await loadImage(dataUrl);app.frameBackground={color:null,dataUrl,image};applyFrameBackground();$("#frameBgFeedback").textContent=`Đã dùng ảnh nền: ${file.name}`;
}
$("#startBtn").addEventListener("click",()=>goTo("themes"));$$('[data-go]').forEach(b=>b.addEventListener("click",()=>{if(b.dataset.go==="welcome")resetSession();goTo(b.dataset.go);}));
$("#randomTheme").addEventListener("click",()=>{let next=app.selectedTheme;while(next===app.selectedTheme)next=THEME_IDS[Math.floor(Math.random()*THEME_IDS.length)];app.userChoice=next;applyTheme(next,"Gợi ý ngẫu nhiên — bạn có thể đổi tiếp");});
$$('[data-frame-color]').forEach(button=>button.addEventListener("click",()=>setFrameBackgroundColor(button.dataset.frameColor)));
$("#frameBgColor").addEventListener("input",event=>setFrameBackgroundColor(event.target.value));
$("#frameBgUpload").addEventListener("change",event=>void setFrameBackgroundImage(event.target.files?.[0]).catch(()=>{$("#frameBgFeedback").textContent="Không đọc được ảnh nền này.";}));
$("#resetFrameBg").addEventListener("click",()=>{app.frameBackground={color:null,dataUrl:null,image:null};applyFrameBackground();$("#frameBgFeedback").textContent="Đã dùng lại nền mặc định của chủ đề.";});
$("#useRecommendation").addEventListener("click",()=>{app.shotIndex=0;app.shots=[];goTo("capture");});$("#captureNow").addEventListener("click",runCountdown);$("#cameraBtn").addEventListener("click",toggleCamera);
$("#themeMode").addEventListener("click",()=>{document.body.classList.toggle("dark");$("#themeMode").textContent=document.body.classList.contains("dark")?"☀":"☾";});$("#retakeAll").addEventListener("click",()=>{app.shotIndex=0;app.shots=[];goTo("capture");});$("#deleteAll").addEventListener("click",()=>{resetSession();goTo("welcome");toast("Đã xóa ảnh trong phiên.");});$("#downloadMock").addEventListener("click",downloadStrip);
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&app.screen!=="welcome")goTo("welcome");if((e.key===" "||e.key==="Enter")&&app.screen==="capture"&&!/INPUT|SELECT|BUTTON/.test(e.target.tagName)){e.preventDefault();runCountdown();}});window.addEventListener("beforeunload",()=>{app.arTracker?.stop();app.cameraStream?.getTracks().forEach(t=>t.stop());});
renderFilterChoices();renderThemeGrid();refreshSuggestion();renderCapture();checkApi();
