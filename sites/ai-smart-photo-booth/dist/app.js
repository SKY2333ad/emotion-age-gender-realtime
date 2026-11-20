import { FaceTracker } from "./ar/faceTracker.js";
import { chooseFilter, isUsableProfile } from "./ar/filterPolicy.js";
import { FilterRenderer } from "./ar/filterRenderer.js";
import { createFusionState, updateExpressionFusion } from "./ar/expressionFusion.js";
import { DeepARAdapter, resolveCoolDeepAREffect, DEEPAR_LICENSE_STORAGE_KEY } from "./ar/deepARAdapter.js?v=20261119-6";
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
const app={screen:"welcome",selectedTheme:"cool",effectTheme:"cool",userChoice:null,shotIndex:0,shots:[],frameBackground:{color:null,dataUrl:null,image:null},virtualBackground:{url:null,name:"Nền thật",objectUrl:null},cameraStream:null,analysisRunId:0,analysisBusy:false,aiState:{...EMPTY},apiOnline:false,arTracker:null,lastFace:null,fusionState:createFusionState(),fusionResult:null,modelToken:0,deepAR:null,deepArEnabled:false,deepArBusy:false,deepArSwitchError:null};
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)], wait=ms=>new Promise(r=>setTimeout(r,ms));
const ambientEffects=new AmbientEffects($("#ambientCanvas"));
const labelEmotion=e=>EMOTIONS[e]||"Đang nhận diện";

function suggestPackage(face=app.aiState){
  const theme=app.userChoice&&THEME_IDS.includes(app.userChoice)?app.userChoice:"cool";
  return {theme,reason:"Chọn chủ đề; AI sẽ phối hiệu ứng theo biểu cảm khi camera bật"};
}

function applyFilter(theme){
  const stage=$("#cameraStage"); if(!stage)return;
  stage.className=`camera-stage ${theme.filterClass}`;
  if(app.deepArEnabled)stage.classList.add("deepar-active");
}

function applyTheme(key,reason=""){
  if(!THEMES[key])return;
  if(key!=="cool"&&app.deepArEnabled)void app.deepAR?.clearFaceEffect().catch(console.error);
  app.selectedTheme=key;app.effectTheme=key; const t=THEMES[key]; applyFilter(t);
  ambientEffects.setTheme(key);
  $("#recommendCard").className=`recommend-card ${t.className}`;
  $("#recommendName").textContent=t.name; $("#recommendDesc").textContent=t.desc;
  $("#recommendReason").textContent=reason||suggestPackage().reason; $("#previewCaption").textContent=t.caption;
  $("#challengePreview").innerHTML=FILTER_EXPRESSIONS.map(expression=>`<span>${labelEmotion(expression)}</span>`).join("");
  $("#framePreview").style.boxShadow=`0 18px 40px ${t.color}35`;
  renderThemeGrid();updateDeepArUi();applyFrameBackground();if(app.screen==="capture")renderCapture();
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

function renderArFrame(face,time){
  app.lastFace=face;
  const fused=updateExpressionFusion(app.fusionState,{
    probs:app.aiState.probs||{},modelLabel:app.aiState.modelEmotion,modelToken:app.modelToken,blendshapes:face?.blendshapes||{}
  },time);
  app.fusionState=fused.state;app.fusionResult=fused.result;app.aiState.expression=fused.result.label;app.aiState.emotion=fused.result.label;app.aiState.fusedProbability=fused.result.score;
  const profile={...app.aiState,themeId:app.selectedTheme,expression:fused.result.label};
  const visible=isUsableProfile(profile);
  if(app.deepArEnabled&&visible&&profile.themeId==="cool")syncDeepArProfile(profile);
  // Filter Canvas có sẵn, không cần key; nhường chỗ khi DeepAR đang vẽ filter mặt ở chủ đề Ngầu.
  app.filterRenderer?.setSelection(chooseFilter(profile),time);
  app.filterRenderer?.draw(face,time,visible&&!(app.deepArEnabled&&profile.themeId==="cool"));
  $("#faceGuide").style.opacity=face&&visible?"0":"1";
}

function updateDeepArUi(message=""){
  const button=$("#deepArToggleBtn"),status=$("#deepArStatus");
  if(!button||!status)return;
  const setup=location.hash==="#setup";
  $("#advancedTools").hidden=!(setup||storedDeepArKey());
  $("#deepArKeyBtn").hidden=!setup;
  button.disabled=app.deepArBusy||app.selectedTheme!=="cool";
  button.classList.toggle("active",app.deepArEnabled);
  button.textContent=app.deepArBusy?"Đang tải...":app.deepArEnabled?"Tắt filter nâng cao":"Bật filter nâng cao";
  status.classList.toggle("ready",app.deepArEnabled);
  status.textContent=message||(app.deepArEnabled?(app.virtualBackground.url?`Đang dùng nền: ${app.virtualBackground.name}`:"Filter nâng cao đang bật"):"Thêm kính, mũ, mặt nạ cho chủ đề Ngầu");
}

function storedDeepArKey(){
  try{return localStorage.getItem(DEEPAR_LICENSE_STORAGE_KEY)||"";}catch{return "";}
}

function saveDeepArKey(licenseKey){
  try{localStorage.setItem(DEEPAR_LICENSE_STORAGE_KEY,licenseKey);return localStorage.getItem(DEEPAR_LICENSE_STORAGE_KEY)===licenseKey;}catch{return false;}
}

function manageDeepArKey(){
  const hasKey=Boolean(storedDeepArKey());
  const next=window.prompt(hasKey?"License Key đã được lưu. Dán key mới để thay đổi, hoặc để trống rồi bấm OK để xóa key đã lưu:":"Dán DeepAR SDK License Key để lưu lâu dài trong trình duyệt này:","");
  if(next===null)return;
  const value=next.trim();
  if(!value){try{localStorage.removeItem(DEEPAR_LICENSE_STORAGE_KEY);}catch{}disableDeepAR();updateDeepArUi("Đã xóa License Key khỏi trình duyệt");toast("Đã xóa DeepAR License Key.");return;}
  if(saveDeepArKey(value)){updateDeepArUi("Đã lưu License Key lâu dài trên trình duyệt này");toast("Đã lưu DeepAR License Key.");}
  else toast("Trình duyệt đang chặn lưu key. Hãy bật quyền lưu dữ liệu trang web.");
}

function syncDeepArProfile(profile){
  const effect=resolveCoolDeepAREffect(profile);
  if(!effect||!app.deepArEnabled||app.deepArBusy||app.deepArSwitchError===effect.key)return;
  if(app.deepAR?.activeEffect?.key===effect.key||app.deepAR?.requestedEffect?.key===effect.key)return;
  updateDeepArUi(`Đang đổi filter: ${effect.name}...`);
  void app.deepAR.setProfile(profile).then(changed=>{
    if(changed){
      app.deepArSwitchError=null;
      const active=app.deepAR.activeEffect||effect;
      updateDeepArUi(`Filter: ${active.name}`);
    }
  }).catch(error=>{
    app.deepArSwitchError=effect.key;
    console.error("DeepAR effect switch failed",error);
    updateDeepArUi(`Không đổi được filter ${effect.name}`);
    setTimeout(()=>{if(app.deepArSwitchError===effect.key)app.deepArSwitchError=null;},1800);
  });
}

function disableDeepAR(){
  app.deepAR?.pause();
  app.deepArEnabled=false;
  app.deepArSwitchError=null;
  $("#cameraStage")?.classList.remove("deepar-active");
  updateDeepArUi();
}

async function enableDeepAR({askForKey=true}={}){
  if(!app.cameraStream){toast("Hãy bật camera trước.");return false;}
  if(app.deepArBusy)return false;
  let licenseKey=storedDeepArKey();
  if(!licenseKey&&askForKey){
    licenseKey=window.prompt("Dán DeepAR SDK License Key. Key chỉ được lưu trong trình duyệt này:","")?.trim()||"";
    if(licenseKey&&!saveDeepArKey(licenseKey))toast("Không lưu được key vì trình duyệt đang chặn dữ liệu trang web.");
  }
  if(!licenseKey){updateDeepArUi("Chưa cài License Key cho filter nâng cao");return false;}
  app.deepArBusy=true;updateDeepArUi("Đang tải filter nâng cao...");
  try{
    app.deepAR ||= new DeepARAdapter($("#cameraVideo"),$("#deepArCanvas"));
    await app.deepAR.enable(licenseKey);
    app.deepArEnabled=true;
    app.deepArSwitchError=null;
    applyFilter(THEMES[app.selectedTheme]);
    if(app.selectedTheme==="cool"){
      const profile={...app.aiState,themeId:"cool",expression:app.fusionResult?.label||"neutral"};
      syncDeepArProfile(profile);
    }else await app.deepAR.clearFaceEffect();
    if(app.virtualBackground.url)await app.deepAR.setBackground(app.virtualBackground.url);
    toast(app.virtualBackground.url?"Đã bật nền phía sau.":"Đã bật filter nâng cao.");
    return true;
  }catch(error){
    console.error("DeepAR init failed",error);
    app.deepArEnabled=false;
    $("#cameraStage")?.classList.remove("deepar-active");
    updateDeepArUi("Chưa tải được, bấm lại để thử");
    toast("Không bật được filter nâng cao. Hãy thử lại.");
    return false;
  }finally{app.deepArBusy=false;updateDeepArUi($("#deepArStatus")?.textContent||"");}
}

async function toggleDeepAR(){
  if(app.deepArEnabled){disableDeepAR();toast("Đã tắt filter nâng cao.");return;}
  await enableDeepAR();
}

function updateVirtualBackgroundUi(){
  $$('[data-camera-background]').forEach(button=>{const active=(button.dataset.cameraBackground||"")===(app.virtualBackground.objectUrl?"__custom__":app.virtualBackground.url||"");button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active));});
  const feedback=$("#cameraBgFeedback");if(feedback)feedback.textContent=app.virtualBackground.url?`Đang dùng nền phía sau: ${app.virtualBackground.name}.`:"Đang giữ nguyên căn phòng phía sau.";
}

async function setVirtualBackground(url,name="Nền tùy chọn",objectUrl=null){
  if(!url){
    await app.deepAR?.clearBackground();
    if(app.virtualBackground.objectUrl)URL.revokeObjectURL(app.virtualBackground.objectUrl);
    app.virtualBackground={url:null,name:"Nền thật",objectUrl:null};updateVirtualBackgroundUi();updateDeepArUi();toast("Đã dùng lại nền thật.");return;
  }
  if(!app.cameraStream){toast("Hãy bật camera trước khi chọn nền phía sau.");return;}
  const ready=app.deepArEnabled||await enableDeepAR();if(!ready){if(objectUrl)URL.revokeObjectURL(objectUrl);return;}
  try{
    await app.deepAR.setBackground(url);
    if(app.virtualBackground.objectUrl&&app.virtualBackground.objectUrl!==objectUrl)URL.revokeObjectURL(app.virtualBackground.objectUrl);
    app.virtualBackground={url,name,objectUrl};updateVirtualBackgroundUi();updateDeepArUi();toast(`Đã thay nền phía sau: ${name}.`);
  }catch(error){if(objectUrl)URL.revokeObjectURL(objectUrl);console.error("DeepAR background replacement failed",error);toast("Không thay được nền này. Hãy thử ảnh khác.");}
}

async function setUploadedVirtualBackground(file){
  if(!file)return;
  if(!["image/png","image/jpeg","image/webp"].includes(file.type)||file.size>8*1024*1024){$("#cameraBgFeedback").textContent="Chỉ nhận PNG/JPG/WebP tối đa 8 MB.";return;}
  const objectUrl=URL.createObjectURL(file);await setVirtualBackground(objectUrl,file.name,objectUrl);
  $("#cameraBgUpload").value="";
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
    if(app.selectedTheme==="cool"&&storedDeepArKey())void enableDeepAR({askForKey:false});
  }catch(error){
    console.error(error);toast("Không tải được bộ nhận diện khuôn mặt. Hãy tải lại trang.");
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
  const s=app.aiState,q=s.quality||EMPTY.quality;
  const items=[["Ánh sáng",q.light,q.light>.5?"Tốt":"Cần sáng hơn"],["Độ nét",q.sharp,q.sharp>.5?"Rõ":"Giữ ổn định"],["Kích thước",q.size,s.hasFace?"Đã thấy mặt":"Không thấy mặt"],["Góc mặt",q.angle,q.angle>.5?"Chính diện":"Nhìn thẳng hơn"]];
  $("#qualityList").innerHTML=items.map(([name,value,label])=>`<div class="quality-item ${value>.5?"":"warn"}"><i>${value>.5?"✓":"!"}</i><div><b>${name}</b><small>${label}</small></div></div>`).join("");
  const m=$("#qualityMessage"),title=document.createElement("b"),detail=document.createElement("span");
  m.className=`quality-message ${s.ready?"good":"bad"}`;title.textContent=s.ready?"✓ Sẵn sàng chụp":"! Chưa thể chụp";detail.textContent=s.ready?labelEmotion(s.expression||s.emotion):(s.message||"Đang chờ kết quả AI");m.replaceChildren(title,detail);
  $("#faceGuide").style.borderColor=s.ready?"#42d7da":"#ff9b62"; $("#captureNow").disabled=!s.ready||!app.cameraStream;
}

function updateProfile(){
  const s=app.aiState,genderReliable=s.genderProbability>=.75;
  const badge=$("#liveAiBadge"),dot=document.createElement("i"),text=document.createElement("span");badge.className=`live-ai-badge ${s.ready?"ready":""}`;text.textContent=s.hasFace?`${labelEmotion(s.expression||s.emotion)}${genderReliable?` • ${GENDERS[s.gender]}`:""}`:s.message;badge.replaceChildren(dot,text);
}

async function checkApi(){try{const r=await fetch("/api/health",{cache:"no-store"});if(!r.ok)throw 0;app.apiOnline=!!(await r.json()).ok;}catch{app.apiOnline=false;toast("Chưa kết nối được máy chủ AI. Hãy chạy lại run_photo_booth.bat.");}updateProfile();}
function frameData(){const v=$("#cameraVideo");if(!v.videoWidth)return null;const scale=Math.min(1,480/v.videoWidth),c=document.createElement("canvas");c.width=Math.round(v.videoWidth*scale);c.height=Math.round(v.videoHeight*scale);c.getContext("2d").drawImage(v,0,0,c.width,c.height);return c.toDataURL("image/jpeg",.7);}

async function analyzeFrame(runId=app.analysisRunId){
  if(!app.cameraStream||app.analysisBusy)return;const image=frameData();if(!image)return;app.analysisBusy=true;
  try{const r=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image})});const d=await r.json();if(!r.ok||!d.ok)throw new Error(d.error||"AI không phản hồi");app.apiOnline=true;
    if(runId!==app.analysisRunId)return;
    app.modelToken+=1;
    if(!d.detected)app.aiState={...EMPTY,probs:d.probs||{},latencyMs:d.latencyMs,modelEmotion:null,message:d.message||"Không tìm thấy khuôn mặt"};
    else app.aiState={...d,modelEmotion:d.emotion,expression:app.fusionResult?.label||"neutral",hasFace:true,quality:d.quality||EMPTY.quality,message:d.message||"Đang phân tích"};
  }catch(e){app.apiOnline=false;app.aiState={...EMPTY,message:e.message||"Không kết nối được AI"};}
  finally{app.analysisBusy=false;updateProfile();renderQuality();}
}
async function startAnalysis(){
  stopAnalysis();const runId=app.analysisRunId;
  while(app.cameraStream&&runId===app.analysisRunId){const started=performance.now();await analyzeFrame(runId);const remaining=250-(performance.now()-started);if(remaining>0)await wait(remaining);}
}
function stopAnalysis(){app.analysisRunId+=1;}
function shutdownCamera(resetState=true){
  stopAnalysis();
  app.arTracker?.stop();
  ambientEffects.stop();
  disableDeepAR();
  if(app.cameraStream)app.cameraStream.getTracks().forEach(t=>t.stop());
  app.cameraStream=null;
  const video=$("#cameraVideo");video.srcObject=null;video.style.display="none";$("#faceGuide").style.opacity="1";
  $("#mockScene").style.display="block";$("#cameraBtn").textContent="Bật camera thật";$("#cameraStatus").innerHTML="<i></i> Camera đang tắt";
  if(resetState)app.aiState={...EMPTY};
  updateProfile();renderQuality();
}

async function toggleCamera(){
  if(app.cameraStream){shutdownCamera();return;}
  try{app.cameraStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"},audio:false});const v=$("#cameraVideo");v.srcObject=app.cameraStream;await v.play();v.style.display="block";$("#mockScene").style.display="none";$("#cameraBtn").textContent="Tắt camera";$("#cameraStatus").innerHTML="<i style='background:#42d7da'></i> Camera + AI đang hoạt động";app.fusionState=createFusionState();app.fusionResult=null;app.modelToken=0;ambientEffects.setTheme(app.selectedTheme);ambientEffects.start();await resetServerSession();startAnalysis();startArTracking();}catch{app.cameraStream=null;toast("Không mở được camera. Hãy cho phép trình duyệt sử dụng webcam.");}
}

function loadImage(src){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=src;});}
async function captureFrame(){
  const v=$("#cameraVideo");if(!v.videoWidth)return null;
  const t=THEMES[app.effectTheme||app.selectedTheme],c=document.createElement("canvas"),sourceAspect=v.videoWidth/v.videoHeight;
  c.width=1280;c.height=Math.round(c.width/sourceAspect);
  const x=c.getContext("2d");
  let deepArImage=null;
  if(app.deepArEnabled){try{const shot=await app.deepAR?.takeScreenshot();if(shot)deepArImage=await loadImage(shot);}catch(error){console.error("DeepAR capture failed",error);}}
  if(deepArImage){
    x.drawImage(deepArImage,0,0,deepArImage.width,deepArImage.height,0,0,c.width,c.height);
  }else{
    x.save();x.translate(c.width,0);x.scale(-1,1);x.filter=t.canvasFilter;x.drawImage(v,0,0,v.videoWidth,v.videoHeight,0,0,c.width,c.height);x.restore();x.filter="none";
  }
  const arCanvas=$("#arCanvas");if(arCanvas.width&&arCanvas.height)x.drawImage(arCanvas,0,0,arCanvas.width,arCanvas.height,0,0,c.width,c.height);
  const ambient=$("#ambientCanvas");if(ambient.width&&ambient.height)x.drawImage(ambient,0,0,ambient.width,ambient.height,0,0,c.width,c.height);
  return c.toDataURL("image/png");
}

async function runCountdown(){const s=app.aiState;if(!s.ready||!app.cameraStream){toast(s.message||"Hãy chờ AI nhận diện khuôn mặt");return;}$("#captureNow").disabled=true;for(const n of [3,2,1]){$("#countdown").textContent=n;await wait(650)}$("#countdown").textContent="";$("#flash").classList.add("fire");const image=await captureFrame();await wait(380);$("#flash").classList.remove("fire");const challenge=THEMES[app.selectedTheme].challenges[app.shotIndex];app.shots.push({image,theme:app.selectedTheme,challenge:{...challenge},ai:{...s},time:new Date()});app.shotIndex++;if(app.shotIndex>=3)goTo("results");else{renderCapture();toast("Đã chụp và giữ filter! Tiếp tục ảnh tiếp theo ✦");}}

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
function resetSession(){app.shots=[];app.shotIndex=0;app.userChoice=null;app.frameBackground={color:null,dataUrl:null,image:null};void setVirtualBackground(null);app.selectedTheme="cool";app.effectTheme="cool";app.fusionState=createFusionState();app.fusionResult=null;resetServerSession();refreshSuggestion();}
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
$("#deepArToggleBtn").addEventListener("click",toggleDeepAR);
$("#deepArKeyBtn").addEventListener("click",manageDeepArKey);
window.addEventListener("hashchange",()=>updateDeepArUi());
$$('[data-camera-background]').forEach(button=>button.addEventListener("click",()=>void setVirtualBackground(button.dataset.cameraBackground||null,button.querySelector("b")?.textContent||"Nền có sẵn")));
$("#cameraBgUpload").addEventListener("change",event=>void setUploadedVirtualBackground(event.target.files?.[0]));
$$('[data-frame-color]').forEach(button=>button.addEventListener("click",()=>setFrameBackgroundColor(button.dataset.frameColor)));
$("#frameBgColor").addEventListener("input",event=>setFrameBackgroundColor(event.target.value));
$("#frameBgUpload").addEventListener("change",event=>void setFrameBackgroundImage(event.target.files?.[0]).catch(()=>{$("#frameBgFeedback").textContent="Không đọc được ảnh nền này.";}));
$("#resetFrameBg").addEventListener("click",()=>{app.frameBackground={color:null,dataUrl:null,image:null};applyFrameBackground();$("#frameBgFeedback").textContent="Đã dùng lại nền mặc định của chủ đề.";});
$("#useRecommendation").addEventListener("click",()=>{app.shotIndex=0;app.shots=[];goTo("capture");});$("#captureNow").addEventListener("click",runCountdown);$("#cameraBtn").addEventListener("click",toggleCamera);
$("#themeMode").addEventListener("click",()=>{document.body.classList.toggle("dark");$("#themeMode").textContent=document.body.classList.contains("dark")?"☀":"☾";});$("#retakeAll").addEventListener("click",()=>{app.shotIndex=0;app.shots=[];goTo("capture");});$("#deleteAll").addEventListener("click",()=>{resetSession();goTo("welcome");toast("Đã xóa ảnh trong phiên.");});$("#downloadMock").addEventListener("click",downloadStrip);
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&app.screen!=="welcome")goTo("welcome");if((e.key===" "||e.key==="Enter")&&app.screen==="capture"&&!/INPUT|SELECT|BUTTON/.test(e.target.tagName)){e.preventDefault();runCountdown();}});window.addEventListener("beforeunload",()=>{app.arTracker?.stop();app.cameraStream?.getTracks().forEach(t=>t.stop());});
window.addEventListener("beforeunload",()=>{if(app.virtualBackground.objectUrl)URL.revokeObjectURL(app.virtualBackground.objectUrl);app.deepAR?.shutdown();});
renderThemeGrid();refreshSuggestion();renderCapture();updateProfile();updateDeepArUi();updateVirtualBackgroundUi();checkApi();
