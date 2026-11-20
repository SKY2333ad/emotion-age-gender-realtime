const TAU = Math.PI * 2;

function withFace(ctx, face, callback, yOffset = 0) {
  const nose = face.landmarks[1];
  ctx.save();
  ctx.translate(nose.x, nose.y + yOffset);
  ctx.rotate(face.roll);
  callback(face.eyeDistance);
  ctx.restore();
}

function text(ctx, face, value, y, color = "white") {
  withFace(ctx, face, unit => {
    ctx.font = `900 ${Math.max(18, unit * .22)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.lineWidth = Math.max(4, unit * .035);
    ctx.strokeStyle = "rgba(40,25,58,.78)";
    ctx.strokeText(value, 0, y * unit);
    ctx.fillStyle = color;
    ctx.fillText(value, 0, y * unit);
  });
}

function glasses(ctx, face, { lift = 0, color = "#62efff" } = {}) {
  const eyeA=face.landmarks[33],eyeB=face.landmarks[263],unit=face.eyeDistance;
  const center={x:(eyeA.x+eyeB.x)/2,y:(eyeA.y+eyeB.y)/2};
  ctx.save();ctx.translate(center.x,center.y);ctx.rotate(face.roll);
  const y=-unit*lift,w=unit*.66,h=unit*.39;
  ctx.lineWidth=unit*.07;ctx.strokeStyle="#171321";ctx.lineJoin="round";
  for(const side of [-1,1]){
    const cx=side*unit*.36;
    ctx.fillStyle="rgba(26,34,48,.56)";ctx.beginPath();ctx.roundRect(cx-w/2,y-h/2,w,h,unit*.105);ctx.fill();ctx.stroke();
    const reflection=ctx.createLinearGradient(cx-w*.3,y-h*.45,cx+w*.3,y+h*.45);reflection.addColorStop(0,"rgba(255,255,255,.7)");reflection.addColorStop(.2,"rgba(255,255,255,.12)");reflection.addColorStop(.55,`${color}55`);reflection.addColorStop(1,"rgba(255,255,255,0)");ctx.fillStyle=reflection;ctx.fill();
  }
  ctx.beginPath();ctx.moveTo(-unit*.07,y);ctx.quadraticCurveTo(0,y-unit*.06,unit*.07,y);ctx.stroke();
  ctx.lineWidth=unit*.045;ctx.beginPath();ctx.moveTo(-unit*.69,y-unit*.03);ctx.lineTo(-unit*1.02,y-unit*.12);ctx.moveTo(unit*.69,y-unit*.03);ctx.lineTo(unit*1.02,y-unit*.12);ctx.stroke();ctx.restore();
}

function neon(ctx, face, time, color = "#54ecff", pulse = .5) {
  const nose=face.landmarks[1],unit=face.eyeDistance;ctx.save();ctx.translate(nose.x,nose.y);ctx.rotate(face.roll);ctx.globalAlpha*=.55+Math.sin(time*.008)*pulse*.25;ctx.strokeStyle=color;ctx.lineWidth=unit*.045;ctx.shadowColor=color;ctx.shadowBlur=unit*.22;ctx.beginPath();ctx.ellipse(0,unit*.05,unit*1.12,unit*1.38,0,0,TAU);ctx.stroke();ctx.restore();
}

function lightning(ctx, face, time, count = 3, color = "#fff36b", large = false) {
  const nose=face.landmarks[1],unit=face.eyeDistance;ctx.save();ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=unit*.12;
  for(let i=0;i<count;i++){const side=i%2?-1:1,x=nose.x+side*unit*(.9+(i%3)*.28),y=nose.y-unit*(.8-(i%2)*.55)+Math.sin(time*.01+i)*unit*.08,s=unit*(large?.32:.16);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-s*.28,y+s*.62);ctx.lineTo(x+s*.03,y+s*.52);ctx.lineTo(x-s*.18,y+s*1.18);ctx.lineTo(x+s*.48,y+s*.36);ctx.lineTo(x+s*.13,y+s*.4);ctx.closePath();ctx.fill();}
  ctx.restore();
}

function particles(ctx, face, time, type, color, count = 10, speed = 1, scale = 1) {
  const nose=face.landmarks[1],unit=face.eyeDistance;ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=Math.max(1.5,unit*.018);
  for(let i=0;i<count;i++){const phase=(time*.0005*speed+i*.618)%1,x=nose.x+Math.sin(i*7.7)*unit*1.25,y=nose.y-unit*1.35+phase*unit*2.8,size=unit*(.035+(i%3)*.018)*scale;ctx.globalAlpha=Math.min(1,(.25+(i%4)*.16)*(scale>1?1.4:1));if(type==="rain"){ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-unit*.06,y+unit*.22);ctx.stroke();}else if(type==="petal"){ctx.save();ctx.translate(x,y);ctx.rotate(time*.002+i);ctx.beginPath();ctx.ellipse(0,0,size,size*.5,0,0,TAU);ctx.fill();ctx.restore();}else{ctx.fillRect(x,y,size,size*.38);}}
  ctx.restore();
}

function hearts(ctx, face, time, count = 6, scale = 1) {
  const nose=face.landmarks[1],u=face.eyeDistance;ctx.save();ctx.fillStyle="#ff6fae";
  for(let i=0;i<count;i++){const a=time*.0007+i*1.7,x=nose.x+Math.cos(a)*u*(.8+i*.07),y=nose.y-u*.25+Math.sin(a*1.3)*u*.9,s=u*.08*scale;ctx.save();ctx.translate(x,y);ctx.rotate(-.7);ctx.beginPath();ctx.moveTo(0,s*.3);ctx.bezierCurveTo(-s,-s*.3,-s*.55,-s,0,-s*.45);ctx.bezierCurveTo(s*.55,-s,s,-s*.3,0,s*.3);ctx.fill();ctx.restore();}
  ctx.restore();
}

function stars(ctx, face, time, count = 10, color = "#ffe55e", scale = 1) {
  const nose=face.landmarks[1],u=face.eyeDistance;ctx.save();ctx.fillStyle=color;
  for(let i=0;i<count;i++){const a=i/count*TAU+time*.0004,r=u*(.95+(i%3)*.14),x=nose.x+Math.cos(a)*r,y=nose.y-u*.25+Math.sin(a)*r,s=u*(.035+(i%2)*.025)*scale;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.fillRect(-s*.18,-s,s*.36,s*2);ctx.fillRect(-s,-s*.18,s*2,s*.36);ctx.restore();}
  ctx.restore();
}

function blush(ctx, face, color = "rgba(255,90,115,.38)") {const u=face.eyeDistance;ctx.save();ctx.fillStyle=color;for(const index of [50,280]){const p=face.landmarks[index];ctx.beginPath();ctx.ellipse(p.x,p.y,u*.28,u*.13,face.roll,0,TAU);ctx.fill();}ctx.restore();}

function partyHat(ctx, face, time) {
  const p=face.landmarks[10],u=face.eyeDistance;
  ctx.save();
  ctx.translate(p.x,p.y-u*.08+face.pitch*u*.08);
  ctx.rotate(face.roll+Math.sin(time*.003)*.012+face.yaw*.04);
  ctx.scale(.82,.82);
  ctx.shadowColor="rgba(35,18,55,.45)";ctx.shadowBlur=u*.15;ctx.shadowOffsetY=u*.08;
  const cone=ctx.createLinearGradient(-u*.55,-u*.25,u*.35,-u*1.55);
  cone.addColorStop(0,"#7a28c6");cone.addColorStop(.45,"#ff49c6");cone.addColorStop(1,"#ff8adb");
  ctx.fillStyle=cone;ctx.beginPath();ctx.moveTo(u*.05,-u*1.58);ctx.quadraticCurveTo(-u*.12,-u*.82,-u*.67,0);ctx.quadraticCurveTo(0,u*.12,u*.67,0);ctx.closePath();ctx.fill();
  ctx.shadowColor="transparent";ctx.strokeStyle="rgba(255,255,255,.55)";ctx.lineWidth=u*.018;ctx.stroke();
  ctx.save();ctx.beginPath();ctx.moveTo(u*.05,-u*1.58);ctx.lineTo(-u*.67,0);ctx.lineTo(u*.67,0);ctx.closePath();ctx.clip();
  for(let i=0;i<13;i++){const x=Math.sin(i*12.73)*u*.5,y=-u*(.12+(i%6)*.24),r=u*(.025+(i%3)*.012);ctx.fillStyle=i%2?"rgba(255,232,110,.92)":"rgba(255,255,255,.78)";ctx.beginPath();ctx.arc(x,y,r,0,TAU);ctx.fill();}
  const shine=ctx.createLinearGradient(-u*.35,0,u*.35,0);shine.addColorStop(0,"rgba(255,255,255,0)");shine.addColorStop(.55,"rgba(255,255,255,.28)");shine.addColorStop(1,"rgba(255,255,255,0)");ctx.fillStyle=shine;ctx.fillRect(-u*.6,-u*1.6,u*1.2,u*1.7);ctx.restore();
  for(let i=0;i<18;i++){const a=i/17*Math.PI,x=Math.cos(a)*u*.65,y=Math.sin(a)*u*.06;ctx.fillStyle=i%2?"#d842d4":"#ff87ef";ctx.beginPath();ctx.arc(x,y,u*.075,0,TAU);ctx.fill();}
  for(let i=0;i<12;i++){const a=i/12*TAU,r=u*(.09+(i%2)*.025);ctx.fillStyle=i%2?"#ff86e8":"#c936cb";ctx.beginPath();ctx.arc(u*.05+Math.cos(a)*u*.075,-u*1.6+Math.sin(a)*u*.075,r,0,TAU);ctx.fill();}
  ctx.restore();
}

function birthdayBokeh(ctx, face, time) {
  const spots=[[-.03,.08,.105,0],[.92,.12,.075,1.3],[.08,.82,.13,2.1],[.84,.76,.11,3.2],[.55,.03,.052,4.1],[.98,.48,.07,5.2],[.22,.18,.045,2.8]];
  ctx.save();ctx.globalCompositeOperation="screen";
  for(const [nx,ny,size,phase] of spots){const drift=Math.sin(time*.00045+phase)*face.width*.012,x=nx*face.width+drift,y=ny*face.height+Math.cos(time*.00038+phase)*face.height*.012,r=size*Math.min(face.width,face.height);const glow=ctx.createRadialGradient(x,y,0,x,y,r);glow.addColorStop(0,"rgba(255,248,210,.72)");glow.addColorStop(.35,"rgba(255,220,170,.28)");glow.addColorStop(1,"rgba(255,220,170,0)");ctx.fillStyle=glow;ctx.beginPath();ctx.arc(x,y,r,0,TAU);ctx.fill();}
  ctx.restore();
}

function birthdayGrade(ctx, face) {
  const grade=ctx.createRadialGradient(face.width*.5,face.height*.42,face.width*.12,face.width*.5,face.height*.5,face.width*.72);
  grade.addColorStop(0,"rgba(255,196,130,.03)");grade.addColorStop(.66,"rgba(255,175,108,.07)");grade.addColorStop(1,"rgba(82,35,65,.16)");ctx.fillStyle=grade;ctx.fillRect(0,0,face.width,face.height);
}

function birthdayTitle(ctx, face) {
  const size=Math.max(24,Math.min(face.width,face.height)*.065),x=face.width*.5,y=face.height*.13;
  ctx.save();ctx.translate(x,y);ctx.rotate(-.035);ctx.textAlign="center";ctx.textBaseline="middle";ctx.font=`italic 950 ${size}px Impact, Arial Black, sans-serif`;ctx.lineJoin="round";ctx.lineWidth=size*.16;ctx.strokeStyle="rgba(82,42,40,.48)";ctx.strokeText("HAPPY BIRTHDAY",size*.05,size*.08);ctx.lineWidth=size*.08;ctx.strokeStyle="#fff3cf";ctx.strokeText("HAPPY BIRTHDAY",0,0);const fill=ctx.createLinearGradient(0,-size*.6,0,size*.6);fill.addColorStop(0,"#fffdf1");fill.addColorStop(.55,"#ffe69c");fill.addColorStop(1,"#ffc75a");ctx.fillStyle=fill;ctx.fillText("HAPPY BIRTHDAY",0,0);ctx.restore();
}

function flower(ctx, x, y, u, roll, time, size = 1) {ctx.save();ctx.translate(x,y);ctx.rotate(roll+time*.0003);for(let i=0;i<6;i++){ctx.rotate(TAU/6);ctx.fillStyle=i%2?"#ff95c5":"#ffe6f3";ctx.beginPath();ctx.ellipse(0,u*.12*size,u*.08*size,u*.18*size,0,0,TAU);ctx.fill();}ctx.fillStyle="#ffd85c";ctx.beginPath();ctx.arc(0,0,u*.08*size,0,TAU);ctx.fill();ctx.restore();}

function flowerCrown(ctx, face, time) {const p=face.landmarks[10],u=face.eyeDistance;for(let i=-2;i<=2;i++){const a=face.roll+i*.32;flower(ctx,p.x+Math.sin(a)*u*.95,p.y+u*.25-Math.cos(i*.32)*u*.42,u,face.roll,time,.62+(i%2?0:.12));}}

function headphones(ctx, face) {withFace(ctx,face,u=>{ctx.strokeStyle="#55e7ff";ctx.lineWidth=u*.09;ctx.beginPath();ctx.arc(0,-u*.1,u*.98,Math.PI*1.05,Math.PI*1.95);ctx.stroke();ctx.fillStyle="#5144a8";for(const side of [-1,1])ctx.fillRect(side*u*.96-u*.12,-u*.02,u*.24,u*.52);});}

// Không khí riêng của từng chủ đề; không phụ thuộc biểu cảm hay giới tính.
function birthdayScene(ctx, face, time) {birthdayGrade(ctx,face);partyHat(ctx,face,time);birthdayBokeh(ctx,face,time);birthdayTitle(ctx,face);}

function bats(ctx, face, time, count = 6) {
  const n=face.landmarks[1],u=face.eyeDistance;ctx.save();ctx.fillStyle="rgba(22,12,36,.88)";
  for(let i=0;i<count;i++){const a=time*.0007+i*1.31,x=n.x+Math.cos(a)*u*(1.05+(i%3)*.18),y=n.y-u*.45+Math.sin(a*1.4)*u*.9,s=u*(.09+(i%2)*.025);ctx.save();ctx.translate(x,y);ctx.rotate(Math.sin(a)*.25);ctx.beginPath();ctx.moveTo(0,s*.2);ctx.quadraticCurveTo(-s*.55,-s*.55,-s,0);ctx.quadraticCurveTo(-s*.48,-s*.15,0,s*.35);ctx.quadraticCurveTo(s*.48,-s*.15,s,0);ctx.quadraticCurveTo(s*.55,-s*.55,0,s*.2);ctx.fill();ctx.restore();}
  ctx.restore();
}

function horrorScene(ctx, face, time) {
  const grade=ctx.createRadialGradient(face.width*.5,face.height*.43,face.width*.08,face.width*.5,face.height*.5,face.width*.7);grade.addColorStop(0,"rgba(93,33,114,.03)");grade.addColorStop(.62,"rgba(50,9,70,.14)");grade.addColorStop(1,"rgba(12,3,20,.46)");ctx.fillStyle=grade;ctx.fillRect(0,0,face.width,face.height);
  bats(ctx,face,time,5);
}

function santaHat(ctx, face, time) {
  const p=face.landmarks[10],u=face.eyeDistance;ctx.save();ctx.translate(p.x,p.y-u*.05);ctx.rotate(face.roll+Math.sin(time*.002)*.012);ctx.fillStyle="#d92e3b";ctx.beginPath();ctx.moveTo(-u*.72,0);ctx.quadraticCurveTo(-u*.18,-u*1.35,u*.5,-u*1.18);ctx.quadraticCurveTo(u*.25,-u*.58,u*.62,-u*.24);ctx.lineTo(u*.5,-u*.05);ctx.closePath();ctx.fill();ctx.fillStyle="#fffaf1";ctx.beginPath();ctx.roundRect(-u*.76,-u*.14,u*1.42,u*.28,u*.12);ctx.fill();ctx.beginPath();ctx.arc(u*.56,-u*.18,u*.18,0,TAU);ctx.fill();ctx.restore();
}

function snow(ctx, face, time, count = 22) {
  const u=face.eyeDistance;ctx.save();ctx.fillStyle="rgba(255,255,255,.9)";ctx.shadowColor="#bdeaff";ctx.shadowBlur=u*.055;
  for(let i=0;i<count;i++){const phase=(time*.00008*(1+i%3*.25)+i*.083)%1,x=(.04+((i*47)%91)/100)*face.width+Math.sin(time*.001+i)*u*.08,y=phase*face.height,r=u*(.018+(i%4)*.009);ctx.globalAlpha=.45+(i%4)*.14;ctx.beginPath();ctx.arc(x,y,r,0,TAU);ctx.fill();}ctx.restore();
}

function christmasScene(ctx, face, time) {
  const glow=ctx.createRadialGradient(face.width*.5,face.height*.45,0,face.width*.5,face.height*.45,face.width*.65);glow.addColorStop(0,"rgba(255,245,190,.04)");glow.addColorStop(1,"rgba(12,91,57,.18)");ctx.fillStyle=glow;ctx.fillRect(0,0,face.width,face.height);
  santaHat(ctx,face,time);snow(ctx,face,time);text(ctx,face,"MERRY CHRISTMAS",-1.42,"#fff2bd");
}

const SCENES = { cool: () => {}, birthday: birthdayScene, horror: horrorScene, christmas: christmasScene };

// 5 cảm xúc × 2 giới tính: mỗi ô là một filter riêng bám theo khuôn mặt.
export const GENDER_FILTERS = Object.freeze({
  male: Object.freeze({
    happy:    { name: "Kính ngầu",     draw: (ctx, face, time) => { glasses(ctx,face); lightning(ctx,face,time,4); } },
    surprise: { name: "WOW tia sét",   draw: (ctx, face, time) => { glasses(ctx,face,{lift:.28}); lightning(ctx,face,time,5,"#fff05c",true); text(ctx,face,"WOW!",-1.15,"#fff05c"); } },
    neutral:  { name: "Tai nghe neon", draw: (ctx, face, time) => { headphones(ctx,face); neon(ctx,face,time,"#54ecff",.3); } },
    sad:      { name: "Mưa xanh",      draw: (ctx, face, time) => { neon(ctx,face,time,"#4169c9",.2); particles(ctx,face,time,"rain","#73cfff",16,.7,1.6); } },
    angry:    { name: "Sét đỏ",        draw: (ctx, face, time) => { neon(ctx,face,time,"#ff5b72",1); lightning(ctx,face,time,5,"#ff5b72",true); } }
  }),
  female: Object.freeze({
    happy:    { name: "Tim hồng",        draw: (ctx, face, time) => { blush(ctx,face,"rgba(255,80,130,.5)"); hearts(ctx,face,time,8,1.8); } },
    surprise: { name: "Lấp lánh OMG",    draw: (ctx, face, time) => { stars(ctx,face,time,14,"#ffca54",1.8); text(ctx,face,"OMG!",-1.15,"#ff8ad8"); } },
    neutral:  { name: "Vương miện hoa",  draw: (ctx, face, time) => { flowerCrown(ctx,face,time); stars(ctx,face,time,6,"#ffe6f3",1.4); } },
    sad:      { name: "Cánh hoa rơi",    draw: (ctx, face, time) => { particles(ctx,face,time,"petal","#ff8cc2",16,.6,2.2); text(ctx,face,"Cố lên nhé!",1.2,"#ffe3f1"); } },
    angry:    { name: "Neon hồng",       draw: (ctx, face, time) => { neon(ctx,face,time,"#ff4fa3",1); neon(ctx,face,time,"#ff9bd0",.6); lightning(ctx,face,time,5,"#ff5fb0",true); } }
  })
});

// Giới tính chưa đủ chắc chắn (dưới 0,75): chỉ viền màu theo cảm xúc, không đoán nam/nữ.
const NEUTRAL_COLORS = { happy: "#ffd84d", surprise: "#fff05c", neutral: "#54ecff", sad: "#4169c9", angry: "#ff5b72" };

export function genderFilterFor(gender, expression) {
  return GENDER_FILTERS[gender]?.[expression] || null;
}

export const expressionMatrix = {
  id:"expression-matrix",name:"Tổ hợp theo biểu cảm và giới tính",icon:"✦",tags:["ai","theme","expression","gender"],
  draw(ctx,face,time){
    const expression=face.expression||"neutral";
    ctx.save();
    (SCENES[face.theme]||SCENES.cool)(ctx,face,time);
    const filter=genderFilterFor(face.gender,expression);
    if(filter)filter.draw(ctx,face,time);
    else neon(ctx,face,time,NEUTRAL_COLORS[expression]||NEUTRAL_COLORS.neutral,.4);
    ctx.restore();
  }
};
