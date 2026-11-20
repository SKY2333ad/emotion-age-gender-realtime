const EFFECTS = Object.freeze({
  cool:{style:"cyber-neon",count:34,colors:["#19e7ff","#8f5cff","#ff3fd8","#ffffff"]},
  birthday:{style:"party-pop",count:40,colors:["#ffe064","#ff5e89","#3ee0c1","#7b68ff","#ffffff"]},
  horror:{style:"haunted-fog",count:26,colors:["#9d4edd","#ff315f","#d7b4ff","#ff9a3d"]},
  christmas:{style:"soft-snow",count:42,colors:["#ffffff","#dff7ff","#fff4c7"]}
});

export function effectForTheme(themeId){return EFFECTS[themeId]||EFFECTS.cool;}

export class AmbientEffects {
  constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext("2d");this.theme="cool";this.items=[];this.raf=0;this.running=false;this.last=0;this.reduceMotion=globalThis.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches||false;}

  setTheme(themeId){this.theme=EFFECTS[themeId]?themeId:"cool";this.items=[];}

  start(){if(this.running)return;this.running=true;this.last=performance.now();this.raf=requestAnimationFrame(time=>this.draw(time));}

  stop(){this.running=false;cancelAnimationFrame(this.raf);this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);}

  resize(){const box=this.canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,1.5),width=Math.max(1,Math.round(box.width*dpr)),height=Math.max(1,Math.round(box.height*dpr));if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;this.items=[];}}

  seed(){const config=effectForTheme(this.theme),w=this.canvas.width,h=this.canvas.height;this.items=Array.from({length:this.reduceMotion?Math.ceil(config.count/2):config.count},(_,i)=>{let x=Math.random()*w;if(this.theme==="cool"&&x>w*.3&&x<w*.7)x=Math.random()<.5?Math.random()*w*.25:w*(.75+Math.random()*.25);return{x,y:Math.random()*h,size:2+Math.random()*6,speed:.25+Math.random()*.7,drift:(Math.random()-.5)*.35,phase:Math.random()*Math.PI*2,color:config.colors[i%config.colors.length],type:i%4};});}

  draw(time){if(!this.running)return;this.resize();if(!this.items.length)this.seed();const ctx=this.ctx,w=this.canvas.width,h=this.canvas.height,dt=Math.min(2.2,(time-this.last)/16.67||1);this.last=time;ctx.clearRect(0,0,w,h);ctx.save();ctx.globalCompositeOperation="screen";
    if(this.theme==="cool")this.drawNeon(ctx,w,h,time,dt);
    else if(this.theme==="christmas")this.drawSnow(ctx,w,h,time,dt);
    else if(this.theme==="birthday")this.drawConfetti(ctx,w,h,time,dt);
    else this.drawEmbers(ctx,w,h,time,dt);
    ctx.restore();this.raf=requestAnimationFrame(next=>this.draw(next));
  }

  drawNeon(ctx,w,h,time,dt){
    const horizon=h*.72,shift=(time*.035)%34;ctx.save();ctx.lineWidth=1;ctx.shadowBlur=0;
    for(let y=horizon+shift;y<h;y+=34){ctx.globalAlpha=.08*(1-(y-horizon)/(h-horizon));ctx.strokeStyle="#20e7ff";ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
    for(let x=-w;x<w*2;x+=w/8){ctx.globalAlpha=.07;ctx.strokeStyle="#a45cff";ctx.beginPath();ctx.moveTo(w/2,horizon);ctx.lineTo(x,h);ctx.stroke();}
    const scan=(time*.08)%h;ctx.globalAlpha=.1;ctx.fillStyle="#35efff";ctx.fillRect(0,scan,w,1.5);ctx.restore();
    ctx.lineCap="round";for(const p of this.items){p.y+=p.speed*dt*2.1;p.x+=p.drift*dt;if(p.y>h+24){p.y=-24;p.x=Math.random()<.5?Math.random()*w*.25:w*(.75+Math.random()*.25);}ctx.globalAlpha=.24+.22*Math.sin(time*.003+p.phase);ctx.strokeStyle=p.color;ctx.shadowColor=p.color;ctx.shadowBlur=14;ctx.lineWidth=Math.max(1,p.size*.38);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(p.x+p.drift*24,p.y+14+p.size*2.4);ctx.stroke();}
    const pulse=(time%5200)/5200;if(pulse<.035&&!this.reduceMotion){const side=Math.floor(time/5200)%2?18:w-18;ctx.save();ctx.globalAlpha=(.035-pulse)*10;ctx.strokeStyle="#e9ffff";ctx.shadowColor="#25e9ff";ctx.shadowBlur=20;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(side,0);for(let y=0;y<h*.6;y+=h*.1)ctx.lineTo(side+(Math.random()-.5)*45,y);ctx.stroke();ctx.restore();}
  }

  drawSnow(ctx,w,h,time,dt){for(const p of this.items){p.y+=p.speed*dt;p.x+=Math.sin(time*.0015+p.phase)*.22*dt;if(p.y>h+p.size){p.y=-p.size;p.x=Math.random()*w;}ctx.globalAlpha=.45+.3*Math.sin(time*.001+p.phase);ctx.fillStyle=p.color;ctx.shadowColor="#fff";ctx.shadowBlur=5;ctx.beginPath();ctx.arc(p.x,p.y,p.size*.55,0,Math.PI*2);ctx.fill();}}

  drawConfetti(ctx,w,h,time,dt){
    const balloon=(x,y,color,flip=1)=>{ctx.save();ctx.globalAlpha=.22;ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=12;ctx.beginPath();ctx.ellipse(x,y,22,29,0,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle="rgba(255,255,255,.45)";ctx.beginPath();ctx.moveTo(x,y+28);ctx.bezierCurveTo(x+18*flip,y+52,x-12*flip,y+72,x+5*flip,y+98);ctx.stroke();ctx.restore();};
    balloon(30+Math.sin(time*.0008)*8,h*.25,"#ff5e89");balloon(w-34+Math.cos(time*.0007)*7,h*.38,"#ffe064",-1);
    for(const p of this.items){p.y+=p.speed*dt*1.45;p.x+=p.drift*dt+Math.sin(time*.002+p.phase)*.08;if(p.y>h+14){p.y=-14;p.x=Math.random()*w;}ctx.save();ctx.translate(p.x,p.y);ctx.rotate(time*.0025+p.phase);ctx.globalAlpha=.62;ctx.fillStyle=p.color;ctx.shadowColor=p.color;ctx.shadowBlur=3;if(p.type===0){ctx.beginPath();for(let k=0;k<10;k++){const r=k%2?p.size*.45:p.size,angle=-Math.PI/2+k*Math.PI/5;ctx.lineTo(Math.cos(angle)*r,Math.sin(angle)*r);}ctx.closePath();ctx.fill();}else if(p.type===1){ctx.beginPath();ctx.arc(0,0,p.size*.65,0,Math.PI*2);ctx.fill();}else if(p.type===2){ctx.fillRect(-p.size,-p.size*.28,p.size*2,p.size*.56);}else{ctx.strokeStyle=p.color;ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<4;k++)ctx.lineTo(k*p.size*.8-p.size*1.2,Math.sin(k*Math.PI)*p.size);ctx.stroke();}ctx.restore();}
  }

  drawEmbers(ctx,w,h,time,dt){
    ctx.save();for(let i=0;i<3;i++){const travel=((time*.018+i*w*.42)%(w*1.45))-w*.25,y=h*(.68+i*.1),gradient=ctx.createRadialGradient(travel,y,0,travel,y,w*.22);gradient.addColorStop(0,i===1?"rgba(112,31,155,.18)":"rgba(73,20,95,.15)");gradient.addColorStop(1,"rgba(30,0,45,0)");ctx.globalAlpha=.75;ctx.fillStyle=gradient;ctx.fillRect(0,h*.45,w,h*.55);}ctx.restore();
    for(const p of this.items){p.y-=p.speed*dt*.75;p.x+=Math.sin(time*.001+p.phase)*.2*dt;if(p.y<-12){p.y=h+12;p.x=Math.random()*w;}ctx.globalAlpha=.18+.28*Math.sin(time*.002+p.phase);ctx.fillStyle=p.type===0?"#ff693d":p.color;ctx.shadowColor=p.color;ctx.shadowBlur=14;ctx.beginPath();ctx.arc(p.x,p.y,p.size*(p.type===0?.34:.5),0,Math.PI*2);ctx.fill();}
    const flash=(time%7000)/7000;if(flash<.025&&!this.reduceMotion){ctx.save();ctx.globalAlpha=(.025-flash)*5;ctx.fillStyle="#c991ff";ctx.fillRect(0,0,w,h);ctx.restore();}
  }
}
