export const FRAME_LAYOUTS = Object.freeze({
  cool: Object.freeze({
    id:"cool", width:1500, height:1350, background:"#10142c", accent:"#55ecff", title:"OWN THE MOMENT",
    photos:Object.freeze([
      {x:110,y:80,w:1280,h:720,rotation:-.008,fit:"contain"},
      {x:90,y:855,w:620,h:349,rotation:-.018,fit:"contain"},
      {x:790,y:855,w:620,h:349,rotation:.018,fit:"contain"}
    ])
  }),
  birthday: Object.freeze({
    id:"birthday", width:1500, height:1350, background:"#ff9b91", accent:"#ffe064", title:"HAPPY BIRTHDAY!",
    photos:Object.freeze([
      {x:110,y:80,w:1280,h:720,rotation:-.012,fit:"contain"},
      {x:90,y:855,w:620,h:349,rotation:-.028,fit:"contain"},
      {x:790,y:855,w:620,h:349,rotation:.028,fit:"contain"}
    ])
  }),
  horror: Object.freeze({
    id:"horror", width:1500, height:1350, background:"#170d24", accent:"#b864ff", title:"SCARY NIGHT",
    photos:Object.freeze([
      {x:110,y:80,w:1280,h:720,rotation:0,fit:"contain"},
      {x:90,y:855,w:620,h:349,rotation:.012,fit:"contain"},
      {x:790,y:855,w:620,h:349,rotation:-.012,fit:"contain"}
    ])
  }),
  christmas: Object.freeze({
    id:"christmas", width:1500, height:1350, background:"#b5242d", accent:"#fff8e8", title:"MERRY CHRISTMAS",
    photos:Object.freeze([
      {x:110,y:80,w:1280,h:720,rotation:0,fit:"contain"},
      {x:90,y:855,w:620,h:349,rotation:-.012,fit:"contain"},
      {x:790,y:855,w:620,h:349,rotation:.012,fit:"contain"}
    ])
  })
});

export function getFrameLayout(themeId) {
  return FRAME_LAYOUTS[themeId] || FRAME_LAYOUTS.cool;
}
