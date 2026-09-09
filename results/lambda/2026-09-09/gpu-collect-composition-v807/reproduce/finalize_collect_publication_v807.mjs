import {readFile,writeFile,readdir,copyFile} from 'node:fs/promises';
import {join,relative} from 'node:path';
import {createHash} from 'node:crypto';
const sha=b=>createHash('sha256').update(b).digest('hex');
const root='results/lambda/2026-09-09/gpu-collect-composition-v807', prior='results/lambda/2026-09-09/gpu-collect-reuse-v806',viewer='results/lambda/2026-09-06/visualiser/data/gtoc12-collect-composition-v807';
const files={};
for(const name of await readdir(viewer)){
 const bytes=await readFile(join(viewer,name));files[name]=sha(bytes);
 if(name.startsWith('.'))continue;
 const response=await fetch('http://127.0.0.1:4173/data/gtoc12-collect-composition-v807/'+name);
 if(!response.ok||sha(Buffer.from(await response.arrayBuffer()))!==files[name])throw Error('HTTP bytes differ: '+name);
}
const manifest=JSON.parse(await readFile(join(viewer,'manifest.json'),'utf8'));
if(manifest.files['fleet.json'].sha256!==files['fleet.json']||manifest.summary.ships!==23||manifest.summary.unique_asteroids!==199)throw Error('Viewer binding');
await writeFile(join(root,'viewer-import.json'),JSON.stringify({files,summary:manifest.summary,kepler_check:manifest.kepler_check,url:'http://127.0.0.1:4173/?dataset=gtoc12-collect-composition-v807&epoch=69807&preset=oblique&z=1',visual_check:'Browser accessibility and screenshot confirm verified 23-ship/199-asteroid fleet, 12,999.825 weighted kg, 14,279.288 raw kg, physical Z scale 1 and active WebGL2. Served dataset bytes match local files.'},null,2));
for(const name of ['prepare_collect_viewer_v807.mjs','finalize_collect_publication_v807.mjs'])await copyFile('build/performance/'+name,join(root,'reproduce',name));
async function walk(dir){let out=[];for(const x of await readdir(dir,{withFileTypes:true})){const p=join(dir,x.name);if(x.isDirectory())out.push(...await walk(p));else out.push(p);}return out;}
for(const dir of [root,prior]){
 const hashes={};for(const p of (await walk(dir)).sort()){if(p===join(dir,'sha256.json'))continue;hashes[relative(dir,p).replaceAll('\\','/')]=sha(await readFile(p));}
 await writeFile(join(dir,'sha256.json'),JSON.stringify(hashes,null,2));
 console.log(dir,Object.keys(hashes).length,'hashed files');
}
const owned=JSON.parse(await readFile('build/performance/owned_collect_v807.json','utf8'));
for(const name of ['README.md','docs/GTOC12_PROGRESS.md',viewer])if(!owned.includes(name))owned.push(name);
await writeFile('build/performance/owned_collect_v807.json',JSON.stringify(owned));
