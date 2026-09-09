import {readFile,writeFile,readdir,copyFile} from 'node:fs/promises';
import {join,relative} from 'node:path';
import {createHash} from 'node:crypto';
const sha=b=>createHash('sha256').update(b).digest('hex');
const root='results/lambda/2026-09-09/gpu-catalogue-cache-v822',viewer='results/lambda/2026-09-06/visualiser/data/gtoc12-catalogue-v822';
const files={};
for(const name of await readdir(viewer)){
 const bytes=await readFile(join(viewer,name));files[name]=sha(bytes);
 if(name.startsWith('.'))continue;
 const response=await fetch('http://127.0.0.1:4173/data/gtoc12-catalogue-v822/'+name);
 if(!response.ok||sha(Buffer.from(await response.arrayBuffer()))!==files[name])throw Error('HTTP bytes differ: '+name);
}
const manifest=JSON.parse(await readFile(join(viewer,'manifest.json'),'utf8'));
if(manifest.files['fleet.json'].sha256!==files['fleet.json']||manifest.summary.ships!==23||manifest.summary.unique_asteroids!==199)throw Error('Viewer binding');
await writeFile(join(root,'viewer-import.json'),JSON.stringify({files,summary:manifest.summary,kepler_check:manifest.kepler_check,url:'http://127.0.0.1:4173/?dataset=gtoc12-catalogue-v822&epoch=69807&preset=oblique&z=1',visual_check:'Browser accessibility and screenshot confirm verified 23-ship/199-asteroid fleet, 13,023.705 weighted kg, 14,291.006 raw kg, physical Z scale 1 and active WebGL2. Served dataset bytes match local files.'},null,2));
for(const name of ['prepare_catalogue_viewer_v822.mjs','finalize_catalogue_v822.mjs'])await copyFile('build/performance/'+name,join(root,'reproduce',name));
const ownedSource=['src/spacepdhcg/gtoc12/data.py','src/spacepdhcg/gtoc12/gpu_completion_model.py','tests/test_gtoc12_gpu_completion_model.py','tests/test_gtoc12_catalogue_immutability.py'];
const frozen=JSON.parse(await readFile(join(root,'h100/runtime/source-manifest.json'),'utf8')).files,source={};
for(const path of ownedSource){source[path]=sha(await readFile(path));if(source[path]!==frozen[path])throw Error('Working source differs from final frozen tests: '+path);}
await writeFile(join(root,'working-source-binding.json'),JSON.stringify(source,null,2));
async function walk(dir){let out=[];for(const x of await readdir(dir,{withFileTypes:true})){const p=join(dir,x.name);if(x.isDirectory())out.push(...await walk(p));else out.push(p);}return out;}
const hashes={};for(const p of (await walk(root)).sort()){if(p===join(root,'sha256.json'))continue;hashes[relative(root,p).replaceAll('\\','/')]=sha(await readFile(p));}
await writeFile(join(root,'sha256.json'),JSON.stringify(hashes,null,2));
const owned=[...ownedSource,'docs/GPU_CATALOGUE_OWNERSHIP.md','results/lambda/2026-09-06/visualiser/app.js',viewer,root];
await writeFile('build/performance/owned_catalogue_v822.json',JSON.stringify(owned));
console.log(JSON.stringify({evidence_files:Object.keys(hashes).length,viewer:manifest.summary,source_files:ownedSource.length}));
