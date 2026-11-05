"""Collect actual evidence, build an offline illustrated HTML tutorial and source ZIP."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

import markdown

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tutorial'
EVIDENCE=DEST/'evidence'


def collect():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    final=ROOT/'outputs/runtime_stand_v1'
    files={name:final/name for name in ['config.json','training_status.json','results.json',
        'progress.csv','eval_nominal.json','eval_sensors.json','eval_dynamics.json','eval_onnx_dynamics.json']}
    files.update({name:ROOT/'outputs'/name for name in ['verification.json','runtime_contract_tests.json']})
    files.update({name:ROOT/'.tools'/name for name in ['setup_runtime_final.log','runtime_contract_tests.log',
        'runtime_stand_v1.log','runtime_smoke.log','runtime_resume_smoke.log',
        'runtime_walk_smoke.log','runtime_export.log']})
    files['onnx_validation.json']=final/'export/validation.json'
    for name,path in files.items():
        shutil.copy2(path,EVIDENCE/name)
    manifest={name:{'source':str(path.relative_to(ROOT)),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bytes':path.stat().st_size} for name,path in files.items()}
    (EVIDENCE/'source_manifest.json').write_text(json.dumps(manifest,indent=2))


def build_html():
    source=(ROOT/'TUTORIAL.md').read_text()
    md=markdown.Markdown(extensions=['fenced_code','tables','toc'],
                         extension_configs={'toc':{'toc_depth':'1'}})
    body=md.convert(source)
    # Rewrite URL attributes only; never change executable command text.
    body=body.replace('src="tutorial/assets/','src="assets/').replace('href="tutorial/assets/','href="assets/')
    body=body.replace('href="tutorial/evidence/','href="evidence/')
    flow=('<div class="flow" role="img" aria-label="观测到物理执行与 PPO 更新流程">'
          '<span>01 · 原始 61 维观测</span><span>02 · 观测归一化</span><span>03 · Actor 推理</span>'
          '<span>04 · 原始 14 维动作</span><span>05 · 缩放与低通</span><span>06 · BAM 与 MuJoCo</span>'
          '<p>物理状态形成下一帧观测；奖励经 GAE 和 PPO 更新 Actor。</p></div>')
    body=re.sub(r'<pre><code class="language-mermaid">.*?</code></pre>',lambda _:flow,body,flags=re.S)
    def gif(match):
        alt,path=match.group(1),match.group(2)
        poster=path.replace('.gif','_poster.png')
        return (f'<figure class="gif"><img src="{path}" alt="{alt}" data-gif="{path}" '
                f'data-poster="{poster}"><button type="button" class="pause" '
                f'aria-label="暂停或播放这段动画">暂停动画</button></figure>')
    body=re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.gif)"\s*/></p>',gif,body)
    def video(match):
        path,label=match.group(1),match.group(2)
        return (f'<li class="video-item"><p>{label}</p>'
                f'<video controls playsinline preload="none" src="{path}"></video>'
                f'<a href="{path}">打开原始 MP4</a></li>')
    body=re.sub(r'<li><a href="([^"]+\.mp4)">([^<]+)</a></li>',video,body)
    css='''
    :root{--ink:#20314c;--muted:#5c6e86;--blue:#265e9e;--line:#dce4ef;--bg:#f3f6fa}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC",sans-serif}
    header{background:#17314e;color:#fff;padding:18px 30px;display:flex;align-items:center;justify-content:space-between;gap:16px}
    header b{font-size:19px}header small{display:block;color:#c8d7e8}button{font:inherit;cursor:pointer}header button{border:1px solid #b5c9de;border-radius:6px;background:transparent;color:#fff;padding:5px 12px;white-space:nowrap}
    .layout{max-width:1420px;margin:auto;display:grid;grid-template-columns:260px minmax(0,1fr);gap:28px;padding:28px}
    aside{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;font-size:13px;line-height:1.7;padding:18px;background:white;border-radius:10px;border:1px solid var(--line)}
    aside ul{list-style:none;padding:0;margin:0}aside li{margin:8px 0}aside a{color:var(--muted);text-decoration:none}aside a:hover{color:var(--blue)}
    article{min-width:0;background:white;padding:30px 42px;border:1px solid var(--line);border-radius:10px}article>h1:first-child{font-size:32px;line-height:1.4;margin:0 0 18px;color:#102b49}
    h1{font-size:25px;line-height:1.5;margin:52px 0 18px;padding-top:10px;border-top:2px solid #e6edf5;scroll-margin-top:20px}h2{font-size:19px;margin:30px 0 12px}p{margin:14px 0}a{color:var(--blue);overflow-wrap:anywhere}
    pre{overflow:auto;background:#12263e;color:#e5edf8;padding:18px 20px;border-radius:8px;font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}pre code{padding:0;background:none;color:inherit;overflow-wrap:normal}
    code{background:#eef3f9;color:#234c77;border-radius:4px;padding:2px 5px;font:13px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}
    table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0;table-layout:auto}th,td{text-align:left;vertical-align:top;border:1px solid var(--line);padding:10px 12px;overflow-wrap:anywhere}th{background:#edf3fa}tr:nth-child(even){background:#fafcfe}
    img{max-width:100%;height:auto;border-radius:7px}figure.gif{position:relative;margin:25px 0}.pause{display:block;border:1px solid var(--line);color:var(--muted);background:#fff;border-radius:5px;padding:3px 10px;font-size:12px;margin-top:4px}
    .flow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:25px 0}.flow span{background:#eaf2fb;border:1px solid #c9dbef;border-radius:7px;padding:14px;font-size:14px}.flow p{grid-column:1/-1;color:var(--muted);font-size:13px;margin:0}
    .video-item{list-style:none;margin:22px 0;border:1px solid var(--line);padding:14px;border-radius:8px}.video-item p{margin:0 0 10px;font-weight:600}.video-item video{width:100%;aspect-ratio:4/3;background:#152b43;border-radius:6px;display:block}.video-item a{font-size:13px}
    footer{text-align:center;padding:25px;color:var(--muted);font-size:13px}.mobile-toc{display:none}
    @media(max-width:900px){.layout{display:block;padding:12px}aside{display:none}.mobile-toc{display:block;background:white;border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}.mobile-toc ul{padding-left:20px}article{padding:22px 18px}article>h1:first-child{font-size:25px}h1{font-size:22px}header{padding:14px 18px}table{font-size:12px}td,th{padding:7px}pre{font-size:12px;padding:14px}}
    @media print{header,aside,.mobile-toc,.pause,video{display:none!important}.layout{display:block;padding:0}article{border:0;padding:0}body{font-size:11pt;background:#fff}pre{white-space:pre-wrap;color:#111;background:#f1f3f6}h1{break-before:auto}img,table{break-inside:avoid}a{color:#111;text-decoration:none}}
    '''
    script='''
    document.querySelectorAll('.gif').forEach(f=>{const i=f.querySelector('img'),b=f.querySelector('button');let playing=!matchMedia('(prefers-reduced-motion: reduce)').matches;function apply(){i.src=playing?i.dataset.gif:i.dataset.poster;b.textContent=playing?'暂停动画':'播放动画'}b.onclick=()=>{playing=!playing;apply()};apply()});
    '''
    html_text=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MicroDuck 从零训练与验证教程</title><style>{css}</style></head><body>
    <header><div><b>MicroDuck · 从零训练与验证</b><small>真实运行记录 · Apple Silicon CPU · 2026 年 11 月 4 日</small></div><button onclick="window.print()">打印或保存 PDF</button></header>
    <div class="layout"><aside><b>教程目录</b>{md.toc}</aside><main><details class="mobile-toc"><summary>展开教程目录</summary>{md.toc}</details><article>{body}</article></main></div>
    <footer>图表来自实际 CSV / JSON；GIF 来自冻结模型的 MuJoCo 回放；实机测试尚未进行。</footer><script>{script}</script></body></html>'''
    (DEST/'index.html').write_text(html_text)


def small_archives():
    sources=[p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in ('.py','.md','.toml','.lock') or p.name=='.gitignore')]
    for folder in ['scripts','contracts']:
        sources.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    evidence=[p for p in EVIDENCE.rglob('*') if p.is_file()]
    for name,files in [('MicroDuck_训练源码.zip',sources),('MicroDuck_训练验证原始结果.zip',evidence)]:
        with zipfile.ZipFile(ROOT/name,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(files)):
                z.write(p,Path('microduck_sim')/p.relative_to(ROOT))


def package():
    archive=ROOT/'MicroDuck_从零训练图文教程.zip'
    selected=[]
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in ('.py','.md','.toml','.lock') or p.name=='.gitignore'):
            selected.append(p)
    for folder in ['scripts','contracts','tutorial']:
        selected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package_info.json')
    for run in ['runtime_smoke','runtime_resume_smoke','runtime_walk_smoke','runtime_stand_v1']:
        for p in (ROOT/'outputs'/run).rglob('*'):
            if p.is_file() and (p.suffix in ('.json','.csv','.zip','.pkl','.onnx','.npz','.md')):
                selected.append(p)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(selected)):
            z.write(p,Path('microduck_sim')/p.relative_to(ROOT))
    result={'archive':archive.name,'bytes':archive.stat().st_size,'files':len(set(selected)),
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (DEST/'package_info.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    collect()
    build_html()
    small_archives()
    package()
