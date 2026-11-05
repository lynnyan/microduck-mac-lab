"""Build evidence-based training figures and combined GIFs from saved rollouts.

Run after recording tutorial/assets/{early_2048,nominal,sensors,dynamics,checkpoint}.mp4.
No model training and no synthetic robot motion is performed here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tutorial/assets"
FONT = next((p for p in [Path('/System/Library/Fonts/STHeiti Medium.ttc'),
                        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
             if p.exists()), None)


def font(size):
    return ImageFont.truetype(str(FONT),size) if FONT else ImageFont.load_default(size=size)


def load_video(name):
    reader=imageio.get_reader(str(ASSETS/f'{name}.mp4'))
    meta=reader.get_meta_data()
    frames=[Image.fromarray(frame) for frame in reader]
    reader.close()
    return frames,float(meta['fps'])


def combine(name, panels, seconds=5, width=384):
    videos=[load_video(p[0]) for p in panels]
    height=width*3//4
    gutter,margin,header,footer=12,16,62,40
    total_width=2*margin+len(panels)*width+(len(panels)-1)*gutter
    images=[]
    for frame_index in range(int(seconds*10)):
        now=frame_index/10
        canvas=Image.new('RGB',(total_width,header+height+footer),(245,247,251))
        draw=ImageDraw.Draw(canvas)
        for panel_index,((source,title,detail),(frames,fps)) in enumerate(zip(panels,videos)):
            x=margin+panel_index*(width+gutter)
            raw_index=int(now*fps)
            frozen=raw_index>=len(frames)
            image=frames[min(raw_index,len(frames)-1)].resize((width,height),Image.Resampling.LANCZOS)
            canvas.paste(image,(x,header))
            draw.text((x,10),title,font=font(20),fill=(18,37,63))
            draw.text((x,36),detail,font=font(13),fill=(86,100,120))
            stamp=f't = {(min(raw_index,len(frames)-1)+1)/fps:.2f} s'
            if frozen:
                stamp+='  失败后定格'
            draw.text((x,header+height+10),stamp,font=font(16),
                      fill=(181,49,45) if frozen else (33,110,87))
        images.append(canvas)
    # One shared palette avoids frame-to-frame color flicker.
    palette_sheet=Image.new('RGB',(total_width,images[0].height*3))
    for i,im in enumerate([images[0],images[len(images)//2],images[-1]]):
        palette_sheet.paste(im,(0,i*im.height))
    palette=palette_sheet.quantize(colors=128)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in images]
    indexed[0].save(ASSETS/f'{name}.gif',save_all=True,append_images=indexed[1:],
                    duration=100,loop=0,optimize=True,disposal=2)
    images[len(images)//2].save(ASSETS/f'{name}_poster.png')


def plot_training():
    with (ROOT/'outputs/runtime_stand_v1/progress.csv').open() as f:
        rows=list(csv.DictReader(f))
    def column(key):
        return np.array([float(r[key]) if r[key] else np.nan for r in rows])
    x=column('time/total_timesteps')/1000
    fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
    fig.patch.set_facecolor('#f5f7fb')
    specs=[('rollout/ep_rew_mean','Mean episode return','Raw return'),
           ('rollout/ep_len_mean','Mean training episode duration','Seconds'),
           ('train/approx_kl','Approximate policy KL','KL'),
           ('train/std','Policy exploration standard deviation','Std (rad)')]
    stages=[(0,200,'Nominal','#dbeafe'),(200,600,'Sensors','#dcfce7'),(600,801,'Dynamics','#fef3c7')]
    for ax,(key,title,ylabel) in zip(axes.flat,specs):
        for a,b,label,color in stages:
            ax.axvspan(a,b,color=color,alpha=.55,zorder=0)
        y=column(key)
        if key=='rollout/ep_len_mean': y=y*.02
        ax.plot(x,y,color='#255a9b',linewidth=1.1)
        ax.set(title=title,xlabel='Collected transitions (thousands)',ylabel=ylabel,xlim=(0,801))
        ax.grid(alpha=.17)
        ax.spines[['top','right']].set_visible(False)
        if key=='rollout/ep_len_mean':
            ax.axhline(10,color='#6b7280',ls='--',lw=1)
            ax.set_ylim(0,10.5)
        if key=='train/approx_kl':
            ax.axhline(.03,color='#a34e35',ls='--',lw=1)
    fig.suptitle('MicroDuck · actual PPO training log · 800,768 transitions',fontsize=16,color='#132a46')
    fig.legend([Patch(facecolor=c,alpha=.55) for _,_,_,c in stages],
               [label for _,_,label,_ in stages],loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(ASSETS/'training_curves.png',dpi=160)
    plt.close(fig)


def plot_evaluation():
    data=json.loads((ROOT/'outputs/runtime_stand_v1/results.json').read_text())
    labels=['Nominal','Sensors','Dynamics','ONNX\nDynamics']
    counts=[data['profiles'][p]['survived'] for p in ('nominal','sensors','dynamics')]+[10*data['onnx_dynamics']['survival_rate']]
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    ax.bar(labels,counts,color=['#4877ae','#388c83','#c6a24b','#7765ad'],width=.55)
    for i,n in enumerate(counts): ax.text(i,n+.15,f'{int(n)}/10',ha='center',fontsize=14)
    ax.set(ylim=(0,12),yticks=[0,2,4,6,8,10],ylabel='Episodes reaching 20 s without falling',
           title='Held-out seeds 2027–2036 · zero velocity command')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    fig.savefig(ASSETS/'evaluation_results.png',dpi=160)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    combine('before_after',[
        ('early_2048','短训模型 · 2,048 步','同一契约 / seed 2027 / 约 0.94 秒摔倒'),
        ('nominal','正式模型 · 800,768 步','同一契约 / seed 2027 / 这里回放前 5 秒')],seconds=5)
    combine('robustness',[
        ('nominal','标称物理','固定标称参数'),
        ('sensors','传感器误差','噪声、偏置与测量延迟'),
        ('dynamics','动力学误差','再加入电压、摩擦、质量、执行延迟')],seconds=6,width=300)
    combine('checkpoint_onnx',[
        ('checkpoint','训练端策略','SB3 + 配对 VecNormalize'),
        ('nominal','导出 ONNX','归一化已在图内 / 原始 61 维输入')],seconds=6)
    plot_training()
    plot_evaluation()
    manifest={}
    for p in sorted(ASSETS.iterdir()):
        if p.suffix in ('.gif','.png','.mp4'):
            entry={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            if p.suffix=='.gif':
                with Image.open(p) as im:
                    entry.update(frames=im.n_frames,size=list(im.size),loop=im.info.get('loop'))
                if p.stat().st_size>10*1024*1024: raise ValueError(f'{p} exceeds Page image limit')
            manifest[p.name]=entry
    (ROOT/'tutorial/evidence/media_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k.endswith('.gif')},indent=2))


if __name__=='__main__':
    main()
