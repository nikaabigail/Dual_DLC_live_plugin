"""Offline video -> native ROI/DLC-Live -> DDLP packets -> native C++ replay.

The infer command uses the production single-camera runtime helpers. The render
command consumes JSONL exported by DualDLCLiveOfflineReplay.ReplayFile; it does
not implement point filtering or side selection in Python. No live UDP is sent.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

os.environ.setdefault('DLClight', 'True')
os.environ.setdefault('MPLBACKEND', 'Agg')
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def json_write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git_output(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


def infer(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'raw_roi_poses.npz').exists():
        raise RuntimeError('Completed inference exists; use another output directory.')
    os.environ['DLC_LIVE_KNEE_MODEL_PATH'] = str(args.model.resolve())
    sys.path.insert(0,str(REPO/'python'))
    import single_rt_dlc_live_bridge as single
    import dlclive
    import torch
    config, dual, live = single.config, single.dual, single.live
    single.live_profiles.apply_profile(config, args.profile)
    logger = logging.getLogger('knees_roi_preview')
    logging.basicConfig(level=logging.INFO)
    dual.configure_runtime_backends(logger)
    # Only serialize the native packet. No bridge.open(), camera SDK or TTL.
    bridge = dual.OpenEphysBridge(logger)
    cap = cv2.VideoCapture(str(args.video.resolve()))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or n <= 0:
        raise RuntimeError('Source must have a known frame count and positive frame rate.')
    if args.limit:
        n = min(n,args.limit)
    dlc = live.build_dlc_live(None)
    names = live.extract_bodyparts(dlc.read_config())
    if not {'hl_knee_l','hl_knee_r'}.issubset(names):
        raise RuntimeError('The selected model does not contain both knees.')
    indices = {name:i for i,name in enumerate(names)}
    roi = single.build_leg_roi_tracker(names,(height,width,3))
    if roi is None:
        raise RuntimeError('This preview requires the native leg ROI tracker.')
    poses = np.lib.format.open_memmap(out/'raw_roi_poses.partial.npy',mode='w+',dtype=np.float32,shape=(n,len(names),3))
    poses[:] = np.nan
    windows = np.empty((n,4),dtype=np.int32)
    routes = np.empty(n,dtype='U5')
    timings = np.empty((n,3),dtype=np.float64)
    runtime = single.make_dummy_runtime('left')
    other_runtime = single.make_dummy_runtime('right')
    source_files = ['python/single_rt_dlc_live_bridge.py','python/dual_rt_dlc_live.py',
                    'python/rt_dlc_live.py','python/live_profiles.py','python/config_dual_rt_dlc_live.py',
                    'python/config_rt_dlc_live.py','python/pose_layout.py','scripts/preview_knees_roi.py']
    manifest = {
        'created_local':datetime.now().astimezone().isoformat(),
        'source_video':str(args.video.resolve()),'source_sha256':file_hash(args.video),
        'model':str(args.model.resolve()),'model_sha256':file_hash(args.model),
        'repository':str(REPO),'git_head':git_output('rev-parse','HEAD').decode().strip(),
        'plugin_version':(REPO/'VERSION').read_text(encoding='utf-8-sig').strip(),
        'source_hashes':{name:file_hash(REPO/name) for name in source_files},
        'profile':args.profile,'fps':fps,'frames':n,'width':width,'height':height,
        'bodyparts':names,'wire_points':list(config.DUAL_USE_POINTS),
        'roi_width':roi.width,'roi_anchor_names':[names[i] for i in roi.leg_idx],
        'runtime':{'dlclive':dlclive.__version__,'torch':torch.__version__,
                   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'precision':config.PRECISION,'convert_to_rgb':config.CONVERT_TO_RGB,
                   'tf32':getattr(config,'DUAL_TORCH_ALLOW_TF32',False),
                   'compile_backend':getattr(config,'DUAL_TORCH_COMPILE_BACKEND',''),
                   'constant_zero_padding':getattr(config,'MODEL_CONSTANT_ZERO_PADDING',False)},
        'inference_path':'native build_dlc_live + LegRoiTracker + run_raw_inference + raw_pose_result + make_pair_result',
        'postprocessing_path':'native C++ DDLP parser/filterPoint/final side selection via offline replay',
        'live_output_enabled':False,'native_postprocessing_applied':False,
    }
    json_write(out/'manifest.json',manifest)
    (out/'inference_source.patch').write_bytes(git_output('diff','--','python'))
    initialized = False
    start = time.perf_counter()
    processed = 0
    try:
        with (out/'native_packets.ddlp.partial').open('wb') as stream:
            for i in range(n):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f'Source ended at frame {i}/{n}')
                rgb_input = str(getattr(config,'GALAXY_OUTPUT_COLOR','bgr')).lower() == 'rgb'
                frame = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB) if rgb_input else bgr
                packet = live.FramePacket(frame_id=i+1,frame=frame,capture_ts=i/fps,source_frame_id=i+1)
                window = roi.window()
                if window is None:
                    raise RuntimeError('Native ROI unexpectedly disabled')
                windows[i] = window
                dlc.cropping = window
                before = time.perf_counter()
                initialized, pose, pre_ms, model_ms = dual.run_raw_inference(dlc,initialized,packet)
                elapsed_ms = (time.perf_counter()-before)*1000
                pose = np.asarray(pose,dtype=np.float32)
                if pose.shape != (len(names),3) or not np.isfinite(pose).all():
                    raise RuntimeError(f'Invalid network output at frame {i}: {pose.shape}')
                roi.update(pose)
                result = dual.raw_pose_result(runtime,pose,indices,elapsed_ms,pre_ms,model_ms)
                pair = single.make_pair_result(i+1,packet,result,'left')
                route = 'left' if pair.left_result is result else 'right'
                encoded = bridge._build_binary_pose_payload(pair,runtime,other_runtime)
                if len(encoded) != 300:
                    raise RuntimeError(f'Expected an 8-point packet, got {len(encoded)} bytes')
                stream.write(struct.pack('<I',len(encoded)))
                stream.write(encoded)
                poses[i] = pose
                routes[i] = route
                timings[i] = (elapsed_ms,pre_ms,model_ms)
                processed = i+1
                if processed%250 == 0 or processed == n:
                    poses.flush()
                    elapsed = time.perf_counter()-start
                    json_write(out/'status.json',{'stage':'inference','completed':processed,'total':n,
                               'fps':round(processed/elapsed,2),'eta_seconds':round((n-processed)*elapsed/processed,1)})
    finally:
        cap.release()
        dlc.close()
        bridge.close()
    if processed != n:
        raise RuntimeError('Incomplete inference')
    with (out/'raw_roi_poses.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream,poses=poses,bodyparts=np.asarray(names),fps=fps,
                            roi=windows,python_route=routes,timings_ms=timings,frame_index=np.arange(n))
    os.replace(out/'raw_roi_poses.npz.tmp',out/'raw_roi_poses.npz')
    os.replace(out/'native_packets.ddlp.partial',out/'native_packets.ddlp')
    manifest['inference_seconds'] = round(time.perf_counter()-start,3)
    manifest['steady_model_ms_median'] = float(np.median(timings[1:,2])) if n>1 else float(timings[0,2])
    json_write(out/'manifest.json',manifest)
    json_write(out/'status.json',{'stage':'ready_for_native_cpp_replay','completed':n,'total':n})


class Encoder:
    def __init__(self,path,width,height,fps):
        exe = shutil.which('ffmpeg')
        if not exe:
            raise RuntimeError('ffmpeg is required to render browser-compatible H264')
        self.path = path
        self.temporary = path.with_name(path.stem+'.rendering.mp4')
        self.log = path.with_suffix('.ffmpeg.log').open('wb')
        command = [exe,'-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','bgr24',
                   '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
                   '-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(self.temporary)]
        self.proc = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def write(self,frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    def finish(self):
        self.proc.stdin.close()
        result = self.proc.wait()
        self.log.close()
        if result:
            raise RuntimeError(f'ffmpeg exited {result}')
        os.replace(self.temporary,self.path)
    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def text(frame,value,position,scale=.6,color=(235,235,235)):
    cv2.putText(frame,value,position,cv2.FONT_HERSHEY_SIMPLEX,scale,(12,12,12),3,cv2.LINE_AA)
    cv2.putText(frame,value,position,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def draw_native(frame,side_record):
    result = frame.copy()
    side = str(side_record.get('picked_side',''))
    suffix = 'l' if side=='left' else 'r' if side=='right' else ''
    color = (205,220,35) if side=='left' else (30,150,255)
    points = {}
    h,w = frame.shape[:2]
    for name,p in side_record.get('points',{}).items():
        if not suffix or not name.endswith('_'+suffix) or not p.get('valid',False):
            continue
        x,y = p.get('x'),p.get('y')
        if x is None or y is None or not np.isfinite([x,y]).all() or not (0<=x<w and 0<=y<h):
            continue
        points[name] = (int(round(x)),int(round(y)))
    chain = [f'hl_{joint}_{suffix}' for joint in ('hip','knee','ankle','toes')]
    for a,b in zip(chain,chain[1:]):
        if a in points and b in points:
            cv2.line(result,points[a],points[b],(10,10,10),4,cv2.LINE_AA)
            cv2.line(result,points[a],points[b],color,2,cv2.LINE_AA)
    for name,xy in points.items():
        cv2.circle(result,xy,4,(10,10,10),-1,cv2.LINE_AA)
        cv2.circle(result,xy,3,color,-1,cv2.LINE_AA)
        if 'knee' in name:
            cv2.circle(result,xy,7,color,2,cv2.LINE_AA)
    return result,side,color


def render(args):
    cv2.setNumThreads(2)
    out = args.output.resolve()
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8-sig'))
    data = np.load(out/'raw_roi_poses.npz',allow_pickle=False)
    rois,routes = data['roi'],data['python_route']
    n,fps = len(rois),float(data['fps'])
    with args.native_jsonl.open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != n or any(int(row['pair_index']) != i+1 for i,row in enumerate(rows)):
        raise RuntimeError('Native replay output does not match all inference frames')
    cap = cv2.VideoCapture(manifest['source_video'])
    if not cap.isOpened():
        raise RuntimeError('Cannot open source video')
    full = Encoder(out/'full_roi_native.mp4',manifest['width'],manifest['height'],fps)
    review = Encoder(out/'review_roi_native.mp4',1280,800,fps)
    start = time.perf_counter()
    preview_indices = {0,500,n//2,n-1}
    stats = {'frames':n,'left':0,'right':0,'knee_visible':0,'native_replay_file':str(args.native_jsonl.resolve()),
             'native_replay_sha256':file_hash(args.native_jsonl), 'point_filter':'native C++ filterPoint',
             'side_selection':'native C++ picked_side on Python-routed active camera',
             'smoothing_in_renderer':False, 'roi_shown':'exact inference crop, no additional viewport tracker'}
    try:
        for i,row in enumerate(rows):
            ok,frame = cap.read()
            if not ok:
                raise RuntimeError(f'Missing source frame {i}')
            routed = row[str(routes[i])]
            annotated,side,color = draw_native(frame,routed)
            if side in ('left','right'):
                stats[side] += 1
                kp = routed.get('points',{}).get('hl_knee_'+side[0],{})
                stats['knee_visible'] += int(bool(kp.get('valid',False)))
            x1,x2,y1,y2 = map(int,rois[i])
            if not (0<=x1<x2<=frame.shape[1] and 0<=y1<y2<=frame.shape[0]):
                raise RuntimeError('Invalid ROI bounds')
            cropped = annotated[y1:y2,x1:x2].copy()
            cv2.rectangle(annotated,(x1,y1),(x2-1,y2-1),(235,235,235),1,cv2.LINE_AA)
            text(annotated,'ROI',(x1+5,min(y1+18,y2-1)),.4)
            canvas = np.full((800,1280,3),(25,24,22),dtype=np.uint8)
            version = manifest.get('plugin_version','')
            text(canvas,f'Dual DLC Live | {version} | native filters',(18,28),.62)
            text(canvas,f'{side.upper() or "NO SIDE"}  {i/fps:.2f} s',(1000,28),.55,color)
            canvas[44:191] = cv2.resize(annotated,(1280,147),interpolation=cv2.INTER_AREA)
            text(canvas,f'Actual ROI: {x2-x1} x {y2-y1}',(82,213),.48)
            canvas[224:774,80:1200] = cv2.resize(cropped,(1120,550),interpolation=cv2.INTER_LINEAR)
            full.write(annotated)
            review.write(canvas)
            if i in preview_indices:
                ok,png = cv2.imencode('.png',canvas)
                if not ok:
                    raise RuntimeError('Preview encoding failed')
                png.tofile(str(out/f'preview_{i:06d}.png'))
            if (i+1)%500 == 0:
                elapsed = time.perf_counter()-start
                json_write(out/'status.json',{'stage':'rendering','completed':i+1,'total':n,
                           'fps':round((i+1)/elapsed,2),'eta_seconds':round((n-i-1)*elapsed/(i+1),1)})
        full.finish()
        review.finish()
    except BaseException:
        full.abort()
        review.abort()
        raise
    finally:
        cap.release()
    stats['render_seconds'] = round(time.perf_counter()-start,3)
    json_write(out/'render_summary.json',stats)
    manifest['native_postprocessing_applied'] = True
    manifest['native_replay_file'] = str(args.native_jsonl.resolve())
    manifest['native_replay_sha256'] = stats['native_replay_sha256']
    json_write(out/'manifest.json',manifest)
    json_write(out/'status.json',{'stage':'complete','completed':n,'total':n})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('infer')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',default='single-knees-strict')
    p.add_argument('--limit',type=int,default=0)
    p = sub.add_parser('render')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-jsonl',type=Path,required=True)
    args = parser.parse_args()
    try:
        infer(args) if args.command=='infer' else render(args)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        json_write(args.output/'status.json',{'stage':'failed','error':str(exc)})
        raise


if __name__ == '__main__':
    main()
