#!/usr/bin/env python3
"""
export_hw4_sfm.py
Re-runs the HW4 custom SfM pipeline (SIFT → 8-pt → RANSAC → PnP → GTSAM BA)
on the Buddha dataset and saves results as COLMAP binary format at:
  data/buddha/sparse_hw4/0/{cameras.bin, images.bin, points3D.bin}

Run with system python: python3 export_hw4_sfm.py
"""
import os, struct, time
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

try:
    import gtsam
    from gtsam.symbol_shorthand import X, L
    HAS_GTSAM = True
    print("✓ GTSAM available")
except ImportError:
    HAS_GTSAM = False
    print("WARNING: GTSAM not found — skipping bundle adjustment")

SCRIPT_DIR       = Path(__file__).parent
IMAGE_DIR        = SCRIPT_DIR / "data/buddha/buddha_images"
CAMERAS_TXT      = SCRIPT_DIR / "data/buddha/cameras.txt"
OUTPUT_DIR       = SCRIPT_DIR / "data/buddha/sparse_hw4/0"
SEED_I, SEED_J   = 15, 16
WINDOW_SIZE      = 5
SIFT_NFEATURES   = 5000
RANSAC_ITERS     = 5000
RANSAC_THRESH    = 1.0
MIN_DEPTH        = 0.1
LOWE_RATIO       = 0.75

# ── 1. IMAGE LOADING ────────────────────────────────────────
def load_images(d):
    files = sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))
    imgs, names = [], []
    for f in files:
        img = cv2.imread(str(d / f))
        if img is not None:
            imgs.append(img); names.append(f)
    print(f"  {len(imgs)} images loaded")
    return imgs, names

# ── 2. INTRINSICS ────────────────────────────────────────────
def load_intrinsics(txt):
    K = {}
    with open(txt) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p = line.split()
            cid = int(p[0]); w,h = int(p[2]),int(p[3])
            fv,cx,cy,k1 = float(p[4]),float(p[5]),float(p[6]),float(p[7])
            K[cid] = {'K': np.array([[fv,0,cx],[0,fv,cy],[0,0,1]]),
                      'f':fv,'cx':cx,'cy':cy,'k1':k1,'w':w,'h':h}
    print(f"  {len(K)} cameras loaded")
    return K

# ── 3. FEATURE EXTRACTION ────────────────────────────────────
def extract_features(images):
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    bf   = cv2.BFMatcher()
    kps, descs = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, d = sift.detectAndCompute(gray, None)
        kps.append(kp); descs.append(d)
    matches = {}
    n = len(images)
    for i in range(n):
        for j in range(i+1, min(i+1+WINDOW_SIZE, n)):
            if descs[i] is None or descs[j] is None: continue
            raw = bf.knnMatch(descs[i], descs[j], k=2)
            good = [m for m,n2 in raw if m.distance < LOWE_RATIO * n2.distance]
            if len(good) >= 8:
                matches[(i,j)] = good
    print(f"  {len(matches)} matched pairs")
    return kps, matches

# ── 4. UNION-FIND TRACKS ─────────────────────────────────────
def build_tracks(n, kps, matches):
    nodes = {}; inv = []
    for i in range(n):
        for k in range(len(kps[i])):
            nodes[(i,k)] = len(inv); inv.append((i,k))
    par = np.arange(len(inv)); rnk = np.zeros(len(inv), dtype=int)
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb: return
        if rnk[ra]<rnk[rb]: ra,rb=rb,ra
        par[rb]=ra
        if rnk[ra]==rnk[rb]: rnk[ra]+=1
    for (i,j),ms in matches.items():
        for m in ms:
            a,b=nodes.get((i,m.queryIdx)),nodes.get((j,m.trainIdx))
            if a is not None and b is not None: union(a,b)
    r2t={}; trk=[dict() for _ in range(n)]
    for (i,k),nid in nodes.items():
        r=find(nid)
        if r not in r2t: r2t[r]=len(r2t)
        trk[i][k]=r2t[r]
    print(f"  {len(r2t)} tracks built")
    return trk

# ── 5. FUNDAMENTAL MATRIX ────────────────────────────────────
def norm_pts(pts):
    c=np.mean(pts,0); d=np.mean(np.linalg.norm(pts-c,axis=1))
    if d<1e-6: return None,None
    s=np.sqrt(2)/d; T=np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1]])
    ph=np.hstack((pts,np.ones((len(pts),1))))
    return (T@ph.T).T[:,:2],T

def est_F(p1,p2):
    n1,T1=norm_pts(p1); n2,T2=norm_pts(p2)
    if n1 is None: return None
    A=np.array([[x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1]
                for (x1,y1),(x2,y2) in zip(n1,n2)])
    _,_,Vt=np.linalg.svd(A); F=Vt[-1].reshape(3,3)
    U,S,Vt2=np.linalg.svd(F); S[2]=0; F=U@np.diag(S)@Vt2
    F=T2.T@F@T1; return F/F[2,2] if abs(F[2,2])>1e-10 else F

def sampson(F,p1,p2):
    h1=np.hstack((p1,np.ones((len(p1),1)))); h2=np.hstack((p2,np.ones((len(p2),1))))
    Fx1=(F@h1.T).T; Fx2=(F.T@h2.T).T
    return np.sum(h2*Fx1,1)**2/(Fx1[:,0]**2+Fx1[:,1]**2+Fx2[:,0]**2+Fx2[:,1]**2+1e-10)

def ransac_F(p1,p2):
    best_F,best_mask=None,np.zeros(len(p1),dtype=bool)
    for _ in range(RANSAC_ITERS):
        idx=np.random.choice(len(p1),8,replace=False)
        F=est_F(p1[idx],p2[idx])
        if F is None: continue
        mask=sampson(F,p1,p2)<RANSAC_THRESH
        if mask.sum()>best_mask.sum(): best_F,best_mask=F,mask
    return best_F,best_mask

# ── 6. ESSENTIAL MATRIX & POSE ────────────────────────────────
def compute_E(F,Ki,Kj):
    E=Kj.T@F@Ki; U,_,Vt=np.linalg.svd(E)
    return U@np.diag([1.,1.,0.])@Vt

def pose_candidates(E):
    U,_,Vt=np.linalg.svd(E)
    if np.linalg.det(U)<0: U[:,-1]*=-1
    if np.linalg.det(Vt)<0: Vt[-1,:]*=-1
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1,R2=U@W@Vt,U@W.T@Vt; t=U[:,2]
    return [(R1,t),(R1,-t),(R2,t),(R2,-t)]

def triangulate(P1,P2,p1,p2):
    p4=cv2.triangulatePoints(P1,P2,p1.T.astype(np.float32),p2.T.astype(np.float32))
    return (p4[:3]/(p4[3]+1e-10)).T

def cheirality(poses,Ki,Kj,p1,p2):
    P1=Ki@np.hstack((np.eye(3),np.zeros((3,1))))
    best,bn=None,-1
    for R,t in poses:
        P2=Kj@np.hstack((R,t.reshape(3,1)))
        pts=triangulate(P1,P2,p1,p2)
        d1=pts[:,2]; d2=(R@pts.T+t.reshape(3,1)).T[:,2]
        n=((d1>0)&(d2>0)).sum()
        if n>bn: best,(bn)=(R,t),n
    return best

# ── 7. PnP EXPANSION ─────────────────────────────────────────
def pnp_expand(indices, map3d, poses, trk, kps, Ki_all, matches):
    for i in indices:
        if i in poses: continue
        p3,p2=[],[]
        for ki,tid in trk[i].items():
            if tid in map3d:
                p3.append(map3d[tid]); p2.append(kps[i][ki].pt)
        if len(p3)<6: continue
        p3=np.array(p3,dtype=np.float32); p2=np.array(p2,dtype=np.float32)
        K=Ki_all[i+1]['K'].astype(np.float32)
        ok,rv,tv,inl=cv2.solvePnPRansac(p3,p2,K,None,iterationsCount=1000,reprojectionError=4.0)
        if not ok or inl is None or len(inl)<6: continue
        R,_=cv2.Rodrigues(rv); t=tv.flatten(); poses[i]=(R,t)
        for j in sorted(poses.keys()):
            if j==i: continue
            key=(min(i,j),max(i,j))
            if key not in matches: continue
            Rj,tj=poses[j]
            Ki=Ki_all[i+1]['K']; Kj=Ki_all[j+1]['K']
            Pi=Ki@np.hstack((R,t.reshape(3,1))); Pj=Kj@np.hstack((Rj,tj.reshape(3,1)))
            for m in matches[key]:
                qi=m.queryIdx if i<j else m.trainIdx
                qj=m.trainIdx if i<j else m.queryIdx
                if qi not in trk[i] or qj not in trk[j]: continue
                tid=trk[i][qi]
                if tid in map3d: continue
                pt=triangulate(Pi,Pj,np.array([kps[i][qi].pt]),np.array([kps[j][qj].pt]))
                if pt[0,2]>MIN_DEPTH and (Rj@pt[0]+tj)[2]>MIN_DEPTH:
                    map3d[tid]=pt[0]

# ── 8. GTSAM BA ───────────────────────────────────────────────
def gtsam_ba(map3d, poses, trk, kps, Ki_all):
    obs=defaultdict(list)
    for i,km in enumerate(trk):
        for ki,tid in km.items():
            if ki<len(kps[i]): obs[tid].append((i,kps[i][ki].pt))
    noise=gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2,1.0))
    graph=gtsam.NonlinearFactorGraph(); vals=gtsam.Values()
    anchor=False
    for i,(R,t) in poses.items():
        Rc2w=R.T; tc2w=-R.T@t
        Twc=gtsam.Pose3(gtsam.Rot3(Rc2w),gtsam.Point3(tc2w))
        vals.insert(X(i),Twc)
        if not anchor:
            graph.add(gtsam.PriorFactorPose3(X(i),Twc,
                gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3+[1e-6]*3))))
            anchor=True
    n_proj=0
    for tid,xyz in map3d.items():
        good_obs=[(ci,uv) for ci,uv in obs.get(tid,[])
                  if ci in poses and (poses[ci][0]@np.array(xyz)+poses[ci][1])[2]>MIN_DEPTH]
        if len(good_obs)<2: continue
        vals.insert(L(tid),gtsam.Point3(*xyz))
        for ci,(u,v) in good_obs:
            Km=Ki_all[ci+1]['K']
            gK=gtsam.Cal3_S2(Km[0,0],Km[1,1],Km[0,1],Km[0,2],Km[1,2])
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u,v),noise,X(ci),L(tid),gK,False,True))
            n_proj+=1
    p=gtsam.LevenbergMarquardtParams(); p.setMaxIterations(100)
    result=gtsam.LevenbergMarquardtOptimizer(graph,vals,p).optimize()
    ei=graph.error(vals); ef=graph.error(result)
    print(f"  BA: err {ei:.1f}→{ef:.1f} | RMSE {np.sqrt(ef/(2*max(n_proj,1))):.3f}px")
    opt_poses={i:(result.atPose3(X(i)).rotation().matrix().T,
                  -result.atPose3(X(i)).rotation().matrix().T@result.atPose3(X(i)).translation())
               for i in poses if result.exists(X(i))}
    opt_map={tid:np.array(result.atPoint3(L(tid))) for tid in map3d if result.exists(L(tid))}
    return opt_poses,opt_map

# ── 9. COLOR EXTRACTION ───────────────────────────────────────
def get_colors(map3d, poses, trk, kps, images):
    obs=defaultdict(list)
    for i,km in enumerate(trk):
        for ki,tid in km.items(): obs[tid].append((i,ki))
    colors={}
    for tid in map3d:
        for i,ki in obs[tid]:
            if i not in poses: continue
            u,v=int(round(kps[i][ki].pt[0])),int(round(kps[i][ki].pt[1]))
            h,w=images[i].shape[:2]
            if 0<=u<w and 0<=v<h:
                b,g,r=images[i][v,u]; colors[tid]=(int(r),int(g),int(b)); break
        if tid not in colors: colors[tid]=(128,128,128)
    return colors

# ── 10. COLMAP BINARY WRITERS ─────────────────────────────────
def quat(R):
    tr=R[0,0]+R[1,1]+R[2,2]
    if tr>0:
        s=0.5/np.sqrt(tr+1); return np.array([0.25/s,(R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s])
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); return np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); return np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); return np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])

def write_cameras(out, Ki_all):
    with open(out/'cameras.bin','wb') as f:
        f.write(struct.pack('<Q',len(Ki_all)))
        for cid,info in sorted(Ki_all.items()):
            f.write(struct.pack('<iiqq',cid,2,info['w'],info['h']))
            f.write(struct.pack('<4d',info['f'],info['cx'],info['cy'],info['k1']))
    print(f"  ✓ cameras.bin  ({len(Ki_all)} cameras)")

def write_images(out, poses, names):
    with open(out/'images.bin','wb') as f:
        f.write(struct.pack('<Q',len(poses)))
        for ci in sorted(poses):
            R,t=poses[ci]; q=quat(R)
            f.write(struct.pack('<i',ci+1))
            f.write(struct.pack('<4d',*q.tolist()))
            f.write(struct.pack('<3d',*t.tolist()))
            f.write(struct.pack('<i',ci+1))
            f.write(names[ci].encode()+b'\x00')
            f.write(struct.pack('<Q',0))
    print(f"  ✓ images.bin   ({len(poses)} images)")

def write_points3d(out, map3d, colors):
    with open(out/'points3D.bin','wb') as f:
        f.write(struct.pack('<Q',len(map3d)))
        for tid,xyz in map3d.items():
            r,g,b=colors.get(tid,(128,128,128))
            f.write(struct.pack('<Q',tid+1))
            f.write(struct.pack('<3d',*xyz.tolist()))
            f.write(struct.pack('<3B',r,g,b))
            f.write(struct.pack('<d',1.0))
            f.write(struct.pack('<Q',0))
    print(f"  ✓ points3D.bin ({len(map3d)} points)")

# ── MAIN ──────────────────────────────────────────────────────
def main():
    t0=time.time()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    print("\n[1/7] Loading images...")
    images,names=load_images(IMAGE_DIR); n=len(images)

    print("\n[2/7] Loading intrinsics...")
    Ki_all=load_intrinsics(CAMERAS_TXT)

    print("\n[3/7] Extracting features & matching...")
    kps,matches=extract_features(images)

    print("\n[4/7] Building feature tracks...")
    trk=build_tracks(n,kps,matches)

    print(f"\n[5/7] Seed pair ({SEED_I},{SEED_J}) → initial triangulation...")
    ms=matches.get((SEED_I,SEED_J),[])
    if len(ms)<8:
        (si,sj)=max(matches.keys(),key=lambda k:len(matches[k])); ms=matches[(si,sj)]
        print(f"  Fallback seed: ({si},{sj}) with {len(ms)} matches")
    else:
        si,sj=SEED_I,SEED_J
    p1=np.array([kps[si][m.queryIdx].pt for m in ms])
    p2=np.array([kps[sj][m.trainIdx].pt for m in ms])
    F,mask=ransac_F(p1,p2)
    p1i,p2i=p1[mask],p2[mask]
    inlier_ms=[ms[i] for i,k in enumerate(mask) if k]
    print(f"  RANSAC: {mask.sum()}/{len(mask)} inliers")
    Ki,Kj=Ki_all[si+1]['K'],Ki_all[sj+1]['K']
    R_s,t_s=cheirality(pose_candidates(compute_E(F,Ki,Kj)),Ki,Kj,p1i,p2i)
    P1=Ki@np.hstack((np.eye(3),np.zeros((3,1))))
    P2=Kj@np.hstack((R_s,t_s.reshape(3,1)))
    pts3d=triangulate(P1,P2,p1i,p2i)
    map3d={}; poses={si:(np.eye(3),np.zeros(3)),sj:(R_s,t_s)}
    for m,pt in zip(inlier_ms,pts3d):
        if pt[2]>MIN_DEPTH and (R_s@pt+t_s)[2]>MIN_DEPTH:
            if m.queryIdx in trk[si]:
                map3d[trk[si][m.queryIdx]]=pt
    print(f"  Seeded: {len(map3d)} points")

    print("\n[6/7] PnP expansion...")
    pnp_expand(range(sj+1,n),map3d,poses,trk,kps,Ki_all,matches)
    pnp_expand(range(si-1,-1,-1),map3d,poses,trk,kps,Ki_all,matches)
    print(f"  {len(poses)}/{n} cameras | {len(map3d)} points")

    if HAS_GTSAM:
        print("\n  Running GTSAM BA...")
        poses,map3d=gtsam_ba(map3d,poses,trk,kps,Ki_all)
        print(f"  After BA: {len(poses)} cameras | {len(map3d)} points")

    print("\n[7/7] Extracting colors & writing COLMAP binary...")
    colors=get_colors(map3d,poses,trk,kps,images)
    write_cameras(OUTPUT_DIR,Ki_all)
    write_images(OUTPUT_DIR,poses,names)
    write_points3d(OUTPUT_DIR,map3d,colors)

    print(f"\n✓ Done in {(time.time()-t0)/60:.1f} min → {OUTPUT_DIR}")

if __name__=="__main__":
    main()
