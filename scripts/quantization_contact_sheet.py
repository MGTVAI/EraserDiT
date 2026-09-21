"""Export sampled baseline/quantized video frames for visual verification."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sources=[cv2.VideoCapture(str(x)) for x in (args.reference,args.candidate)]
    try:
        counts=[int(s.get(cv2.CAP_PROP_FRAME_COUNT)) for s in sources]
        if not all(s.isOpened() for s in sources) or counts[0]!=counts[1] or counts[0]<1:
            raise ValueError('matching nonempty videos required')
        rows=[]
        for index in (0,counts[0]//3,2*counts[0]//3,counts[0]-1):
            tiles=[]
            for label,source in zip(('BF16','INT8'),sources):
                source.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,frame=source.read()
                if not ok:raise ValueError('decode failed')
                frame=cv2.resize(frame,(480,round(frame.shape[0]*480/frame.shape[1])),interpolation=cv2.INTER_AREA)
                tile=cv2.copyMakeBorder(frame,32,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                cv2.putText(tile,f'{label}, frame {index}',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
                tiles.append(tile)
            rows.append(np.concatenate(tiles,axis=1))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(args.output),np.concatenate(rows,axis=0)):
            raise RuntimeError('failed to write comparison')
    finally:
        for source in sources:source.release()


if __name__=='__main__':main()
