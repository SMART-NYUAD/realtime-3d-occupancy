"""Person detection + tracking wrapper around Ultralytics YOLO (COCO class 0)."""
from ultralytics import YOLO

PERSON_CLASS = 0


class PersonTracker:
    def __init__(self, cfg):
        d = cfg["detector"]
        self.model = YOLO(d["model"])
        self.conf = float(d["conf"])
        self.iou = float(d["iou"])
        self.imgsz = int(d["imgsz"])
        self.device = d["device"]
        self.tracker = d["tracker"]

    def track(self, frame):
        """Run detection + tracking on one BGR frame.

        Returns a list of dicts: {id, bbox:(x1,y1,x2,y2), conf}. With a pose model
        (yolo11*-pose.pt) each dict also carries {kxy:(17,2), kconf:(17,)} COCO
        keypoints, which the caller turns into an occlusion-robust foot point.
        """
        results = self.model.track(
            frame,
            persist=True,                 # keep IDs across calls
            classes=[PERSON_CLASS],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker,
            verbose=False,
        )
        out = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        # Pose models add keypoints, row-aligned with the boxes. Absent for a plain
        # detection model -> caller falls back to the box bottom.
        kobj = getattr(results[0], "keypoints", None)
        kxy_all = kobj.xy.cpu().numpy() if (kobj is not None and kobj.xy is not None) else None
        kcf_all = kobj.conf.cpu().numpy() if (kobj is not None and kobj.conf is not None) else None
        for i, ((x1, y1, x2, y2), tid, c) in enumerate(zip(xyxy, ids, confs)):
            d = {
                "id": int(tid),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(c),
            }
            if kxy_all is not None and i < len(kxy_all) and kcf_all is not None:
                d["kxy"] = kxy_all[i]
                d["kconf"] = kcf_all[i]
            out.append(d)
        return out
