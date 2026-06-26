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

        Returns a list of dicts: {id, bbox:(x1,y1,x2,y2), conf}.
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
        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
            out.append({
                "id": int(tid),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(c),
            })
        return out
