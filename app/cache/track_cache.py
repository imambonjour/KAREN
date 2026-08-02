import time


class TrackCache:
    def __init__(self, ghost_ttl: float = 5.0):
        """
        ghost_ttl: Berapa detik nama tetap "diingat" setelah track hilang.
                   Default 5 detik — cukup untuk menutup wajah sebentar.
        """
        # Maps track_id -> name (str)
        self.cache = {}

        # Ghost cache: menyimpan nama track yang baru hilang
        # format: list of {'bbox': [...], 'name': str, 'expires_at': float}
        self._ghosts: list[dict] = []
        self.ghost_ttl = ghost_ttl

    def get(self, track_id):
        return self.cache.get(track_id, None)

    def set(self, track_id, name):
        self.cache[track_id] = name

    def remove(self, track_id):
        if track_id in self.cache:
            del self.cache[track_id]

    def clear_stale(self, active_track_ids, active_tracks_with_bbox=None):
        """
        Removes track IDs that are no longer active.
        Jika active_tracks_with_bbox disediakan (list of Track),
        track yang hilang akan disimpan ke ghost cache.
        """
        stale_ids = [tid for tid in self.cache if tid not in active_track_ids]
        for tid in stale_ids:
            name = self.cache.get(tid)
            # Simpan ke ghost hanya jika namanya sudah diketahui (bukan Unknown)
            if name and name != "Unknown" and active_tracks_with_bbox:
                # Cari bbox dari track yang baru saja expired
                for t in active_tracks_with_bbox:
                    if t.track_id == tid:
                        self._ghosts.append({
                            'bbox': list(t.bbox),
                            'name': name,
                            'expires_at': time.time() + self.ghost_ttl,
                        })
                        break
            self.remove(tid)

        # Hapus ghost yang sudah expired
        now = time.time()
        self._ghosts = [g for g in self._ghosts if g['expires_at'] > now]

    def lookup_ghost(self, bbox, iou_fn, iou_threshold: float = 0.3):
        """
        Cek apakah bbox baru ini cocok dengan ghost track yang tersimpan.
        Jika cocok (IoU >= iou_threshold), kembalikan nama-nya dan hapus ghost tsb.
        Berguna agar wajah yang muncul kembali langsung dikenali tanpa re-query DB.
        """
        now = time.time()
        best_iou = -1.0
        best_idx = -1

        for idx, ghost in enumerate(self._ghosts):
            if ghost['expires_at'] < now:
                continue
            iou = iou_fn(bbox, ghost['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            name = self._ghosts[best_idx]['name']
            self._ghosts.pop(best_idx)   # ghost dikonsumsi
            return name

        return None
