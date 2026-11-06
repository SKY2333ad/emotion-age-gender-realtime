"""Theo dõi nhẹ cho tối đa vài mặt: dự đoán vị trí, ghép toàn cục theo IoU + khoảng cách, giữ ID qua che khuất ngắn."""
IOU_WEIGHT = 0.7                 # IoU ưu tiên hơn khoảng cách; khoảng cách cứu trường hợp di chuyển nhanh, IoU = 0
MAX_COST = 1.1                   # IoU = 0 vẫn ghép được nếu tâm lệch dưới khoảng 1.3 lần cỡ mặt


def iou(first, second):
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1 = min(first[0] + first[2], second[0] + second[2])
    y1 = min(first[1] + first[3], second[1] + second[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    return overlap / max(first[2] * first[3] + second[2] * second[3] - overlap, 1e-9)


def _cost(predicted, box):
    """0 = trùng khít; khoảng cách được chuẩn hóa theo cỡ mặt để không phụ thuộc độ phân giải."""
    size = max(predicted[2], predicted[3], box[2], box[3], 1)
    gap = (((predicted[0] + predicted[2] / 2) - (box[0] + box[2] / 2)) ** 2 +
           ((predicted[1] + predicted[3] / 2) - (box[1] + box[3] / 2)) ** 2) ** 0.5
    return IOU_WEIGHT * (1 - iou(predicted, box)) + (1 - IOU_WEIGHT) * min(gap / size, 2.0)


class FaceTracker:
    """update(boxes) trả danh sách track ID cùng thứ tự với boxes; ID không tái sử dụng trong một phiên."""

    def __init__(self, max_missed=10):
        if max_missed < 0:
            raise ValueError('max_missed phải >= 0.')
        self.max_missed = max_missed
        self.tracks = {}         # id -> {'box', 'velocity', 'missed'}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def _predict(self, track):
        steps = track['missed'] + 1
        x, y, width, height = track['box']
        return (x + track['velocity'][0] * steps, y + track['velocity'][1] * steps, width, height)

    def _assign(self, ids, boxes):
        """Ghép toàn cục (nhiều cặp nhất, rồi tổng chi phí nhỏ nhất); chỉ duyệt cặp khả thi vì số mặt rất nhỏ."""
        costs = {(index, track): _cost(self._predict(self.tracks[track]), box)
                 for index, box in enumerate(boxes) for track in ids}

        def search(index, used):
            if index == len(boxes):
                return (0, 0.0), {}
            best_key, best = search(index + 1, used)
            for track in ids:
                if track not in used and costs[index, track] <= MAX_COST:
                    key, pairs = search(index + 1, used | {track})
                    key = (key[0] + 1, key[1] - costs[index, track])
                    if key > best_key:
                        best_key, best = key, {**pairs, index: track}
            return best_key, best

        return search(0, frozenset())[1]

    def update(self, boxes):
        assigned = self._assign(list(self.tracks), boxes)
        output = []
        for index, box in enumerate(boxes):
            track = assigned.get(index)
            if track is None:
                track = self.next_id
                self.next_id += 1
                self.tracks[track] = {'box': box, 'velocity': (0.0, 0.0), 'missed': 0}
            else:
                state = self.tracks[track]
                steps = state['missed'] + 1
                state['velocity'] = ((box[0] - state['box'][0]) / steps, (box[1] - state['box'][1]) / steps)
                state['box'], state['missed'] = box, 0
            output.append(track)
        for track in [key for key in self.tracks if key not in output]:
            self.tracks[track]['missed'] += 1
            if self.tracks[track]['missed'] > self.max_missed:
                del self.tracks[track]
        return output
