"""
Dummy GSM8K-style math task for smoke-testing the RL pipeline.
Uses simple Vietnamese arithmetic word problems with GSM8K-style "#### <answer>" format.
"""

import re

_PROBLEMS = [
    {"question": "An có 5 quả táo và mua thêm 3 quả. An có bao nhiêu quả táo?", "answer": 8},
    {"question": "Bình có 10 viên bi, cho bạn 4 viên. Bình còn lại bao nhiêu viên?", "answer": 6},
    {"question": "Mỗi hộp có 6 bút chì. Có 4 hộp. Tổng cộng có bao nhiêu bút chì?", "answer": 24},
    {"question": "Lớp có 30 học sinh. Vắng 5 em. Có bao nhiêu học sinh có mặt?", "answer": 25},
    {"question": "Một cửa hàng bán 12 bánh vào buổi sáng và 8 bánh vào buổi chiều. Tổng cộng bán được bao nhiêu bánh?", "answer": 20},
    {"question": "Nam có 15 tờ giấy. Nam dùng 7 tờ. Còn lại bao nhiêu tờ?", "answer": 8},
    {"question": "Mỗi bàn có 4 chiếc ghế. Có 6 bàn. Tổng cộng có bao nhiêu ghế?", "answer": 24},
    {"question": "Hương mua 3 quyển sách, mỗi quyển giá 5 nghìn đồng. Hương trả bao nhiêu nghìn đồng?", "answer": 15},
    {"question": "Một xe buýt chở 40 người. Ở trạm đầu 10 người xuống, 5 người lên. Còn bao nhiêu người trên xe?", "answer": 35},
    {"question": "Minh đếm được 7 con chim trên cây, sau đó thêm 9 con bay đến. Có bao nhiêu con chim?", "answer": 16},
    {"question": "Một kho có 50 túi gạo. Xuất kho 20 túi. Còn bao nhiêu túi?", "answer": 30},
    {"question": "Mỗi ngày Lan học 3 tiếng. Sau 5 ngày Lan học tổng cộng bao nhiêu tiếng?", "answer": 15},
    {"question": "Có 4 hàng cây, mỗi hàng 9 cây. Tổng cộng có bao nhiêu cây?", "answer": 36},
    {"question": "Tuấn có 100 đồng, mua kẹo hết 45 đồng. Tuấn còn lại bao nhiêu đồng?", "answer": 55},
    {"question": "Một thùng chứa được 20 lít nước. Đã có 8 lít. Cần thêm bao nhiêu lít để đầy thùng?", "answer": 12},
    {"question": "Lớp học có 25 em. Số em nữ là 14. Có bao nhiêu em nam?", "answer": 11},
    {"question": "Một vườn có 3 hàng cà chua, mỗi hàng 8 cây. Tổng số cây là bao nhiêu?", "answer": 24},
    {"question": "Bảo có 18 viên kẹo, ăn 5 viên rồi cho bạn 6 viên. Bảo còn bao nhiêu viên?", "answer": 7},
    {"question": "Một bể có 60 con cá. Thêm 15 con vào. Bể có bao nhiêu con cá?", "answer": 75},
    {"question": "Mỗi túi có 7 quả cam. Có 5 túi. Tổng số cam là bao nhiêu?", "answer": 35},
]

_TRAIN = _PROBLEMS[:16]
_TEST = _PROBLEMS[16:]


def _build_conversation(item):
    content = (
        f"{item['question']}\n\n"
        "Hãy giải từng bước và viết đáp án cuối cùng theo định dạng: #### <số>"
    )
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"#### {item['answer']}"},
        ],
        "answer": item["answer"],
    }


def _extract_answer(text):
    """Extract the integer after '####' in generated text."""
    match = re.search(r"####\s*([\d,]+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


class GSM8K:
    def __init__(self, subset="main", split="train"):
        self._data = _TRAIN if split == "train" else _TEST

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return _build_conversation(self._data[idx])

    def reward(self, conversation, generated_text):
        expected = conversation["answer"]
        predicted = _extract_answer(generated_text)
        return 1.0 if predicted == expected else 0.0

    def evaluate(self, conversation, generated_text):
        return self.reward(conversation, generated_text) > 0.5
