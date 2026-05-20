"""
Minimal multiple-choice eval dataset for smoke-testing the ChatCORE metric.
Questions are formatted as MMLU-style: 4 options, one correct answer (0-indexed).
"""

_MC_QUESTIONS = [
    {
        "question": "Thủ đô của Việt Nam là gì?",
        "choices": ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Huế"],
        "answer": 0,
    },
    {
        "question": "2 + 2 bằng bao nhiêu?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
    },
    {
        "question": "Nước sôi ở nhiệt độ bao nhiêu độ C ở áp suất tiêu chuẩn?",
        "choices": ["50°C", "75°C", "100°C", "120°C"],
        "answer": 2,
    },
    {
        "question": "Bầu trời có màu gì vào ban ngày khi trời quang?",
        "choices": ["Đỏ", "Xanh lam", "Vàng", "Tím"],
        "answer": 1,
    },
    {
        "question": "Ngôn ngữ lập trình nào thường được dùng trong khoa học dữ liệu?",
        "choices": ["COBOL", "Pascal", "Python", "Assembly"],
        "answer": 2,
    },
    {
        "question": "Sông nào được coi là sông dài nhất thế giới?",
        "choices": ["Amazon", "Nile", "Yangtze", "Mississippi"],
        "answer": 1,
    },
    {
        "question": "Đơn vị cơ bản của thông tin trong máy tính là gì?",
        "choices": ["Byte", "Kilobyte", "Bit", "Megabyte"],
        "answer": 2,
    },
    {
        "question": "Công thức tính diện tích hình tròn là gì?",
        "choices": ["2πr", "πr²", "πd", "r²"],
        "answer": 1,
    },
    {
        "question": "Trí tuệ nhân tạo viết tắt là gì?",
        "choices": ["IT", "ML", "AI", "DL"],
        "answer": 2,
    },
    {
        "question": "Chủ tịch đầu tiên của nước Việt Nam Dân chủ Cộng hòa là ai?",
        "choices": ["Võ Nguyên Giáp", "Phạm Văn Đồng", "Hồ Chí Minh", "Lê Duẩn"],
        "answer": 2,
    },
    {
        "question": "Nguyên tử là đơn vị cơ bản của?",
        "choices": ["Tế bào", "Vật chất", "Năng lượng", "Sóng"],
        "answer": 1,
    },
    {
        "question": "Bao nhiêu giờ trong một ngày?",
        "choices": ["12", "18", "24", "48"],
        "answer": 2,
    },
    {
        "question": "Ngôn ngữ chính thức của Brazil là gì?",
        "choices": ["Tây Ban Nha", "Bồ Đào Nha", "Anh", "Pháp"],
        "answer": 1,
    },
    {
        "question": "Hành tinh gần Mặt Trời nhất là gì?",
        "choices": ["Sao Kim", "Sao Hỏa", "Trái Đất", "Sao Thủy"],
        "answer": 3,
    },
    {
        "question": "Tốc độ ánh sáng trong chân không xấp xỉ bao nhiêu?",
        "choices": ["300,000 km/s", "150,000 km/s", "30,000 km/s", "3,000 km/s"],
        "answer": 0,
    },
    {
        "question": "HTTP là viết tắt của?",
        "choices": [
            "HyperText Transfer Protocol",
            "High Transfer Text Protocol",
            "HyperText Transport Process",
            "Hyperlink Text Transfer Protocol",
        ],
        "answer": 0,
    },
]

_LABELS = ["A", "B", "C", "D"]


def format_mc_prompt(item):
    """Format a multiple-choice item as a conversation dict."""
    choices_text = "\n".join(f"{_LABELS[i]}. {c}" for i, c in enumerate(item["choices"]))
    content = (
        f"{item['question']}\n\n{choices_text}\n\n"
        "Hãy chọn đáp án đúng. Chỉ trả lời bằng một chữ cái: A, B, C hoặc D."
    )
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": _LABELS[item["answer"]]},
        ]
    }


def run_sample_mc_eval(model, tokenizer, engine, max_problems=None):
    """
    Evaluate model on sample multiple-choice questions.
    Uses greedy generation — checks if the first meaningful token matches the correct label.
    Returns accuracy (float 0-1).
    """
    import torch

    questions = _MC_QUESTIONS
    if max_problems is not None:
        questions = questions[:max_problems]

    label_ids = {label: tokenizer.encode(label)[0] for label in _LABELS}

    correct = 0
    for item in questions:
        prompt_ids = tokenizer.render_for_completion(format_mc_prompt(item))
        tokens = torch.tensor([prompt_ids], dtype=torch.long)
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=4, temperature=0)
        generated = tokenizer.decode(sample[0]).strip()
        # Take the first letter that is a valid label
        predicted = next((c for c in generated.upper() if c in _LABELS), None)
        if predicted == _LABELS[item["answer"]]:
            correct += 1

    return correct / len(questions) if questions else 0.0
