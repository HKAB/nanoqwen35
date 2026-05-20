"""
Minimal sample SFT dataset for smoke-testing the training pipeline.
Each item is a conversation dict with {"messages": [...]} matching
the format expected by tokenizer.render_conversation.
"""

_CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "Xin chào! Bạn có thể giúp tôi không?"},
            {"role": "assistant", "content": "Xin chào! Tôi sẵn sàng giúp bạn. Bạn cần hỗ trợ gì?"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Thủ đô của Việt Nam là gì?"},
            {"role": "assistant", "content": "Thủ đô của Việt Nam là Hà Nội."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "2 cộng 2 bằng bao nhiêu?"},
            {"role": "assistant", "content": "2 cộng 2 bằng 4."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Viết cho tôi một bài thơ ngắn về mùa xuân."},
            {"role": "assistant", "content": "Mùa xuân đến, hoa đào nở rộ,\nGió nhẹ thổi, chim hót líu lo.\nĐất trời tươi sáng, lòng người vui,\nMùa xuân mang lại bao hy vọng mới."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Python là ngôn ngữ lập trình gì?"},
            {"role": "assistant", "content": "Python là ngôn ngữ lập trình bậc cao, đa mục đích, nổi tiếng với cú pháp dễ đọc và dễ học. Python được dùng rộng rãi trong khoa học dữ liệu, trí tuệ nhân tạo, phát triển web và tự động hóa."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Cho tôi biết về lịch sử Việt Nam."},
            {"role": "assistant", "content": "Việt Nam có lịch sử hơn 4000 năm. Nhà nước Văn Lang được coi là nhà nước đầu tiên của người Việt. Qua nhiều thế kỷ, Việt Nam trải qua các triều đại như Lý, Trần, Lê và các cuộc kháng chiến bảo vệ đất nước. Ngày 2/9/1945, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập, khai sinh nước Việt Nam Dân chủ Cộng hòa."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Làm thế nào để nấu phở bò?"},
            {"role": "assistant", "content": "Để nấu phở bò ngon, bạn cần: xương bò, thịt bò, bánh phở, hành, gừng và các gia vị như hoa hồi, quế, thảo quả. Hầm xương trong 6-8 tiếng để lấy nước dùng trong và ngọt. Nướng hành và gừng trước khi cho vào nồi để tăng hương vị. Thêm gia vị cho vừa khẩu vị và dùng kèm với rau thơm, chanh, ớt."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Giải thích trí tuệ nhân tạo là gì."},
            {"role": "assistant", "content": "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các tác vụ thường đòi hỏi trí thông minh của con người, như nhận dạng giọng nói, hiểu ngôn ngữ tự nhiên, ra quyết định và học hỏi từ kinh nghiệm."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Sông nào dài nhất thế giới?"},
            {"role": "assistant", "content": "Sông Nile ở châu Phi thường được coi là sông dài nhất thế giới với chiều dài khoảng 6.650 km, chảy qua nhiều quốc gia châu Phi trước khi đổ ra Địa Trung Hải."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Bạn có thể dịch 'hello' sang tiếng Việt không?"},
            {"role": "assistant", "content": "'Hello' trong tiếng Việt có thể dịch là 'Xin chào' hoặc đơn giản hơn là 'Chào'. Tùy ngữ cảnh, bạn có thể dùng 'Chào buổi sáng', 'Chào buổi chiều' hoặc 'Chào buổi tối'."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tại sao bầu trời màu xanh?"},
            {"role": "assistant", "content": "Bầu trời màu xanh do hiện tượng tán xạ Rayleigh. Ánh sáng mặt trời chứa nhiều màu sắc khác nhau. Khi đi qua khí quyển, ánh sáng xanh bị tán xạ nhiều hơn các màu khác do có bước sóng ngắn hơn, khiến bầu trời trông có màu xanh lam."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Cho tôi một công thức toán học đơn giản."},
            {"role": "assistant", "content": "Đây là một công thức cơ bản: Diện tích hình tròn = π × r², trong đó r là bán kính của hình tròn và π ≈ 3.14159. Ví dụ, hình tròn có bán kính 5 cm sẽ có diện tích ≈ 78.54 cm²."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Kể cho tôi nghe một câu chuyện ngắn."},
            {"role": "assistant", "content": "Ngày xưa, có một cậu bé nghèo tên Minh sống cùng bà nội. Mỗi ngày cậu đi học xa hàng chục cây số. Dù vất vả, cậu không bao giờ bỏ học. Với sự kiên trì và nỗ lực, cậu đã đỗ đại học và trở thành kỹ sư giỏi, xây dựng cuộc sống tốt đẹp cho gia đình. Câu chuyện dạy ta rằng: kiên trì và chăm chỉ sẽ mang lại thành công."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Nước sôi ở nhiệt độ bao nhiêu?"},
            {"role": "assistant", "content": "Nước sôi ở 100°C (212°F) ở điều kiện áp suất khí quyển tiêu chuẩn (1 atm). Tuy nhiên, nhiệt độ sôi có thể thay đổi theo áp suất: ở nơi cao hơn mực nước biển, nước sôi ở nhiệt độ thấp hơn; trong nồi áp suất, nước sôi ở nhiệt độ cao hơn."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Giải thích khái niệm học máy."},
            {"role": "assistant", "content": "Học máy (Machine Learning) là nhánh của trí tuệ nhân tạo, cho phép máy tính học hỏi từ dữ liệu mà không cần lập trình cụ thể cho từng nhiệm vụ. Thay vì viết quy tắc, chúng ta cung cấp dữ liệu để mô hình tự tìm ra các mẫu và quy luật. Có ba loại chính: học có giám sát, học không có giám sát và học tăng cường."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hà Nội có những địa điểm du lịch nổi tiếng nào?"},
            {"role": "assistant", "content": "Hà Nội có nhiều địa điểm du lịch nổi tiếng như: Hồ Hoàn Kiếm với Tháp Rùa, Đền Ngọc Sơn; Văn Miếu - Quốc Tử Giám; Lăng Chủ tịch Hồ Chí Minh; Phố cổ Hà Nội với 36 phường nghề; Hồ Tây rộng lớn; Chùa Một Cột độc đáo và Hoàng Thành Thăng Long - di sản UNESCO."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Tôi nên học lập trình từ đâu?"},
            {"role": "assistant", "content": "Để bắt đầu học lập trình, bạn có thể: (1) Chọn ngôn ngữ dễ học như Python hoặc JavaScript; (2) Dùng tài nguyên miễn phí như freeCodeCamp, Codecademy, W3Schools; (3) Xem video trên YouTube hoặc học khóa học trên Coursera, edX; (4) Thực hành bằng cách làm các dự án nhỏ; (5) Tham gia cộng đồng lập trình để nhận hỗ trợ. Quan trọng nhất là kiên trì luyện tập mỗi ngày."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Biến đổi khí hậu là gì và tại sao nó quan trọng?"},
            {"role": "assistant", "content": "Biến đổi khí hậu là sự thay đổi lâu dài của nhiệt độ và các mô hình thời tiết trên Trái Đất. Hiện nay, hoạt động của con người như đốt nhiên liệu hóa thạch là nguyên nhân chính gây ra hiện tượng này. Biến đổi khí hậu quan trọng vì nó gây ra: nước biển dâng, thiên tai cực đoan, mất đa dạng sinh học, ảnh hưởng đến sản xuất nông nghiệp và sức khỏe con người."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Cách tính diện tích tam giác?"},
            {"role": "assistant", "content": "Diện tích tam giác được tính theo công thức: S = (đáy × chiều cao) / 2. Ví dụ: tam giác có đáy 6 cm và chiều cao 4 cm sẽ có diện tích = (6 × 4) / 2 = 12 cm². Nếu biết ba cạnh a, b, c, bạn có thể dùng công thức Heron: S = √(p(p-a)(p-b)(p-c)) trong đó p = (a+b+c)/2."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Cho tôi lời khuyên để sống khỏe mạnh."},
            {"role": "assistant", "content": "Để sống khỏe mạnh, bạn nên: (1) Ăn uống cân bằng với nhiều rau củ, trái cây, protein và ngũ cốc nguyên hạt; (2) Uống đủ 2 lít nước mỗi ngày; (3) Tập thể dục ít nhất 30 phút/ngày; (4) Ngủ đủ 7-8 tiếng mỗi đêm; (5) Tránh căng thẳng bằng thiền định hoặc yoga; (6) Không hút thuốc và hạn chế rượu bia; (7) Kiểm tra sức khỏe định kỳ."},
        ]
    },
]


class SampleSFT:
    """
    Minimal in-memory SFT dataset for smoke-testing.
    Returns conversation dicts compatible with tokenizer.render_conversation.
    """
    def __init__(self, split="train"):
        # Use first 16 items for train, last 4 for val
        if split == "train":
            self._data = _CONVERSATIONS[:16]
        else:
            self._data = _CONVERSATIONS[16:]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx % len(self._data)]
