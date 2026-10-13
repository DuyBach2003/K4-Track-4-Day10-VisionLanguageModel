Họ tên: <điền họ tên>
`dataset_version`: d10-21e4070301

Cách chạy lại: mở `lab_day10.ipynb` trên Colab T4 và chạy từ trên xuống. Mã nằm trong `lab/` (`labels.py`, `vlm.py`, `run.py`, `reward.py`, `grpo.py`, `viz.py`, `summarize.py`), test của mình ở `lab/tests/` (`python -m pytest -q lab/tests`).

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

<điền: dán output của `python -m lab.summarize` sau khi chạy xong E0–E4 trên Colab>

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | | | | | | | | |
| E1 | LoRA LLM | 33,0M | | | | | | | | |
| E2 | + projector | 142,1M | | | | | | | | |
| E3 | + LoRA ViT | 148,4M | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm trên cây module thật của `Qwen/Qwen3-VL-4B-Instruct` (transformers 5.19, dựng trên meta device):

| Khối | Tên module | Tham số |
|---|---|---|
| ViT (tổng) | `model.visual` | 415,3M |
| └ 24 khối transformer | `model.visual.blocks` | 302,3M |
| └ projector chính | `model.visual.merger` | 27,3M |
| └ projector DeepStack (lớp 5, 11, 17) | `model.visual.deepstack_merger_list` | 81,8M |
| └ patch_embed + pos_embed | | 3,9M |
| LLM (36 lớp, kể cả embedding 389,0M) | `model.language_model` | 4.022,5M |
| `lm_head` | dùng chung trọng số với `embed_tokens` | (0) |
| **Tổng** | | **4.437,8M** |

Khối được cập nhật (số tham số huấn luyện do `lab.vlm.apply_setup` in ra):

- **E1**: chỉ LoRA r=16 trên q/k/v/o/gate/up/down của 36 lớp LLM = 252 module, **33.030.144** tham số. ViT và projector đông lạnh.
- **E2**: E1 + mở toàn bộ projector (`merger` + `deepstack_merger_list`) ở fp32 = **142.135.296** tham số (hơn E1 đúng 109.105.152).
- **E3**: E2 + LoRA trên qkv/proj/linear_fc1/linear_fc2 của 24 khối ViT (thêm 96 module, tổng 348) = **148.426.752** tham số (LoRA ViT chỉ thêm 6,3M).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
<điền sau khi chạy E0: parse_valid_rate trong report/evidence/e0.json, 5 output thô in ở notebook bước 3>

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Mình kiểm chứng bằng **dự đoán oracle** (chuyển chính nhãn đúng của 311 ảnh test thành JSON) rồi đưa qua `scripts/score.py` (`lab/tests/test_labels.py::test_oracle_answers_through_scorer`):

| Nhãn sinh theo | Chấm với `--coord` | F1 | mean IoU |
|---|---|---|---|
| norm1000 | norm1000 | **1,000** | 0,988 (chỉ mất do làm tròn số nguyên) |
| pixel | norm1000 | **0,000** | 0 |
| norm1000 | pixel | **0,000** | 0 |

Lệch thang làm F1 về **đúng 0**, không phải "giảm một chút": ảnh 640×360 nên toạ độ pixel bị hiểu thành 0–1000 sẽ co box về góc trên-trái (x nhân 0,64, y nhân 0,36), và chiều ngược lại thì phóng box ra ngoài khung. Box nhỏ (trung vị ≈ 0,6–1% khung hình) nên IoU với box đúng gần như luôn < 0,5. JSON vẫn hợp lệ 100%, nên triệu chứng là "parse_valid_rate cao mà F1 ≈ 0".

Qwen3-VL được pretrain grounding trên thang tương đối 0–1000, độc lập với độ phân giải. Model không thấy ảnh 640×360 gốc mà thấy ảnh đã resize về 640×352 (lưới 40×22 patch 16 px → 220 token sau khi gộp 2×2). Toạ độ tương đối không đổi khi resize, còn toạ độ pixel thì model phải tự biết kích thước gốc.

<điền sau khi chạy E4 coord_scale: F1 thật khi huấn luyện + chấm bằng pixel so với norm1000>

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Với ảnh train đầu tiên (9 box): chuỗi đầy đủ có **625 token**, gồm 220 token `<|image_pad|>`, khoảng 155 token prompt/khung chat và chỉ **250 token đáp án** (JSON + `<|im_end|>\n`) được tính loss. Trung vị độ dài đáp án ở train là 56 token, tức với ảnh điển hình chỉ khoảng 13% chuỗi là phần cần học. Mình kiểm tra bằng cách giải mã riêng các token có `labels != -100`: kết quả đúng bằng JSON đáp án cộng `<|im_end|>`.

Nếu tính loss trên cả chuỗi:
- **Token ảnh**: `<|image_pad|>` là placeholder cố định, embedding của nó bị thay bằng đặc trưng ViT. Dự đoán "token kế tiếp là `<|image_pad|>`" là bài toán vô nghĩa nhưng chiếm 220/625 vị trí, làm loãng gradient.
- **Prompt**: prompt giống hệt nhau ở mọi ví dụ, model sẽ học thuộc nó (loss về ≈ 0 rất nhanh). Đường loss tổng giảm đẹp nhưng gần như không phản ánh việc học box, và một phần dung lượng LoRA bị dùng để mô hình hoá prompt thay vì toạ độ.
- Tín hiệu thật (toạ độ, nhãn) chỉ là một phần nhỏ của tổng loss, nên với cùng 300 bước model học grounding chậm hơn.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

- **4-bit không huấn luyện được trực tiếp**: `Linear4bit` lưu trọng số NF4 đã đóng gói (uint8 + hệ số scale theo block), không phải tensor float liên tục. Gradient theo các mã 4-bit không có nghĩa, optimizer không cập nhật được. QLoRA chỉ huấn luyện adapter float đặt *bên cạnh* lớp 4-bit. Muốn mở hoàn toàn 109,1M tham số của `merger` + `deepstack_merger_list` thì các lớp này phải ở dạng float.
- **Lỗi mình gặp**: pseudo code ghi `llm_int8_skip_modules=["visual", "lm_head"]`, nhưng với transformers 5.19, `should_convert_module()` khớp mẫu bằng `re.match` (neo ở **đầu** tên). Tên thật là `model.visual.blocks.0.attn.qkv`, nên `"visual"` **không khớp**, và toàn bộ ViT kể cả projector **vẫn bị lượng tử hoá**. Mình kiểm chứng trực tiếp bằng hàm của transformers: với `["visual","lm_head"]` thì `model.visual.merger.linear_fc1` trả về "convert = True". Cách sửa: dùng `["model.visual", "lm_head"]` (`lab/vlm.py: SKIP_QUANT`), và `check_quantization()` assert sau khi nạp rằng không có `Linear4bit` nào trong `visual` hay `lm_head`.
- **Vì sao fp32**: huấn luyện fp16 dùng `GradScaler`, mà `scaler.unscale_` từ chối gradient fp16 ("Attempting to unscale FP16 gradients"). Quan trọng hơn, cập nhật AdamW với lr 2e-4 trên trọng số fp16 (10 bit mantissa) thường nhỏ hơn bước lượng tử của số fp16 nên bị làm tròn mất. Giữ master weight fp32, còn phép nhân trong forward vẫn chạy fp16 nhờ autocast. Adapter LoRA cũng được ép về fp32 vì lý do này.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1.422 | 13 | 1.074 | 1.750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

Mình xác nhận không phiên nào xuất hiện ở hai split (giao các tập `session` đều rỗng).

Ảnh là frame cắt từ video dashcam, nên hai frame liền nhau gần như cùng con đường, cùng ổ gà, box lệch vài pixel. Chia ngẫu nhiên thì hầu hết ảnh test có "anh em sinh đôi" trong train. Model chỉ cần nhớ cảnh là đoán đúng box, và F1 đo khả năng *ghi nhớ* chứ không phải *tổng quát hoá* sang con đường chưa thấy. Đó là F1 cao giả, sẽ sụp khi triển khai trên đường mới. Chia theo phiên cho test là những đoạn đường model chưa từng thấy.

Cái giá: phân bố lớp lệch (val chỉ có 66 pothole; trung vị diện tích box crack ở test là 1,98% khung hình so với 1,35% ở train). Và tập test thực chất chỉ có 3 phiên, riêng `vlc-2025-02-18-23` chiếm 213/311 ảnh (xem câu 8).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền sau khi chạy E1–E3>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền độ rộng CI95 thật của E1–E3 và có chồng lấn không>

Lưu ý khi đọc CI: `score.py` bootstrap **theo ảnh**, coi 311 ảnh là độc lập. Nhưng 311 ảnh đến từ chỉ 3 phiên (213 + 83 + 15), các frame liền nhau tương quan mạnh, nên số mẫu hiệu dụng nhỏ hơn 311 nhiều và CI thật còn **rộng hơn** con số in ra.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền sau khi chạy: F1 từng lớp + report/figures/e*_examples.png>

Số liệu dữ liệu để đối chiếu: trung vị diện tích box ở train là pothole 0,38%, manhole 0,39%, crack 1,35% khung hình. Ở test là pothole 0,42%, manhole 0,67%, crack 1,98%. Crack có ranh giới mơ hồ (vết nứt dài, box chú thích không nhất quán), còn pothole/manhole thì rất nhỏ.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền sau khi chạy E4>

## Nhìn lại

<điền>

Ghi nhận đến giờ: (1) pseudo code `llm_int8_skip_modules=["visual", ...]` âm thầm lượng tử hoá cả ViT trên transformers 5.19, và không công cụ chấm nào bắt được lỗi này. Nếu không assert, bước "mở projector" ở E2/E3 sẽ thao tác trên trọng số NF4 đã đóng gói thay vì fp16 (xem câu 5). (2) Tập test chỉ có 3 phiên nên CI bootstrap theo ảnh là lạc quan.
