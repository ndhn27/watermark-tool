# P4 — mark ẩn cho video: đo TRƯỚC khi đụng vào `cli/video.py` (bước 3/5)

Trạng thái: **[đo trước xong, code SẢN XUẤT chưa đổi]**. `cli/video.py` hiện tại KHÔNG nhúng dấu ẩn nào -
`--invisible` bị bỏ qua cho video (README "Video support"). Video chỉ có lớp lưới chữ nhìn thấy được +
C2PA (nếu bật). Đây là lỗ hổng lớn nhất được ghi trong HANDOFF v23 mục 2.3.

Câu hỏi đặt ra trước khi viết code sản xuất: nếu nhúng `wmcore.embed_spread_spectrum()` - CHÍNH XÁC
hàm ảnh đang dùng cho `--invisible --invisible-mode spread` - vào từng khung hình rồi mã hoá thật bằng
ffmpeg/libx264, dấu còn đọc được không, đọc được bao nhiêu, và cái giá (thời gian, độ lộ pixel) là gì?

Hai script mới, chưa đụng `cli/video.py`:
- `bench/experiments/video_invisible_probe.py` - kết quả thô: `bench/results/p4-video-invisible-probe.txt`
- `bench/experiments/video_alpha_sweep.py` - kết quả thô: `bench/results/p4-video-alpha-sweep.txt`

## 1. Thiết lập [đã xác minh]

Video tổng hợp 640×480, 24fps, 60 khung (2.5s): một ảnh "pink" (kết cấu 1/f, `bench_corpus._pink`,
xem `bench/README.md`) lớn hơn khung hình, quét (pan) qua bằng chuyển động cos mượt - khung liền kề
tương quan thật, không phải nhiễu độc lập mỗi khung. `wm_id` mới mỗi lần chạy (`wmid.generate()`),
`key` cố định (`derive_seed`). Nhúng bằng `wmcore.embed_spread_spectrum(frame, wm_id, key)`, TỪNG khung
độc lập, đơn luồng (không qua worker pool hiện có của video.py). Đọc lại bằng
`wmcore.find_spread_spectrum_id(spread_spectrum_block_correlations(frame, key))`, cũng từng khung.
ffmpeg thật (`libx264`, `-preset medium`), không phải OpenCV writer fallback.

## 2. Nhúng từng khung tốn bao lâu [đã xác minh]

~60-63 ms/khung, đơn luồng, không dùng `--workers`. Ở tốc độ này một video 1080p/30fps dài vài phút sẽ
tốn hàng chục phút nếu nhúng tuần tự - **cần đưa vào worker pool đã có của video.py** (giống cách
`_blend_frame` chạy song song), không nhúng đơn luồng trong bản sản xuất. Chưa đo trên khung 1080p/4K
hay trong worker pool thật - chỉ đo đơn luồng, 640×480.

## 3. Có đọc được sau khi mã hoá thật không, alpha mặc định (12.0) [đã xác minh]

Không mã hoá gì (khung thô ngay sau nhúng): **60/60**.
Sau mã hoá thật (mỗi dòng chạy lại 2 lần, `wm_id` ngẫu nhiên mỗi lần nên số dao động nhẹ - xem §6):

| Kiểu tấn công (video-specific)      | Đọc được / 60 khung |
|--------------------------------------|----------------------|
| CRF 18 (nén nhẹ)                     | 60/60                |
| **CRF 23 (mặc định của chính x264)** | **39-42/60 (~65-70%)** |
| CRF 32 (nén nặng, kiểu repost)       | 3/60 (~5%)           |
| CRF 40 (nén rất nặng)                | 3/60 (~5%)           |
| Giảm nửa fps (24→12), CRF 23         | 32/32 (100%, khung còn lại) |
| Crop 40px chỉ ở TRÊN (thẳng khối 8px, giữ nguyên số cột khối) | 40/60 |
| Crop 40px MỖI BÊN trái+phải (đổi số cột khối) | **0/60** |
| Letterbox: đệm 40px viền đen trên+dưới (thẳng khối 8px) | 39-44/60 |

## 4. Ý nghĩa của từng dòng

- **CRF 23 chỉ ~65-70% với alpha mặc định là điểm cần sửa trước khi thêm vào sản xuất** - CRF 23 không
  phải trường hợp khắc nghiệt, nó là *mặc định* của chính libx264. §5 đo alpha cao hơn có sửa được không.
- **fps_half cao hơn hẳn CRF23 thường (100% so với ~65-70%) dù cùng CRF 23 - [chưa xác minh] vì sao.**
  Có thể do khung bị giữ lại đổi loại (I/P/B) khi cấu trúc GOP đổi theo `-r`, nhưng đây là suy đoán,
  chưa đo tách bạch. Không dựa vào số này để kết luận "giảm fps giúp đọc tốt hơn" - chỉ ghi lại, chưa hiểu.
- **Crop trái+phải (đổi số cột khối) = 0/60, khớp với giới hạn đã biết ở ảnh** (`crop_topleft_64`,
  `gate=False` trong `bench/bench_attacks.py`: "left crop changes the block-column count, scrambling the
  raster tiling"). Đây là kiểu crop rất thường gặp khi đổi tỉ lệ khung hình cho video (repost cắt hai bên
  cho vừa 9:16 chẳng hạn) - **video sẽ thừa hưởng đúng lỗ hổng này từ ảnh, không phải lỗ hổng mới**, và
  P3's cơ chế lật (thử 4 chiều) KHÔNG giải quyết được crop đổi cột khối - đây là việc khác.
- **Crop trên, letterbox: mất mát ngang với CRF23 trơn** (~65-70%), không tệ hơn thấy rõ - khớp với suy
  luận trong `spread.py` (`find_spread_spectrum_id` dò cả 112 pha, nên phép dịch khối thẳng hàng 8px chỉ
  đổi "pha", không phá tín hiệu) - **đây là dự đoán trong docstring, lần đầu được đo thật cho video**.
- **Quan trọng nhất, đổi cách nghĩ về "% khung đọc được":** ảnh chỉ cho MỘT lượt đọc; video cho một lượt
  đọc MỖI KHUNG. Nếu `extract` cho video chỉ cần đọc được **MỘT khung bất kỳ** là đủ báo "có dấu" (không
  cần mọi khung), thì ngay cả 5% (CRF 32/40) cũng gần như chắc chắn thành công trên một video thật dài
  hàng trăm/hàng nghìn khung - **không cần alpha cao để "cứu" CRF 32/40, cần thiết kế `extract` quét
  nhiều khung và dừng ở khung đầu tiên đọc được** (giống cách ảnh dò nhiều key trong ledger, dừng ở lần
  trúng đầu - `wmapp/extraction.py`). Đây là điểm cần bàn trước khi viết code, không phải điều "chưa
  xác minh" - nó là một lựa chọn thiết kế.

## 5. Tăng alpha có sửa được CRF 23 không (`video_alpha_sweep.py`) [đã xác minh]

Cùng video, CRF 23 cố định, alpha thay đổi (`embed_spread_spectrum(..., alpha=...)`), 2 lần chạy:

| alpha | Đọc được / 60 | mean\|Δpixel\| (thô, không phải đo tri giác thật) |
|-------|----------------|------------------------------|
| 12.0 (mặc định hiện tại, tinh cho JPEG ảnh) | 41-47/60 | 4.7 |
| 18.0 | **60/60 cả 2 lần** | 6.9 |
| 24.0 | 60/60 | 9.3 |
| 32.0 | 60/60 | 12.4 |
| 40.0 | 60/60 | 15.5 |

alpha=18 đã đủ để CRF 23 đạt 100/100% trong cả hai lần chạy ở đây (chưa chạy đủ nhiều lần để có khoảng
tin cậy thật). **Chưa đo alpha cao hơn có giúp CRF 32/40 hay không** - nghi ngờ là không nhiều, vì CRF
32/40 nén nặng hơn nhiều so với khoảng cách 12→18 bù được, nhưng đây là suy đoán, chưa đo.

## 6. Hạn chế của phép đo này - đọc trước khi tin số liệu

- **Chỉ 1 video tổng hợp (pink, pan mượt), 640×480, 60 khung, 2 lần chạy mỗi kiểu tấn công.** Không phải
  benchmark nhiều mẫu như `bench/run_benchmark.py` dùng cho ảnh (nhiều size, nhiều seed, ảnh thật). Số
  liệu ở đây đủ để quyết định "có đáng làm tiếp không / alpha bao nhiêu là điểm khởi đầu hợp lý", KHÔNG
  đủ để công bố "X% trên video thật".
- **Chưa thử ảnh/video thật (chỉ tổng hợp)**, chưa thử nội dung có cắt cảnh (scene cut) - dấu theo khung
  tĩnh không bị ảnh hưởng bởi logic segment/jitter-refresh của lớp lưới nhìn thấy được (chưa tích hợp).
  Chưa thử các codec khác ngoài H.264 (VP9, AV1 các nền tảng thường tự chuyển mã lại khi upload).
  Chưa thử audio bị bỏ/đổi (không liên quan đến mark hình ảnh nhưng liên quan tới pipeline ffmpeg thật).
- **`wm_id` ngẫu nhiên mỗi lần chạy** (không cố định seed cho `wmid.generate()`) → sign pattern nhúng đổi
  giữa các lần chạy dù cùng `key` → số liệu dao động nhẹ giữa các lần (39-44/60 ở CRF23/letterbox, ví dụ)
  - không phải nhiễu đo lường, là do payload đổi. Nên cố định payload nếu muốn so sánh chính xác tuyệt đối
  giữa các lần chạy sau này.
- Nhúng đơn luồng, chưa tích hợp worker pool/shared-memory của `video.py` - chi phí thời gian thật trong
  sản xuất (nhất là 1080p/4K, video dài) chưa đo.

## 7. Đề xuất cho bước "thêm" (chưa làm) [đề xuất]

Không tự làm - đây là ghi lại lựa chọn để người dùng/quyết định trước khi viết code sản xuất, đúng tinh
thần P5 (torch) của kế hoạch: đừng tự quyết những thứ ảnh hưởng lớn.

1. Alpha riêng cho video (ví dụ 18-20 thay vì 12) khi nhúng dấu ẩn cho khung hình - không dùng chung
   `DEFAULT_SS_ALPHA` của ảnh mà không hỏi, vì ảnh và video khác biệt export điển hình.
2. `extract` cho video: quét N khung (bao nhiêu, mỗi bao lâu - cả video hay lấy mẫu?), dừng ở khung đầu
   tiên đọc được ID hợp lệ, giống cách ảnh dừng ở key đầu tiên trúng trong ledger - CHỨ KHÔNG cần mọi
   khung đều đọc được.
3. Nhúng cần chạy qua worker pool có sẵn của `video.py` (giống `_blend_frame`), không đơn luồng - 60ms/
   khung đơn luồng quá chậm cho video dài/độ phân giải cao.
4. Vẫn CHƯA rõ: nhúng mỗi khung riêng biệt (payload/key giống hệt mọi khung, như phép đo ở đây) hay theo
   segment như lớp lưới nhìn thấy (`jitter_refresh_seconds`) - segment đổi pattern để chống collusion/
   trung bình cộng nhiều khung; nhưng dấu ẩn bản thân nó đã cố định vì cần đọc lại bằng đúng 1 `key`, nên
   collusion-theo-thời-gian có áp dụng được ở đây không cần cân nhắc riêng, khác lý do đổi jitter của lớp
   nhìn thấy. Chưa đo, chưa quyết.
5. Crop đổi số cột khối (0/60, §4) vẫn là lỗ hổng mở, không có gì trong kế hoạch 5 bước hiện giải quyết
   nó (P3 giải quyết lật, không phải crop lệch khối).

## 8. Đo SAU khi viết code sản xuất (v25) [đã xác minh]

Người dùng đã quyết cả 4 điểm ở §7: alpha riêng cho video (Ok) → `DEFAULT_SS_ALPHA_VIDEO = 20.0`
trong `wmcore/spread.py`; `extract` quét hết mọi khung, báo cả breakdown (Quét hết) →
`video.scan_invisible()` + `wmapp.extraction._extract_video()`; key đổi theo đoạn thời gian
(Thay đổi) → key xoay theo `--jitter-refresh-seconds`, lịch cố định riêng, KHÔNG theo boundary do
scene-cut của lớp lưới (xem docstring `apply_watermark_video`); crop đổi cột khối để đó (Nâng cấp
sau) → chưa đụng, vẫn 0/60 như §4.

Test qua ĐÚNG pipeline sản xuất (`wmapp.watermark()` → `cli/video.py` → ffmpeg thật, không gọi
thẳng `wmcore.embed_spread_spectrum()` như probe ở §1-§5 nữa), video tổng hợp 640×480/24fps/60
khung, mã hoá lần đầu của chính tool (CRF 18, mặc định `_ffmpeg_encode_pipe`), rồi giả lập bị
re-up (nén lại lần 2, kiểu một nền tảng mạng xã hội mã hoá lại) ở nhiều mức:

| Kịch bản | Đọc được / 60 |
|----------|----------------|
| Ngay sau tool (CRF18, chưa ai đụng vào) | 60/60 |
| Bị re-up ở CRF18 (nén lại nhẹ) | 60/60 |
| Bị re-up ở CRF23 (mặc định x264 - trước đây với alpha 12.0 chỉ ~65-70%, giờ với alpha 20.0) | **60/60** |
| Bị re-up ở CRF32 (nén khá nặng) | 32/60 (~53%) |
| Bị re-up ở CRF40 (nén rất nặng, NÉN HAI LẦN vì đã qua CRF18 trước đó) | 0/60 |

Alpha 20.0 giải quyết đúng vấn đề đo được ở §5 (CRF23 giờ 100% dù bị nén hai lần, không phải một
lần như phép đo trước). CRF40 (nén hai lần) tệ hơn phép đo đơn lẻ ở §3 (3/60) - hợp lý vì đây là
nén chồng nén, chưa đo tách bạch phần nào (nén lần 1 ở CRF18 hay lần 2 ở CRF40) đóng góp bao nhiêu
vào mất mát; không cần thiết cho quyết định hiện tại (nén hai lần CRF40 là trường hợp cực đoan,
video thật dài hơn 60 khung nhiều nên vẫn có khả năng cao trúng ít nhất 1 khung).

Thời gian nhúng qua pipeline thật (4 worker, 640×480, 60 khung): **~8.1s tổng** (~135ms/khung quy
đổi, chậm hơn 60ms/khung đơn luồng đo ở §2 chia cho 4 - có overhead chưa tách bạch: shared-memory
transport, ffmpeg mux, hay chính việc PIL Image round-trip trong mỗi worker - **[chưa xác minh]**
overhead nằm ở đâu, chưa cần thiết phải biết cho quyết định hiện tại).

`pytest tests/` đầy đủ (kể cả `test_frontends_parity`): **325 passed, 5 failed**. 5 lỗi là C2PA
(`test_c2pa.py`, `test_id_ledger.py`, `test_wmapp.py::test_c2pa_signing_reports_signed_and_a_manifest_appears`)
- đã xác minh KHÔNG PHẢI do sửa đổi phiên này: chạy y hệt bộ test đó trên bản zip v23 gốc, CHƯA
sửa gì, cũng ra đúng 5 lỗi giống hệt ("assertion.action.softBindingMissing: watermark action
missing soft binding assertion" - `c2pa-python` 0.37.12 trong sandbox này có vẻ khắt khe hơn code
đang viết cho). Môi trường, không phải code - không sửa (ngoài phạm vi phiên này).

6 test mới `tests/test_video_invisible.py` (round trip qua pipeline thật, báo cáo đủ khung khi
không tìm thấy, key thực sự đổi theo đoạn - không phải trùng hợp, `lsb` bị bỏ qua có cảnh báo,
không có key/ledger thì không tìm gì cả).

README (`Video support`, bảng Options, mục Watermark ID & ledger, mục C2PA, mục Removal-resistance
features) đã cập nhật khớp code mới.
