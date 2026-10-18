# Crop resync: đọc ID từ bản bị CROP (chụp màn hình rồi cắt) — v27, bổ sung v28 (§5b, §8, §9, §10)

Mục tiêu người dùng nêu ở phiên này: chống bị lách bằng "chụp ảnh màn hình rồi crop". Đo trước khi viết, như các bước trước.
Nhãn: **[đã xác minh]** = đã chạy/đo thật. **[chưa xác minh]** = còn phải kiểm tra. Script đo: `bench/experiments/crop_resync_probe.py`
(`--exp aligned|unaligned|null|pipeline|limits`). Mã sản xuất: `cli/wmcore/resync.py`; nối vào `wmapp/extraction.py`,
`video.py`; test: `tests/test_resync.py`.

## 1. Lỗ hổng trước khi sửa [đã xác minh]

Một ảnh tổng hợp 1280×960, đóng dấu spread + ID, `wmapp.extract()` bản v26, 7 tình huống (1 ảnh mỗi tình huống):
đối chứng đọc được; crop 40 px phía trên (bội của 8) đọc được; **crop 37 px phía trên, crop trái 40 px, crop lệch cả trái
và trên, crop lệch + JPEG q85, và thêm thu nhỏ 90%: không đọc được cả 5**. Số cũ trong `P2-NOTES.md` khớp: crop trái và crop
lệch đều 0/48. Nghĩa là lớp ẩn chính chỉ sống sót với crop cắt phần trên theo bội của 8 - mà ảnh chụp màn hình rồi crop gần như
luôn bị lệch. Sau khi sửa, cùng 7 tình huống: **6/7 đọc được**; còn lại là "crop + thu nhỏ 90%" (xem §7).

## 2. Cách làm (vì sao crop làm hỏng, và vì sao sửa được)

Dấu được xếp theo thứ tự raster trên lưới khối 8×8 của ẢNH GỐC, chu kỳ 112 bit. Bit của khối (r, c) trong bản crop là bit
`((r*s + c + phi) mod 112)` với `s` = (số khối theo chiều ngang của ảnh gốc) mod 112 và `phi` = phần bù do cắt trên/trái. Chỉ có
cặp (s, phi) quan trọng: 112×112 giả thuyết. Hai việc cần làm:
1. **Tìm lại lưới 8×8** (khi crop lệch bội của 8): tương quan của một cửa sổ 8×8 với dãy trải phổ của khoá là một phép
   tương quan 2-D tuyến tính với MỘT ma trận 8×8 cho mỗi khoá, nên một lần `cv2.filter2D` cho tương quan tại cả 64 độ lệch cùng lúc
   (kiểm tra: khớp với phép DCT từng khối đến ~2e-4; 8 ms ở 1280×960, 21 ms ở 3000×2000, so với 0.1-0.8 s cho một lần DCT).
   Độ lệch đúng là độ lệch có bình phương tương quan trung bình lớn nhất; thử độ lệch (0,0) và 3 độ lệch điểm cao nhất.
2. **Dò (s, phi)**: gấp bảng tương quan theo modulo 112 hai chiều một lần (bảng 112×112, bất kể cỡ ảnh), rồi mỗi giả thuyết là một
   phép tra bảng: cả 12,544 giả thuyết ~20 ms. Bộ giải mã cũ chính là trường hợp s = (chiều rộng ảnh) mod 112.

Chấp nhận kết quả: chỉ một ID chuẩn, đúng mã kiểm tra (như mọi nơi khác). Chỉ đọc ID, không đọc payload chữ (cùng lý do với ảnh lật).

Hai điều mình làm sai/đi vòng trong lúc làm, ghi lại để khỏi lặp:
- Bản thăm dò đầu dùng phát hiện lệch bằng DCT trên một mảnh 256 px cho 64 độ lệch (~420 ms, hơn cả một lần giải mã đầy đủ). Đổi sang bộ lọc 8×8 (8-21 ms,
  và chính xác trên cả ảnh thay vì một mảnh).
- Docstring đầu của `resync.py` nói giới hạn chiều rộng áp dụng cho "bội của 56 khối" bị cắt BẤT KỲ cột nào. Sai: xem §4 cho quy tắc đã kiểm chứng.

## 3. Số đo [đã xác minh, n nhỏ, một CPU]

### 3.1 Qua đúng mã sản xuất (`--exp pipeline`), số lần đọc được / số lần thử, bộ giải mã cũ → resync
Ảnh: tổng hợp "pink" 640×480 (6 ảnh × 4 crop = 24), pink 1280×960 (2 ảnh × 4 = 8), 4 ảnh mẫu thật (astronaut, coffee, rocket, hubble_deep_field;
4 crop mỗi ảnh). Mỗi nhóm đo ở 3 mức: không JPEG, JPEG q85, q60. Phát hiện độ lệch do bộ lọc tự chọn (không dùng đáp án).

| Loại crop | cũ | resync |
|---|---|---|
| chỉ trên/dưới, theo bội của 8 | tất cả đọc được (24/24, 8/8, 4/4 mỗi ảnh thật) | tất cả đọc được, cùng số |
| trái/phải, theo bội của 8 | **0** ở mọi nhóm và mức JPEG | **tất cả** (24/24, 8/8, 4/4 mỗi ảnh thật), ở cả ba mức JPEG |
| độ lệch bất kỳ (kiểu chụp màn hình) | **0** ở mọi nhóm | **tất cả**, ở cả ba mức JPEG |

Thời gian một lần giải mã resync (ms): ~17-19 (640×480, bội của 8), 26-34 (lệch); 25-28 / 35-42 (1280×960); ảnh thật 15-37.
Lật rồi crop lệch: pink 640×480 lật trái-phải / trên-dưới / cả hai 12/12 mỗi loại; astronaut 2/2 mỗi loại.

### 3.2 Giới hạn (`--exp limits`), pink 1280×960, crop lệch, giữ một tỷ lệ DIỆN TÍCH, 3 ảnh × 3 crop
JPEG q85 / q60 / q40 / q30: **9/9 ở mọi mức diện tích 80%, 50%, 25%, 10%, 5%** (đối chứng ảnh nguyên: 3/3 mỗi mức JPEG). Mẫu nhỏ, một loại ảnh tổng hợp;
5% diện tích của 1280×960 vẫn còn ~960 khối (~8.6 chu kỳ). Với ảnh mẫu thật chỉ đo crop vừa phải (0-20% mỗi cạnh).

### 3.3 Thời gian `wmapp.extract()` (ledger 1 khoá), bản v25 → bản này, nhỏ nhất của 3 lần
| | 640×480 | 3000×2000 |
|---|---|---|
| ảnh có dấu, không crop | 0.08 → 0.10 s | 1.65 → 1.60 s |
| ảnh KHÔNG có dấu | 0.21 → **0.53 s** | 4.96 → **5.82 s** |
| crop lệch 37/23 | 0.17 (không đọc được) → 0.51 s (đọc được) | 4.45 (không đọc được) → 4.91 s (đọc được) |

Ảnh không dấu và ảnh crop chậm hơn vì bộ giải mã cũ vẫn chạy trước (gồm 3 lượt DCT cho 3 hướng lật ở 6 MP) rồi resync mới chạy. Chỉ riêng resync: 0.14 s
đọc xong một ảnh 3000×2000 crop lệch; 0.55 s cho cả tìm kiếm (4 hướng × 4 độ lệch) trên ảnh không dấu cùng cỡ. **[đề xuất, chưa làm]** 3 lượt lật của bộ
cũ là tập con của không gian giả thuyết của resync (s = bề rộng, lệch (0,0)); thay chúng bằng resync sẽ cắt ~3 s ở 6 MP cho ảnh không dấu/crop, nhưng chạm vào thứ tự đã ghim
của P3 (`test_orient.py`) nên chưa làm.

### 3.4 Dương tính giả (`--exp pipeline`, phần cuối) [đã xác minh]
55 ảnh KHÔNG có dấu (pink 640×480 ×20, pink 1280×960 ×6, gradient+nhiễu ×20, 4 ảnh cấu trúc: phẳng / sọc 8 px / bàn cờ 8 px / khối 8×8 ngẫu nhiên, 5 ảnh mẫu thật),
khoá ngẫu nhiên, 4 hướng × 4 độ lệch × 12,544 giả thuyết mỗi ảnh: **0 ID giả**. Lý thuyết ~1e-5 kỳ vọng cho cả 55 ảnh (2^-40 mỗi giả thuyết). Trung bình 255 ms/ảnh.
Tất nhiên 0/55 không chứng minh tỷ lệ thật bằng 1e-8 - nó chỉ khớp lý thuyết và không thấy gì lạ ở các ảnh cấu trúc là loại nguy hiểm nhất.

## 4. Giới hạn chiều rộng: không phải mọi crop trái/phải đều cứu được [đã xác minh]

Bit `(r*bw0 + c) mod 112`: một CỘT chỉ mang các bit cách nhau `g = gcd(bw0, 112)`. Sau khi cắt trái/phải, bản crop phải còn ít nhất `g` cột (g×8 px), nếu không một số bit payload mất vĩnh viễn -
tính chất của cách nhúng, mọi bộ giải mã đều dính. Với hầu hết chiều rộng g rất nhỏ (1280 px: g=16 → 128 px; 1080/3000 px: g=1). Chỉ đau ở chiều rộng là bội của 56 khối (448, 896, 1344, 1792 px ...).
Quy tắc dự đoán đúng cả 10/10 trường hợp thử (8 lần mỗi ca): 448 px còn 408 → mất; 896 còn 816 → mất, còn 896 → đọc; 1344 còn 1100 / 500 → đọc, còn 400 → mất; 1792 còn 1000 → đọc, còn 800 → mất;
1280 còn 300 → đọc, còn 100 → mất. (Chuyện "chelsea 0/6" đầu phiên là đúng trường hợp này: ảnh mẫu 451×300 rộng 56 khối. Mình ban đầu tưởng lỗi tìm kiếm.)

## 5. Tác dụng phụ: khoá sai đọc ra ID thật nhiều hơn [đã xác minh]

Khoá không phải bí mật (đã ghi từ trước: ~1/4 khoá sai vẫn đọc được). Nhưng resync thử nhiều lưới lệch, mỗi lưới là một lần "bốc thăm" nhiễu mới trên cùng tín hiệu yếu, nên tỷ lệ tăng mạnh. Ảnh pink 640×480 có dấu, 60 khoá sai ngẫu nhiên:
bộ giải mã cũ 18/60; resync chỉ lưới (0,0): 18/60; +1 độ lệch: 33/60; +3 độ lệch (mặc định): 54/60; thêm 4 hướng: 60/60. Pink 1280×960, 30 khoá: cũ 15/30, +resync 28/30.
ID đọc ra vẫn là ID thật được nhúng (mã kiểm tra bảo đảm), không phải ID bịa; nhưng khoá còn ít giá trị "cổng" hơn. Mình thử chặn bằng ngưỡng điểm độ lệch: không được - khoá sai có điểm
r = max/median trung vị 2.8 (p90 5.2), còn khoá đúng sau crop + JPEG q30 thấp nhất 2.11, hai phân bố chồng nhau (khoá sai đọc được vì nó thực sự tương quan một phần với khoá đúng). Nên không thêm ngưỡng.
Hệ quả thực tế: hai test cũ "khoá sai thì không đọc được" (video) vốn là xổ số theo khoá ngẫu nhiên; đã đổi sang dùng video chưa đóng dấu.

## 5b. Lỗi nhãn "đã lật" gắn sai [đã xác minh] — người dùng phát hiện, v27 sai, v28 đã sửa

**Triệu chứng người dùng báo:** ảnh chỉ bị cắt (không lật) mà báo cáo vẫn ghi "ảnh này đã bị lật trên-dưới". ID vẫn đúng.
**Tái hiện (v27 như đã giao):** sổ cái 8 khoá, mỗi ảnh cắt lệch (37,23): 2/8 bị báo "lật" (`v`). **Nguyên nhân:** đúng như gợi ý của người dùng, cùng gốc với §5 (khoá sai đọc ra ID thật),
nhưng cụ thể là luồng điều khiển: tìm kiếm resync duyệt hướng ngoài, khoá trong, và lấy KHOÁ ĐẦU TIÊN đọc được ở hướng `none`; các hướng lật sau đó chỉ kiểm lại ĐÚNG khoá đó. Khoá đó
thường là khoá SAI (sổ cái xếp mới nhất trước), nên so sánh `none` với `v` bằng một khoá sai chỉ là so hai con số nhiễu. Bảng chẩn đoán cho một ảnh: khoá đúng ở `none` đọc với độ tin cậy ~492k;
các khoá sai đọc ID thật với 150k-250k ở cả `none` và `v`; nhưng v27 không bao giờ thử khoá đúng ở hướng khác vì đã "khoá" vào khoá sai.
**Đo độ rộng (3 sổ cái × 8 khoá, mỗi ảnh cắt lệch rồi để nguyên / lật trái-phải / trên-dưới / cả hai; extract qua sổ cái):**
| trạng thái thật | v27 báo đúng | v28 báo đúng |
|---|---|---|
| không lật | 16/24 | **24/24** |
| lật trái-phải | 17/24 | **24/24** |
| lật trên-dưới | 10/24 | **24/24** |
| lật cả hai | 11/24 | **24/24** |
| ID đọc đúng | 96/96 | 96/96 |
Tức là v27 dán sai nhãn lật cho 42 trên 96 bản (44%) khi sổ cái có nhiều khoá; báo cáo ban đầu của người dùng (2/8 ở trường hợp không lật) là phần nhẹ của vấn đề. Kỳ vọng thấp hơn ở sổ cái 1 khoá (các test cũ chỉ dùng 1 khoá nên không bắt được).
**Sửa:** `find_spread_spectrum_id_resync` giờ thử MỌI khoá ở MỌI hướng và lấy bản đọc có độ tin cậy cao nhất (khoá đúng ở hướng đúng đọc với khối lượng tương quan gấp 2-3 lần mọi bản đọc sai trong phép đo).
Đường tìm kiếm resize (`--rescale`) cũng vậy: lần thử đầu của chính mình gắn nhãn "bị cắt" cho ảnh chỉ resize, cùng một nguyên nhân (khoá đầu tiên đọc được), đã sửa cùng cách (test `test_a_copy_that_was_only_resized_is_read_but_not_called_cropped`).
Việc này còn dẫn tới một thay đổi về đường đi: với `--resync` bật, các lượt giải mã DCT thường cho ảnh LẬT (3 hướng lật) không chạy nữa — resync đã bao trùm (đọc ≥ bản thường trên mọi trường hợp đo, ảnh không crop, lật 3 kiểu × JPEG -/q60/q40/q30: pink 640×480 144/144 mỗi bên, 3 ảnh thật (astronaut, coffee, rocket) 108/108 mỗi bên, nhãn của resync đúng 100%),
và các lượt cũ dùng cùng kiểu "khoá đầu tiên đọc được" nên có cùng bệnh nhãn. Chạy ở chế độ `--no-resync` thì đường cũ y nguyên.
**Còn lại [chưa xác minh]:** nhãn vẫn dựa vào chênh lệch độ tin cậy 2-3×; ở nhiễu rất cao (JPEG cực thấp, mảnh rất nhỏ) khoảng cách này co lại và nhãn có thể sai trở lại, ID vẫn đúng. Chưa đo ở mức đó. Hai bản đọc cùng ID khác khoá chỉ đẩy lên "độ tin cậy" chứ không có cờ "mơ hồ".

## 6. Video [đã xác minh, một clip tổng hợp]

`video.scan_invisible(resync=True)`: khung nào bộ giải mã thường không đọc được thì thử lại dạng crop (chỉ hướng không lật, tối đa 5 s mỗi khung). `VideoScanResult.regrid_hits` đếm số khung đọc được nhờ việc đó.
Clip 640×480, 24 fps, 60 khung, đóng dấu qua `wmapp.watermark` thật, rồi `ffmpeg -vf crop=...` + x264, đọc bằng `extract --ss-key`:
| | v25 | bây giờ |
|---|---|---|
| không crop (đối chứng) CRF23 | 60/60 | 60/60 |
| cắt hai bên 80 px mỗi bên CRF23 | **0/60** | **60/60** |
| dải giữa 270×480 (kiểu 9:16) CRF23 | 0/60 | 57/60 |
| crop lệch 37,23 → 573×430 CRF23 | 0/60 | 51/60 |
| crop lệch CRF28 | 0/60 | 19/60 |
Thời gian đọc 3-6 s. Chỉ một clip tổng hợp (nhiễu), chưa đo trên video thật, video dài, 1080p+.

## 7. Chưa phủ / chưa đo [chưa xác minh]
- ~~Crop + thu nhỏ trong một lượt vẫn mất~~ → làm ở v28, xem §8 (có giới hạn: chỉ `--rescale`, không lật, không dò được ở ảnh có tỷ lệ khung đã đổi).
- Xoay, méo phối cảnh; payload chữ (chỉ ID); video thật; ảnh JPEG đầu vào bất kỳ ngoài q85/60/40/30; ảnh thật lớn (các ảnh thật ở đây đều ≤ 1411 px).
- Ảnh mẫu thật chỉ 5 ảnh nhỏ (astronaut, coffee, rocket, hubble_deep_field, chelsea); chelsea bị loại khỏi bảng vì rơi đúng vào giới hạn §4.
- Với công cụ AI xoá logo / tái tạo khuếch tán: không đo gì trong phiên này; README đã nói không kỹ thuật nào chống được tái tạo khuếch tán.

## 8. Crop + thu nhỏ trong một lượt (`--rescale`) [đã xác minh, n nhỏ] — bản cuối (v29); lịch sử và các lần sửa ở §8b, §8c

Bản chụp màn hình ở mức zoom khác rồi cắt = ảnh gốc, cắt ở độ lệch chưa biết, rồi phóng/thu theo g. Khôi phục f = 1/g đưa lưới 8×8 về lại (ở độ lệch chưa biết), rồi tìm resync như trên. `rescale.py` cũ tìm f nhưng neo thống kê ở góc trên-trái (pha lưới đã biết = 0), mà crop làm mất mốc đó.
**Thống kê quét tỷ lệ** (`resync._patch_ratio`): max/median của năng lượng tương quan theo 64 độ lệch lưới trên một mảnh ảnh được resample theo f. Ảnh sạch: 2.9-5.2 ở f đúng, 1.1-1.3 ở chỗ khác. Ảnh nền dày kết cấu: chỉ 1.3-1.7 ở f đúng (xem §8b) — **không có ngưỡng nào tách được nó khỏi đỉnh nhiễu**, nên thuật toán không dùng ngưỡng.
**Cách làm cuối (`resync.find_spread_spectrum_id_resync_scaled`):**
1. quét f ∈ [0.5, 2.0), bước 0.25%, mảnh 384 px (tự co nhỏ cho ảnh dưới ~430 px); mọi khoá dùng chung mảnh đã resample;
2. mỗi khoá giữ 4 đỉnh cao nhất (cách nhau ≥ 1%), **không có ngưỡng**;
3. kiểm chứng bằng bộ giải mã (luật chấp nhận duy nhất đáng tin: ID chuẩn, đúng mã kiểm tra), đỉnh mạnh nhất của mọi khoá trước, rồi hạng 2-4: tinh chỉnh f trên mảnh 512 px (±0.6%, bước 0.04%), giải mã ở f tinh chỉnh và ±5 bước (bộ giải mã chịu ±6 bước = ±0.24% quanh f thật: 59/59; tinh chỉnh rơi trong 6 bước 97% số lần, tối đa 8), trên vùng giữa ≤ 1.5 MP của ảnh đã khôi phục;
4. khi một f đọc được: mọi khoá đọc lại đúng vùng đó và lấy bản tin cậy nhất (cùng lý do §5b); nhãn "đã cắt" xét với độ lệch cho phép tỷ lệ với cỡ ảnh (lệch tỷ lệ 0.25% làm lưới trôi ~6 px ở giữa ảnh 10 MP).
Chạy SAU tìm lưới thường và TRƯỚC tìm resize cũ trong `--rescale` (nhận 40% ngân sách; tìm resize cũ chỉ dừng khi trúng hoặc hết ngân sách).
**Kết quả (qua mã sản xuất, crop lệch ~10% mỗi cạnh + thu/phóng g + JPEG q85, khoá đúng):**
- Ảnh thật ≥ 480 px, 7 ảnh (astronaut, hubble_deep_field, retina, scipy_face, building, board, fruits) × g ∈ {0.9, 0.75, 1.25} × 4 lần: **84/84**, trung vị ~0.65 s mỗi lần (đo khi máy rảnh); sổ cái 4 khoá cho cùng kết quả.
  (Bản v28 đầu có ngưỡng: 12/14, 12/14, 13/14 trên 2 lần mỗi ảnh; `board.jpg` trượt hết — §8b.)
- Ảnh tổng hợp pink 1280×960 × 6 mức g × {không JPEG, q85}: 12/12.
- Ảnh **thật 10 MP** (OpenCV stitching/boat1, 3888×2592): crop + zoom 0.75 và 0.5 đọc được, crop + 0.33 không (§10b).
- Ảnh CHỈ resize (không crop): đọc được và KHÔNG bị gắn nhãn "cắt" (có test, cả ở 3.8 MP và 10 MP).
- Không dấu: 0 ID giả trên 30 lượt quét (7 ảnh thật + 8 pink × 2 tỷ lệ, 1 khoá ngẫu nhiên; đo trước khi thêm §8c).
**Chi phí:** vì không còn ngưỡng nên mỗi khoá luôn tốn bước quét (~0.15 s) VÀ kiểm chứng 4 đỉnh: ảnh không dấu ~1.4 s mỗi khoá ở ~1 MP (trung bình 30 lượt), sổ cái 1 / 4 / 8 / 16 khoá: ảnh không dấu 0.7 / 2.9 / 5.3 / 10.9 s, ảnh có dấu (khoá đúng xếp cuối) 0.5 / 1.4 / 2.7 / 4.5 s (đo trước khi thêm chọn vị trí mảnh §8c). Phần 40% của 20 s = 8 s nên hết ngân sách quanh 10 khoá trên ảnh không dấu (báo "chưa kết luận", rồi tìm resize cũ chạy với phần còn lại).
3000×2000: crop + 75% zoom + q85 đọc được trong 0.89 s; ảnh không dấu cùng cỡ, 1 khoá: 1.5 s.
**Giới hạn:** chỉ ảnh không lật; chỉ co dãn đều (giữ tỷ lệ khung); dưới ~0.5× dấu không còn (§10b, kiểm bằng khôi phục đúng tỷ lệ — không phải lỗi tìm kiếm); sổ cái dài (> ~10 khoá) hết ngân sách; tỷ lệ khung/gamma/màu/overlay của chụp màn hình thật chưa đo riêng.

## 8b. Vì sao `board.jpg` trượt ở crop+resize 0.75 [đã xác minh] — và lần sửa

**Câu hỏi của người dùng.** `board.jpg` (ảnh chụp bo mạch điện tử, 640×480) trượt 3/3 lần ở g=0.75 trong §8 cũ.
**Chẩn đoán (tách hai giai đoạn):** trượt hoàn toàn ở GIAI ĐOẠN QUÉT TỶ LỆ, không phải ở giải mã. Với 6 bản ngẫu nhiên: quét trả về danh sách đỉnh RỖNG ("peaks []") trong khi giải mã sau khi khôi phục ĐÚNG tỷ lệ 1/g (bỏ qua tìm kiếm) đọc được 6/6 — dấu vẫn nguyên.
**Nguyên nhân:** cổng trong v28 đầu (z ≥ 12 và r ≥ 1.6, chọn từ vài ảnh không dấu: nhiễu ≤ 1.34) loại đỉnh thật vì ảnh bo mạch dày kết cấu tuần hoàn khiến thống kê ở tỷ lệ đúng chỉ 1.3-1.7 (z 6.6-14), thấp hơn cổng. Không phải mất tín hiệu.
**Cổng nào cũng không cứu được:** đo 52 lượt quét không dấu (ảnh thật + pink × 2 tỷ lệ × 2 khoá): đỉnh cao nhất có z tới 12.5 và r tới 1.50 ở ảnh thật (retina) — trùng khoảng giá trị của đỉnh THẬT trên board (z 5.7-14, r 1.34-1.73); và ở board đỉnh thật chỉ là đỉnh số 1 trong 2 trên 6 lần. (Một số ảnh pink tổng hợp cho r tới 7.4 ở mức không dấu: nội dung tổng hợp có cấu trúc lưới riêng, nên không dùng làm mốc nhiễu.)
**Cái cứu được:** thứ hạng. Với mảnh 384 px, tỷ lệ đúng xếp **hạng 1 trong 120/120 lần quét** (board, retina, building, fruits, astronaut × 3 g × 8 lần), mảnh 256 px: hạng ≤ 3 trên board. Nên sửa là bỏ ngưỡng, giữ 4 đỉnh xếp hạng mỗi khoá và để BỘ GIẢI MÃ (ID chuẩn, 2^-40) quyết định.
**Bằng chứng sau sửa:** board 12/12 (3 g × 4 lần), toàn bộ 84/84; 0 ID giả trên 30 lượt quét không dấu. Những lần sửa chữa dọc đường (mỗi cái tôi làm sai lần đầu): (a) bỏ cắt mảnh cho vừa ảnh làm thống kê trôi theo f; (b) chặn "ảnh nhỏ hơn mảnh thì thoát" gắn với cỡ mảnh làm bỏ mọi ảnh < 384 px (64/84); (c) xác nhận lại trên TOÀN ảnh khi vùng đã đọc được làm hàm trả `None` và sập ở ảnh 10 MP; (d) độ lệch cho phép cố định 1 px làm ảnh chỉ resize bị gắn nhãn "cắt" ở 10 MP.
**Phần còn lại không cứu được:** nền tổng hợp CỰC dày (cường độ lưới 60 + vệt 55, bắt chước bo mạch nhưng nặng hơn nhiều): đọc ~80% (24/30 cặp khoá-ảnh ngẫu nhiên), phụ thuộc khoá — khoá 902 không bao giờ đọc được trên bất kỳ nền nào (tín hiệu mất vì năng lượng tuần hoàn của nền rơi đúng vào mẫu của khoá). Test dùng cặp (khoá, nền) cố định đã biết là đọc được, không khẳng định 100%.

## 8c. Ảnh chụp màn hình THẬT (X server) và chọn vị trí mảnh đo [đã xác minh, một ảnh]

Cách chụp: Xvfb 1600×1000 + ImageMagick `display` hiện ảnh đã đóng dấu (scipy `face` 1024×768) ở 100% / 75% / 50% rồi `import -window root`. Đây là chụp màn hình thật (không phải "ảnh gốc cắt rồi lưu"): ảnh nằm trong cửa sổ, quanh là nền đen, có viền cửa sổ 2 px, co bằng bộ lọc của ImageMagick.
Bảng (đọc được / thử; `+R` = có `--rescale`; png không mất dữ liệu và JPEG q90):
| | whole 1600×1000 | cắt theo cửa sổ (kèm viền 2 px) | cắt bên trong ảnh |
|---|---|---|---|
| 100% | png OK, q90 OK (0.6-0.7 s) | OK, OK (0.3 s) | OK, OK (0.3 s) |
| 75% | cần `+R`: png OK, q90 OK (1.4-1.5 s); không `+R`: no | `+R`: OK, OK (~1.2 s); không `+R`: no | `+R`: OK, OK (~1.0 s); không `+R`: no |
| 50% | **không đọc được** (kể cả `+R`, 19-20 s hết ngân sách) | `+R`: OK, OK (0.8-0.9 s) | `+R`: OK, OK (0.7 s) |
Cắt bất kỳ kiểu nào (theo cửa sổ hay bên trong ảnh) đọc được ở cả 3 mức, kể cả JPEG q90. Chỉ **toàn màn hình khi ảnh chỉ chiếm một góc ở 50%** là trượt. Với toàn màn hình ở 75%: trước đó (mảnh luôn ở tâm ảnh chụp, tức nền đen) JPEG q90 trượt; sau khi chọn vị trí mảnh đọc được cả png và q90.
**Vì sao toàn màn hình 50% trượt (đã tìm, chưa sửa xong):** (1) mảnh ở tâm màn hình là nền đen → đã sửa: `find_patch` giữ tâm ảnh khi nó có nội dung (mọi ảnh thật đo vẫn ở tâm, nên số đo cũ còn giá trị) và chỉ khi tâm rỗng mới chuyển sang cửa sổ mượt nhất không có mảng đen/trắng thuần; (2) hai bản đầu của mình chọn mảnh có NHIỀU kết cấu nhất, sai ngược: thống kê tỷ lệ với năng lượng dấu trên năng lượng nền, nên giảm theo kết cấu (tương quan −0.85 giữa r và gradient trung bình trên 20 mảnh của một ảnh chụp; r = 1.4 ở mảnh nhiều kết cấu nhất, 2.3 ở mảnh mượt) và bị thắng bởi mảnh nằm đè lên viền ảnh; (3) còn lại, **nghi ngờ chưa kiểm**: ở 50% ảnh chỉ cao 388 px nên cửa sổ hợp lệ lớn nhất có thể chỉ ~256 px, khi đó mảnh quét chỉ 128 px (256 khối/độ lệch), có lẽ quá ít để xếp hạng tỷ lệ đúng trong top 4 (mình chưa in ra cửa sổ nào được chọn). Chỗ đo thêm: quét không thấy f=2.0 trong top 4 ở bất kỳ tâm nào đã thử (r@2.0 từ 1.34 đến 2.35). Giải mã ở đúng f=2.0 (oracle) đọc được cả vùng lẫn cả ảnh.
**Hệ quả thực tế:** người dùng "chụp rồi crop" (cách họ mô tả) không bị ảnh hưởng. Ảnh chiếm một góc nhỏ của màn hình và bị thu nhỏ (50%) mà không crop thì chưa đọc được.

## 8d. Chi phí `--rescale` với sổ cái dài [đã xác minh] — người dùng yêu cầu làm trước P5

**Vấn đề (người dùng nêu):** bỏ ngưỡng (§8b) nghĩa là mỗi khoá luôn phải quét đầy đủ, nên sổ cái > ~10 khoá dễ hết ngân sách.
**Đo trước (profile theo giai đoạn, ảnh scipy_face crop + zoom 0.75 → 684×513, ngân sách tính theo 40% của 20 s = 8 s):** với bản crop có dấu, **95% thời gian là bước quét tỷ lệ** (~0.43 s mỗi khoá = 0.8 ms mỗi (tỷ lệ, khoá) × 555 tỷ lệ), kiểm chứng gần như không tốn gì vì đỉnh của khoá đúng xếp hạng 1 (0.1 s). Bản cũ quét TẤT CẢ khoá rồi mới kiểm chứng: 1 / 16 / 60 khoá, khoá đúng xếp cuối: 0.8 / 7.4 / 27.5 s → với ≥ ~20 khoá hết ngân sách NGAY TRONG LÚC QUÉT, trước khi giải mã lần nào, dù khoá đúng có trong sổ cái. (Ảnh không dấu: 1.1 / 15.6 / 59.6 s — luôn dùng hết ngân sách, giống `--rescale` cũ của v25 trên ảnh không đọc được.)
**Cách sửa (`find_spread_spectrum_id_resync_scaled`):** quét theo VÒNG 6 khoá (theo thứ tự cho sẵn: sổ cái liệt kê khoá mới nhất trước), kiểm chứng ngay các đỉnh của vòng đó (mạnh nhất trước) rồi mới quét vòng sau. Khi một tỷ lệ đọc được, mọi khoá của CẢ sổ cái được xét tại tỷ lệ/vùng đó và lấy bản tin cậy nhất như cũ, nhưng chỉ khoá nào có thống kê mảnh ≥ 1.45 (như lọc sơ bộ §9) mới đọc đầy đủ (khoá vừa đọc được thì luôn đọc) — chi phí bước chọn khoá ~ 1 ms mỗi khoá thay vì ~25 ms.
**Vì sao vòng đầu thường đã trúng ngay cả khi khoá đúng ở cuối:** khoá SAI đọc ra ID thật khá thường (§5), nên ứng viên của vòng đầu đã đọc được ở tỷ lệ đúng, rồi bước chọn khoá trên cả sổ cái tìm ra khoá đúng. Điều này KHÔNG được bảo đảm (xem ca `board` dưới đây).
**Kết quả (ảnh scipy_face như trên, ngân sách 8 s, khoá đúng ở vị trí cho sẵn, 1 ảnh mỗi dòng):**
| sổ cái | vị trí khoá đúng | đọc, đúng khoá | thời gian | khoá đã quét |
|---|---|---|---|---|
| 16 | cuối | có | 3.0 s | 6 / 16 |
| 60 | đầu / giữa / cuối | có / có / có | 2.7 / 3.1 / 3.2 s | 6 / 60 |
| 200 | cuối | có | 4.8 s | 6 / 200 |
(Trước sửa: 7.4 s ở 16 khoá và 27.5 s ở 60 khoá, tức không bao giờ kịp ngân sách 8 s từ ~20 khoá.)
**Phân tích rộng hơn** (7 ảnh thật ≥ 480 px × 3 mức zoom × 2 lần = 42 ca, sổ cái 60 khoá, khoá đúng CUỐI, ngân sách 8 s): **38/42 đọc đúng khoá**, trung vị 4.8 s, tối đa 8.2 s; 4 ca trượt đều là `board.jpg` (nền dày kết cấu, §8b), mỗi ca hết ngân sách khi mới quét 6-12/60 khoá. Ở `board` các khoá sai không đọc được ID ở tỷ lệ đó nên không có "vòng đầu may mắn" — phải quét tới vòng của khoá đúng (~0.4 s mỗi khoá → ~24 s cho 60 khoá). Giới hạn này là có thật.
**Độ nhớ không đổi:** sổ cái 1 khoá, 7 ảnh thật × 3 mức zoom × 4 lần: 84/84 (như §8); nền tổng hợp dày amp 30/40: 10/10, amp 60/55: 5/10 (như cũ).
**Báo cáo:** khi hết ngân sách, `extract` nói thêm "the crop + resize search got through N of M ledger keys, newest first" (`ExtractResult.scaled_keys_reached`, trong `to_dict()`), để người dùng biết cần `--rescale-budget-seconds` lớn hơn hay không.
**Đã thử và KHÔNG dùng:**
- *Quét hai tầng* (thô bằng mảnh nhỏ 96-192 px, bước 0.4-0.8%, rồi mịn quanh đỉnh): mỗi lượt quét chỉ 44-141 ms nhưng mất độ nhớ — trên 63 ca ảnh thật, đỉnh đúng có hạng > 12 hoặc không có trong top 40 ở 2-5% số ca (mảnh 128 px, bước 0.4%: ≤ 12 ở 98%, `board` 8/9; bước 0.8%: 97%, `board` 7/9). Không đổi độ nhớ lấy tốc độ.
- *Chia sẻ phép lọc giữa các khoá qua ma trận Gram* của 16 hệ số dấu (năng lượng tại 64 độ lệch của mọi khoá = `pn^T G pn`): chính xác đến 1e-7 so với tính trực tiếp, nhưng chi phí cố định ~30 ms mỗi mảnh dù bao nhiêu khoá (40.0 / 34.7 / 31.0 / 28.4 ms ở 1 / 8 / 60 / 200 khoá) → ~17 s cho cả lượt quét 555 tỷ lệ, chỉ thắng cách trực tiếp (~0.4 s mỗi khoá) khi sổ cái trên ~25 khoá và vẫn chậm hơn ngân sách. Cần tối ưu thêm mới đáng; chưa làm.
- Hàm `shift_scores`: gấp hai lượt liên tiếp thay vì reshape-và-tổng theo trục (0, 2): 0.48 → 0.11 ms mỗi lần (đã dùng, tương đương số học). Phần còn lại của mỗi (tỷ lệ, khoá) là `cv2.filter2D` 8×8 (~0.5-0.8 ms tuỳ lúc máy bận), không thu nhỏ được nếu không đổi bài toán.
**Còn lại [chưa xác minh/đề xuất]:** ảnh không dấu hoặc không đọc được với sổ cái dài vẫn dùng hết phần ngân sách (không thể biết "không có gì" mà không thử hết); tối ưu Gram thật sự (giảm 30 ms cố định xuống ~10 ms bằng ghép các lời gọi) và/hoặc mảnh 256 px cho sổ cái rất dài; sắp xếp khoá theo xác suất (ví dụ khoá có kích thước ảnh gần với bản sao).

## 9. Tốc độ [đã xác minh]

Ba thay đổi; mỗi cái đo riêng:
1. **Tìm (bước, pha) bằng FFT:** tổng cắt theo bước `b[s,j] = Σ_ρ g[ρ,(j−ρs) mod 112]` = gather 12,544 số của `fft2(g)` rồi 112 FFT ngắn; khớp định nghĩa brute-force đến 4e-14. 10.8 → 4.5 ms mỗi lưới. `shift_scores` gộp 64 lượt strided thành một lượt reshape-sum: 35 → 20 ms ở 3000×2000 (lệch tương đối 5e-4, do bỏ vài pixel mép).
2. **Bỏ 3 lượt giải mã DCT cho ảnh lật khi `--resync` bật** (resync bao trùm chúng, §5b). Đây là phần lớn mức giảm ở ảnh lớn.
3. **Lọc sơ bộ (pre-screen)**: mỗi cặp (khoá, hướng) trước tiên được thử trên mảnh giữa 320 px (~1 ms); chỉ cặp nào có tỷ số năng lượng ≥ 1.45 mới chạy tìm toàn ảnh (~30 ms ở 1280×960). Ảnh < 480 px không lọc. Tỷ số nhiễu của ảnh không dấu: trung vị 1.12, p99 1.43, max 1.51 (360 lần bốc: pink + ảnh thật); khoá đúng sau crop: ≥ 4.8 không JPEG, ≥ 2.5 ở q60, thấp nhất ~1.5 ở q20 trên ảnh thật.
   Cổng đặt hơi trên trung vị nhiễu vì một cặp nhiễu lọt qua chỉ tốn một lần tìm đầy đủ, còn một dấu thật bị loại thì mất ID. Độ nhớ (ảnh thật ≥ 480 px, crop 50% diện tích, 4 lần mỗi ảnh): tìm vét cạn 28/28, 26/28, 27/28, 27/28, 24/28 và có lọc 28/28, 26/28, 27/28, 27/28, 24/28 ở không JPEG/q60/q40/q30/q20 — hệt nhau. (Một cổng 1.7 thử trước đó làm mất 1-2 ca ở q ≤ 40 nên đã hạ.)
   Ảnh có kết cấu khối 8×8 lưới thẳng hàng (như ảnh khối ngẫu nhiên trong test) tự có năng lượng cao ở độ lệch (0,0) và LỌT cổng: chỉ tốn thời gian, không ID giả.

`wmapp.extract()`, nhỏ nhất của 3 lần (sổ cái 1 khoá, một CPU), v25 → v27 (đã giao) → v28 (bản cuối, SAU khi thêm lọc sơ bộ; lần đo cuối cùng của phiên):
| | 640×480 | 3000×2000 |
|---|---|---|
| ảnh có dấu, không crop | 0.08 → 0.07 → 0.07 s | 1.46 → 1.57 → 1.43 s |
| ảnh KHÔNG có dấu | 0.20 → 0.41 → **0.18** s | 4.07 → 5.32 → **2.07** s |
| crop lệch 37/23 | 0.15 (không đọc được) → 0.32 → **0.18** s (đọc được) | 4.02 (không đọc được) → 4.68 → **2.05** s (đọc được) |
(Các lần đo trước đó trong phiên cho số hơi khác, ví dụ 0.13 s / 1.85 s cho ảnh không dấu: nhiễu giữa các lần chạy khoảng ±15%; bảng này là lần cuối, sau mọi thay đổi code.)
Theo số khoá trong sổ cái (1280×960, hàm resync riêng, thời gian của bản đọc): ảnh không dấu 1 / 8 / 30 / 100 khoá: vét cạn 0.10 / 0.70 / 2.65 / 9.9 s → có lọc 0.01 / 0.03 / 0.09 / 0.29 s (3000×2000, 100 khoá: 22.8 → 0.36 s). Ảnh crop có dấu, khoá đúng xếp CUỐI: vét cạn 9.3 s → có lọc 3.0 s ở 100 khoá (1280×960), 24 → 5.5 s (3000×2000): phần còn lại là các khoá SAI cũng qua cổng vì chúng đọc ra ID thật một phần (§5) và mỗi khoá như vậy tốn ~30-55 ms.
**Chưa làm [đề xuất]:** dùng chung việc resize mảnh giữa các khoá trong tìm-tỷ-lệ (~25% của 0.4 s mỗi khoá); lọc sơ bộ theo ngưỡng thích nghi theo kích thước ảnh thay vì một cổng cố định.

## 10. Ảnh/video thật lớn hơn [đã xác minh — nhưng không có gì lớn hơn ~2 MP]

Nguồn ảnh thật dùng được offline: 5 ảnh của skimage (astronaut 512², coffee 600×400, rocket 640×427, hubble_deep_field 1000×872, retina 1411²), `scipy.datasets.face` (1024×768), 5 ảnh mẫu của OpenCV tải từ GitHub (building 868×600, board 640×480, messi5 548×342, fruits 512×480, home 512×384). **Lớn nhất 1411×1411 ≈ 2 MP**; không có ảnh thật nào ở cỡ camera 12 MP. Phần lớn nguồn khác (gitlab của skimage) bị chặn 403.
(`bench/experiments/real_corpus_probe.py`, 11 ảnh × 3 lần × 3 mức JPEG = 33 lần mỗi ô.)
- **Crop trong nhóm ảnh thật:** đối chứng không crop, giải mã thường: 33/33 ở cả 3 mức JPEG. Crop lệch bất kỳ: giải mã thường **0/33**, resync **33/33** ở không JPEG / q85 / q60. Crop trái/phải theo bội của 8: thường **0/33**, resync **32/33** ở cả ba mức.
  Để hiểu ca trượt, mình quét lại 6 crop trái/phải theo bội của 8 mỗi ảnh × 11 ảnh × 3 mức JPEG (198 lần): 11 lần trượt. 6 trong 11 rơi đúng quy tắc §4 (số khối còn lại < g = gcd(chiều rộng khối, 112): home 512 px / astronaut 512 px / board 640 px / retina 1411 px, g=16, còn 3-13 khối).
  5 trong 11 KHÔNG phải quy tắc g: coffee (g=1) còn 8, 18, 20 khối; board (g=16) còn 20 và 22 khối - tức là crop cực hẹp (8-22 khối chiều ngang, ~64-176 px) với JPEG q85/q60, một phần là giới hạn lượng dữ liệu còn lại chứ không phải quy tắc g. Mình chưa tách riêng mức "bao nhiêu khối thì đủ" cho ảnh thật.
- **Giới hạn diện tích trên ảnh thật** (crop lệch giữ một phần diện tích, 11 ảnh × 3 lần): 80% 33/33; 50% 33/33; 25% 30-32/33; 10% 24-26/33; 5% 9-13/33 (không JPEG / q85 / q60 đều tương tự). Khác hẳn bảng tổng hợp §3.2 (9/9 ở 5%): ảnh thật nhỏ nên 5% diện tích chỉ còn ~400 khối (3.5 chu kỳ), tổng hợp 1280×960 còn ~960 khối. **Kết luận §3.2 "đọc được tới 5% diện tích" chỉ đúng cho ảnh tổng hợp lớn; với ảnh thật nhỏ, giới hạn là khoảng 10%.**
- **Video thật** (vtest.avi của OpenCV, 768×576, 10 fps, cảnh người đi bộ thật; 60 khung, đóng dấu qua `wmapp.watermark`, rồi `ffmpeg crop` + x264): không crop 60/60; cắt hai bên 80 px **0/60 → 60/60**; dải 9:16 **0/60 → 60/60**; crop lệch 37,23 CRF23 **0/60 → 60/60**; CRF28 **0/60 → 60/60** (tất cả "qua căn lại": 60/60). Đóng dấu 0.15 s/khung (2 tiến trình); đọc 2.3-3.5 s cho 60 khung.
  Tốt hơn hẳn clip tổng hợp ở §6 (51/60 và 19/60 ở crop lệch): nội dung thật (mịn, ít nhiễu) giữ dấu tốt hơn khung nhiễu trắng sau x264, hoặc chỉ là clip này dễ. Không kết luận được tổng quát từ một clip.
- **Video 1080p** (cùng cảnh phóng lên 1920×1080 bằng lanczos — nội dung thật nhưng mờ hơn 1080p quay thật; 24 khung): không crop 24/24; cắt hai bên 200 px, 9:16, crop lệch CRF23, crop lệch CRF28: **mỗi cái 24/24 (0/24 trước)**. Đóng dấu 0.86 s/khung (2 tiến trình); đọc 2.4-6.6 s cho 24 khung (~0.1-0.27 s/khung).
**Chưa đo [chưa xác minh]:** ảnh thật ≥ 4 MP (không có); video quay thật ở 1080p/4K, video dài (> 60 khung), codec khác (HEVC/VP9/AV1), re-encode hai lần; ảnh chụp màn hình thật (không phải ảnh gốc cắt rồi lưu); công cụ AI xoá logo / tái tạo khuếch tán.

## 10b. Ảnh thật 10 MP (OpenCV stitching boat1/boat3, 3888×2592) [đã xác minh, 2 ảnh, ảnh nguồn là JPEG đã nén]

Nguồn: `https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat1.jpg` … `boat6.jpg` (6 ảnh chụp thật bến cảng, ~0.7 MB mỗi ảnh, JPEG nén mạnh nên nội dung đã ít chi tiết cao tần — dễ hơn ảnh RAW/PNG). Đóng dấu qua `wmapp.watermark` (spread + ID, sổ cái), tấn công, đọc lại bằng `wmapp.extract`. `bench/experiments/large_real_probe.py`.
Đóng dấu: 9-30 s mỗi ảnh (tuỳ lúc máy bận), `extract` ảnh nguyên 3.1-4.7 s (bộ giải mã thường), crop 3.0-5.6 s.
Bản cuối (boat1; boat3 từ lần chạy trước khi sửa nhãn, cùng kết quả đọc được): 
| tấn công | đọc ID | nhãn "cắt" | thời gian |
|---|---|---|---|
| đối chứng | có | không | 3.1 s |
| crop lệch 1.6%/cạnh (+q85, +q60) | có, có, có | có | 3.0-3.5 s |
| cắt hai bên 400 px mỗi bên + q85 | có | có | 2.6 s |
| crop lệch giữ 25% / 10% / 5% diện tích + q85 (1944×1296, 1229×819, 869×579) | có, có, có | có | 1.0 / 0.4 / 0.3 s |
| chỉ resize ×0.75 + q85 (`--rescale`) | có (f=1.333) | không | 3.1 s |
| crop lệch + resize ×0.75 + q85 | có (f=1.334) | có | 3.2 s |
| chỉ resize ×0.5 + q85 | có (f=2.002) | **không** (ở lần chạy trước khi sửa độ lệch cho phép: bị gắn "cắt" sai) | 2.3 s |
| crop lệch + resize ×0.5 + q85 | có (f=1.999) | có | 2.3 s |
| resize ×0.33 + q85, có hoặc không crop | **không** (cả boat1 và boat3) | — | 20-40 s hết ngân sách |
**Dưới ~0.5× là giới hạn của tín hiệu, không phải của tìm kiếm** (boat5 đóng dấu, chỉ resize, khôi phục ĐÚNG tỷ lệ 1/g không tìm kiếm rồi giải mã): ×0.5 không nén: đọc được; ×0.5 + q85: **không** (nhưng `extract --rescale` vẫn đọc được vì nó giải mã theo vùng và thử vài mức f); ×0.4, ×0.33, ×0.25 (không nén và q85): không đọc được. Thu nhỏ mạnh làm mất phần lớn năng lượng hệ số trung tần nơi dấu nằm, trước cả khi có JPEG.
Chưa đo: ảnh thật lớn KHÔNG nén sẵn (RAW/PNG), ảnh điện thoại 12 MP, 4K+ video.
