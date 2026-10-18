# `extract` gộp mọi lớp, báo từng lớp (bước 4/5 của kế hoạch v25 - gọi là "P2" trong handoff)

Lưu ý tên: `P2-NOTES.md` cùng thư mục là việc KHÁC (bộ giải mã spread-spectrum biết ID, v18-v21). "P2" trong
kế hoạch 5 bước của HANDOFF v25 là việc này.

Nhãn: **[đã xác minh]** = đã chạy/đo thật. **[đề xuất]** = ý tưởng, chưa làm. **[chưa xác minh]** = còn phải kiểm tra.

## 1. Hành vi mới [đã xác minh]

`wmapp.extract()` ở chế độ `auto` trước đây dừng ở lớp đầu tiên cho ra ID: LSB có ID thì spread / metadata /
C2PA không bao giờ được thử; spread có ID thì metadata / C2PA không được đọc. Giờ mọi lớp đều được kiểm tra
RIÊNG, và `ExtractResult.layers` có đúng 4 mục theo thứ tự cố định `lsb, spread, metadata, c2pa`, mỗi mục là
một `LayerResult(layer, status, text, wm_id, note)`:

| status | nghĩa |
|---|---|
| `found` | đọc ra một watermark ID hợp lệ (`wm_id` là dạng chuẩn) |
| `text` | đọc ra một payload KHÔNG phải ID: payload chữ kiểu cũ, hoặc rác (ảnh không có dấu / sai key đôi khi cho ra chuỗi ngắn - tool không phân biệt được hai trường hợp) |
| `none` | đã kiểm tra, không đọc được gì |
| `inconclusive` | tìm kiếm resize hết ngân sách thời gian trước khi quét hết - KHÔNG phải "chắc chắn không có" |
| `skipped` | không kiểm tra; `note` nói lý do (`--mode`, không có key để thử, chưa cài C2PA, video không có lớp này) |

Không có con số tổng hợp nào. `note` ghi thêm cách đọc được (lật lại, resize x1.33, bao nhiêu key sổ cái đã thử,
bao nhiêu khung video).

**Kết quả "chính" (`found`, `text`, `scheme`, `wm_id`, `record`, `exact_match`, và `report_lines()`) chọn y như
cũ**: LSB, nếu không có ID thì spread, nếu không thì metadata, nếu không thì C2PA. Không đổi chữ nào trong báo cáo
mặc định - nhiều test kiểm tra từng chữ (`== ["Hidden payload: secret"]`, `hit.stdout.strip() == ...`) và
`test_frontends_parity.py` đòi CLI = API Python = HTTP.

**Hai quyết định thiết kế (đề xuất của phiên này; người dùng đã chốt cả hai, giữ nguyên như dưới):**
1. Bản in từng lớp là tuỳ chọn mới `--layers` (Python `layers=True`, HTTP `{"layers": true}`), một dòng khai báo
   trong `EXTRACT_OPTIONS` nên 3 cửa vào giống nhau. Dữ liệu từng lớp thì LUÔN có trong kết quả Python/JSON
   (`layers`, `layers_disagree`). Lý do không bật mặc định: sẽ đổi đầu ra mặc định của CLI mà các test và có thể
   cả script của người dùng đang dựa vào. Đổi sang mặc định là sửa 1 dòng (`default` của option `layers` thành
   `True`) + sửa vài test kiểm tra chữ.
2. Khi hai lớp đọc ra HAI ID KHÁC NHAU, báo cáo mặc định có thêm 1 dòng "Warning: the layers disagree ..." (không
   cần `--layers`) - nếu không thì việc chọn một lớp làm kết quả chính sẽ lặng lẽ giấu một mâu thuẫn. Kết quả
   chính vẫn là lớp đầu tiên theo thứ tự trên, không nhất thiết là dấu gốc. Dòng này chỉ xuất hiện khi có mâu thuẫn
   thật nên không đổi đầu ra của bất kỳ trường hợp thường nào.

## 2. Chi phí [đã xác minh, n nhỏ]

Một ảnh tổng hợp mỗi cỡ, `wmapp.extract()` qua pipeline thật, lấy giá trị nhỏ nhất của 3 lần chạy, so v25 (cũ) với
bản này. Sổ cái có đúng 1 key spread.

| Trường hợp | 640×480 cũ → mới | 3000×2000 cũ → mới |
|---|---|---|
| đóng dấu LSB-ID, `extract` auto + sổ cái | 0.03 s → 0.19 s | 0.36 s → 3.78 s |
| đóng dấu spread-ID, `extract` auto + sổ cái | 0.07 s → 0.07 s | 1.43 s → 1.49 s |
| LSB-ID, `--rescale`, key sai | 0.03 s → 0.18 s | 0.35 s → 3.25 s |
| video 640×480, 60 khung (auto + sổ cái) | 3.42 s → 3.10 s (nhiễu, không có khác biệt thực) | |

Chỉ trường hợp "LSB đọc trước" chậm lên: trước đó spread/metadata/C2PA bị bỏ qua, giờ thì spread phải thử thật,
mà một lần thử "không thấy" gồm 4 hướng (không lật + 3 hướng lật), mỗi hướng một lần DCT cả ảnh (đo riêng: 1 hướng 1.39 s, 4 hướng 3.99 s ở 3000×2000; 0.08 s / 0.18 s ở 640×480).
Đó đúng bằng chi phí hiện tại của một ảnh KHÔNG có dấu nào. Trường hợp thường gặp nhất (ảnh repost, dấu spread còn)
không đổi, vì metadata / C2PA đọc rất rẻ (không đo được khác biệt).

## 3. Đường tắt duy nhất và vì sao chỉ là bước resize [đã xác minh]

Với `--rescale`, bước tìm resize của lớp spread KHÔNG chạy khi lớp LSB đã đọc ra ID. Lý do: resize phá dấu LSB, nên
tìm resize trên bản như vậy chỉ đốt hết ngân sách (`--rescale-budget-seconds`, mặc định 20 s) để ra "không thấy".
Đo riêng ở 640×480 (ảnh đóng dấu LSB, key spread sai, `--rescale`): có đường tắt 0.78 s; bước tìm resize một mình, không có đường tắt, 20.41 s - tức đụng đúng ngân sách 20 s mặc định mà không tìm ra gì.

Bản nháp đầu của phiên này còn bỏ cả bước tìm LẬT, dựa trên giả định "LSB đọc được ID thì ảnh còn nguyên". Giả định
đó **sai**; phép thử (PNG không mất dữ liệu, 640×480, 1 ảnh, 1 ID, lớp LSB):

| Biến đổi sau khi đóng dấu LSB-ID | Lớp LSB |
|---|---|
| không đổi | found (đúng ID) |
| lật trái-phải | text (chuỗi rác, không phải ID) |
| **lật trên-dưới** | **found (đúng ID)** |
| xoay 180° | text (rác) |
| resize 75% | none |
| JPEG q95 | none |

Lật trên-dưới KHÔNG phá LSB, nên một bản vừa lật vừa còn dấu spread sẽ bị bỏ sót nếu bỏ bước lật. Có test chặn đúng
trường hợp này (`test_but_the_flip_search_still_runs_...`): ảnh có dấu spread ID X, lật trên-dưới, rồi đóng LSB-ID Y
lên; `extract` đọc LSB ra Y, lớp spread vẫn tìm ra X "sau khi lật lại", và cảnh báo bất đồng hiện ra. Ghi chú này
được giữ lại để lần sau không "tối ưu" lại sai chỗ đó.

## 4. Test [đã xác minh]

`tests/test_extract_layers.py`: 14 test (cấu trúc 4 lớp, bản repost bị lột metadata + JPEG q60, payload chữ,
LSB rác không giấu ID spread, bản in `--layers` chỉ thêm dòng và không đổi các dòng cũ, public dict không lộ đường dẫn
sổ cái / key, lớp "không kiểm tra" nói lý do, thiếu C2PA = `skipped` chứ không phải trượt, hết thời gian = `inconclusive`,
đường tắt resize, lật vẫn chạy, bất đồng được báo, đồng thuận thì không báo, video). Thêm 1 test vào
`test_frontends_parity.py`: CLI `--layers` == `wmapp.extract(layers=True).report_lines()` == `message` của HTTP, và
dữ liệu `layers` trong JSON có cả khi không xin bản in. Kiểm tra đột biến: tắt đường tắt, tắt cảnh báo bất đồng,
và cho lớp metadata luôn trả `none` đều làm các test tương ứng đỏ.

## 5. Chưa đo / còn mở [chưa xác minh]

- Đường C2PA "tìm thấy" qua lớp C2PA riêng chưa chạy được ở sandbox này: 5 test C2PA vẫn đỏ như v25 (cùng 5 test,
  `c2pa-python` là 0.38.0 ở phiên này, 0.37.12 ở phiên trước - vẫn là lỗi môi trường, không phải code). Lớp C2PA chỉ
  được kiểm tra ở trạng thái `none` / `skipped`.
- Sổ cái có rất nhiều key (chỉ thử 1 key); ảnh lớn hơn 6 MP; ảnh JPEG đầu vào; các kiểu resize khác (phóng to
  nearest-neighbour...) có thể để lại LSB đọc được hay không - đường tắt resize giả định là không, mới đo resize 75%.
- `text` không phân biệt được payload chữ thật với rác; đó là giới hạn sẵn có của bộ giải mã LSB, không phải của thay đổi này.
- Trang web (`web/index.html`) không có giao diện `extract` nên không có gì để hiển thị lớp; HTTP `/api/extract` đã trả
  `layers` và nhận `{"layers": true}`.
