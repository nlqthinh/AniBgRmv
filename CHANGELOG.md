# CHANGELOG

## M13.1 - Sửa lỗi exe đọc nhầm `app_config.json` cũ/default

### Vấn đề

Người dùng đã sửa `app_config.json` sang tên model mới, ví dụ:

```json
"model": {
  "unet_onnx_path": "models/f3_unet_mit_b1_768.onnx",
  "vitmatte_onnx_path": "models/m9_vitmatte_best_768.onnx"
}
```

nhưng exe vẫn báo thiếu model cũ:

```text
models/f2a1_unet_mit_b1_768.onnx
```

Nguyên nhân thường gặp:

- `app_config.json` được PyInstaller bundle vào `_internal`, nhưng code lại tìm config cạnh exe.
- Nếu cạnh exe chưa có config, app tự tạo default config cũ.
- Hoặc folder `dist/AniBGRmv/` đang giữ lại `app_config.json` cũ từ lần build trước.

### Sửa trong `app.py`

Thêm logic config rõ ràng:

```python
get_external_config_path()
get_bundled_config_path()
ensure_external_config_exists()
```

Quy tắc mới:

```text
1. Ưu tiên app_config.json cạnh exe/app.py.
2. Nếu chưa có, copy app_config.json được bundle ra cạnh exe/app.py.
3. Nếu không có cả hai, tạo default config.
```

Như vậy sau khi build exe, file người dùng cần sửa luôn là:

```text
dist/AniBGRmv/app_config.json
```

### Sửa thông báo lỗi

Nếu load model fail, popup sẽ hiển thị thêm:

```text
Config đang đọc
UNet ONNX path
ViTMatte ONNX path
```

để biết app thật sự đang đọc file config nào và đường dẫn model nào.

### Có train không?

Không.

### File thay đổi

```text
app.py
CHANGELOG.md
```

---

## M13 - Tách đường dẫn và cấu hình runtime ra `app_config.json`

### Tổng quan

Bản này chỉ gửi các file thay đổi:

```text
app.py
app_config.json
AniBGRmv.spec
CHANGELOG.md
```

Không train model. Không đổi `bg_engine.py`. Không đổi ONNX pipeline.

Pipeline AI vẫn là:

```text
image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> clamp known trimap regions -> final alpha -> cutout PNG
```

---

## app.py

### 1. Bỏ hard-code các biến cấu hình chính khỏi `app.py`

Các biến sau không còn cần sửa trực tiếp trong `app.py`:

```python
F2_A1_ONNX_PATH
M6_ONNX_PATH
MAX_BATCH_IMAGES
SINGLE_OUTPUT_DIR
BATCH_OUTPUT_DIR
SAVE_DEBUG_OUTPUTS
```

Thay vào đó, app đọc từ:

```text
app_config.json
```

### 2. Thêm loader config

Thêm các hàm:

```python
get_app_base_dir()
get_config_path()
write_default_config()
load_app_config()
resolve_app_path()
deep_merge_config()
```

### 3. Quy tắc resolve path

Đường dẫn tuyệt đối:

```text
D:/AniBGRmv/output/single_saved
```

được dùng nguyên.

Đường dẫn tương đối:

```text
output/single_saved
models/model.onnx
```

được hiểu là nằm cạnh:

```text
app.py
```

khi chạy source code, hoặc cạnh:

```text
AniBGRmv.exe
```

khi chạy bản exe.

### 4. Tự tạo config nếu thiếu

Nếu `app_config.json` chưa tồn tại, app sẽ tự tạo file mặc định cạnh `app.py` hoặc cạnh `AniBGRmv.exe`.

---

## app_config.json

Thêm file cấu hình mới:

```json
{
  "model": {
    "unet_onnx_path": "models/f2a1_unet_mit_b1_768.onnx",
    "vitmatte_onnx_path": "models/m6_vitmatte_m6_768.onnx"
  },
  "output": {
    "single_output_dir": "output/single_saved",
    "batch_output_dir": "output/batch_saved",
    "save_debug_outputs": true
  },
  "batch": {
    "max_batch_images": 30
  }
}
```

---

## AniBGRmv.spec

### Đổi logic include model

Trước đây spec phải ghi tên từng file model cụ thể.

Sau bản này, spec tự include tất cả file trong folder `models/` có dạng:

```text
*.onnx
*.onnx.data
```

Lý do: khi đổi tên model ONNX mới, nếu model vẫn nằm trong folder `models/`, bạn chỉ cần sửa `app_config.json`, không cần sửa spec.

### Include config

Spec cũng include:

```text
app_config.json
```

nếu file tồn tại.

---

## Có train không?

Không.

---

## File thay đổi

```text
app.py
app_config.json
AniBGRmv.spec
CHANGELOG.md
```

---


---

## M12.1 - Folder confirm dialog

### Tổng quan

Bản này chỉ sửa đúng phần xác nhận cắt folder trong batch mode.

File thay đổi:

```text
app.py
CHANGELOG.md
```

Không train model. Không đổi `bg_engine.py`. Không đổi ONNX/checkpoint. Không đổi logic cắt ảnh.

---

## app.py

### Sửa quy trình `Cắt cả folder`

Trước:

```text
Người dùng bấm Cắt cả folder
-> chọn folder
-> app hiện thêm button Xác nhận cắt folder trên giao diện chính
-> bấm button đó mới cắt
```

Sau:

```text
Người dùng bấm Cắt cả folder
-> chọn folder
-> hiện cửa sổ xác nhận
-> bấm Xác nhận thì cắt
-> bấm Hủy thì không cắt
```

Cửa sổ xác nhận gồm text:

```text
Xác nhận cắt folder "tên folder"?
```

và 2 nút:

```text
Xác nhận
Hủy
```

### Xóa logic button xác nhận cũ

Đã bỏ:

```text
self.btn_confirm_folder
confirm_folder_cut()
pending_folder_paths
pending_folder_display
clear_pending_folder_selection()
```

Lý do: xác nhận folder giờ được xử lý ngay bằng popup `QMessageBox`, không cần giữ trạng thái folder chờ xác nhận trên giao diện chính.

---

## Có train không?

Không.

---

## Ảnh hưởng khi chạy

- Người dùng chọn folder xong sẽ thấy popup xác nhận ngay.
- Nếu bấm `Hủy`, app không cắt và không giữ folder pending.
- Nếu bấm `Xác nhận`, app bắt đầu cắt folder.
- Giới hạn tối đa 30 ảnh/lần vẫn giữ nguyên.
- Output mặc định cạnh file exe vẫn giữ nguyên.

---

## M12 - Stop button, folder confirmation, batch limit, default output beside exe

### Tổng quan

Bản này chỉ sửa những file cần thiết:

```text
app.py
CHANGELOG.md
```

Không train model. Không đổi `bg_engine.py`. Không đổi checkpoint. Không đổi ONNX pipeline.

Pipeline AI vẫn là:

```text
image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> clamp known trimap regions -> final alpha -> cutout PNG
```

---

## app.py

### 1. Thêm nút `Dừng cắt`

Thêm nút:

```text
Dừng cắt
```

Vị trí:

```text
bên dưới thanh tiến trình
```

Style:

```text
nền đen
chữ đỏ
```

Áp dụng cho cả:

```text
Cắt nền 1 hình
Cắt nhiều hình
```

Lưu ý kỹ thuật:

- Single cut dùng ONNX inference một lần lớn, nên không thể ngắt thật giữa chừng ngay lập tức.
- Khi bấm `Dừng cắt`, app đánh dấu hủy và bỏ kết quả khi worker trả về.
- Batch cut có thể dừng giữa các ảnh. Nếu đang xử lý một ảnh, app sẽ dừng sau khi ảnh hiện tại trả về.

### 2. Cắt cả folder phải xác nhận trước khi chạy

Trước:

```text
Bấm Cắt cả folder -> chọn folder -> chạy ngay
```

Sau:

```text
Bấm Cắt cả folder -> chọn folder -> hiện nút Xác nhận cắt folder -> bấm xác nhận mới chạy
```

Lý do: tránh trường hợp người dùng chọn nhầm folder lớn rồi app chạy ngay.

### 3. Giới hạn batch tối đa 30 ảnh

Thêm hằng số:

```python
MAX_BATCH_IMAGES = 30
```

Áp dụng cho:

```text
chọn nhiều ảnh
kéo-thả nhiều ảnh
kéo-thả folder
cắt cả folder
```

Nếu vượt quá 30 ảnh, app sẽ báo lỗi và không bắt đầu cắt.

### 4. Default output nằm cạnh file exe

Trước:

```text
outputs/single_saved
outputs/batch_saved
```

tính theo current working directory.

Sau:

```text
output/single_saved
output/batch_saved
```

nằm cạnh:

```text
AniBGRmv.exe
```

Ở source mode, folder `output` nằm cạnh `app.py`.

Code mới:

```python
def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent
```

Lý do: khi người dùng chạy exe, output luôn nằm cạnh exe, dễ tìm hơn và không phụ thuộc user mở app từ thư mục nào.

### 5. Đổi màu chữ button sang đen

Các button sau dùng màu chữ đen:

```text
Lưu ảnh
Cắt cả folder
Đường dẫn lưu
Mở folder đã lưu
Edit
Cắt nền 1 hình
Cắt nhiều hình
Trang chủ
```

Nút `Lưu ảnh` vẫn có màu nền nổi bật, nhưng chữ là màu đen.

### 6. Tách câu hướng dẫn khỏi hover area

Trước:

```text
Upload ảnh
Kéo-thả ảnh vào đây...
```

nằm chung trong `UploadPreviewCanvas`, nên câu hướng dẫn cũng bị hover theo khung.

Sau:

```text
UploadPreviewCanvas chỉ chứa Upload ảnh / loading / preview.
Câu hướng dẫn là QLabel riêng ở ngoài canvas.
```

Các câu sau giờ chỉ là text bình thường, không còn là phần hover:

```text
Kéo-thả ảnh vào đây hoặc bấm vào khung để chọn ảnh.
Kéo-thả nhiều ảnh/folder vào đây hoặc bấm vào khung để chọn nhiều ảnh.
```

---

## Có train không?

Không.

---

## File thay đổi

```text
app.py
CHANGELOG.md
```

---

## Ảnh hưởng khi chạy

- Single và batch đều có nút `Dừng cắt` khi đang xử lý.
- Chọn folder batch không chạy ngay; phải bấm xác nhận.
- Batch tối đa 30 ảnh/lần.
- Nếu chưa config đường dẫn lưu, output mặc định nằm ở folder `output` cạnh exe.
- Text hướng dẫn không còn bị hover chung với `Upload ảnh`.

---


---

## M11 - Unified upload/preview UI + save path config

### Tổng quan

Bản này giữ pipeline AI hiện tại:

```text
image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> clamp known trimap regions -> final alpha -> cutout PNG
```

Không train model. Không đổi checkpoint. Không đổi logic ONNX. Bản này chủ yếu sửa UI và luồng lưu file.

### File sửa chính

```text
app.py
bg_engine.py
AniBGRmv.spec
requirements.txt
requirements_export.txt
tools/*.py
CHANGELOG.md
```

Trong đó `bg_engine.py` đang dùng bản M10.1 có clamp trimap known regions. `app.py` là phần sửa nhiều nhất.

---

## app.py

### 1. Gom upload và preview vào một layout duy nhất

Trước đó ở mỗi mode có 2 vùng riêng:

```text
upload area
preview area
```

Bản này thay bằng component dùng chung:

```python
UploadPreviewCanvas
```

Component này có 3 trạng thái:

```text
upload  -> hiện chữ Upload ảnh
loading -> hiện thanh tiến trình
preview -> hiện ảnh preview + nút X
```

Lý do sửa: giao diện gọn hơn, đúng flow người dùng hơn. Ban đầu người dùng chỉ thấy vùng upload. Sau khi cắt xong, chính vùng đó chuyển thành preview.

### 2. Thêm nút X ở góc phải phía trên preview

Khi đã có preview, component hiện nút:

```text
X
```

Bấm X sẽ:

- Xóa preview hiện tại.
- Xóa prediction đang giữ trong RAM.
- Disable nút lưu.
- Quay lại trạng thái `Upload ảnh`.

Single mode: xóa 1 prediction hiện tại.

Batch mode: xóa toàn bộ batch prediction hiện tại, rồi cho upload lại.

### 3. Sửa text upload và style chữ Upload ảnh

Upload text được đổi thành dạng văn bản trực tiếp, không còn layout cũ gây cảm giác dư dòng hover.

Chữ `Upload ảnh` được style rõ hơn:

```text
nền đen
chữ trắng
viền đen
bo góc
```

Lý do sửa: chữ trắng không còn bị lẫn vào nền trắng.

### 4. Thêm tiến trình khi đang cắt

Single mode:

- Sau khi upload/kéo-thả ảnh, canvas chuyển sang loading.
- Thanh tiến trình chạy dạng indeterminate.
- Text: `Đang cắt ảnh bằng F2_A1 → ViTMatte M6...`

Batch mode:

- Sau khi upload/kéo-thả/chọn folder, canvas chuyển sang loading.
- Thanh tiến trình có số lượng thật: `current / total`.
- Text cập nhật theo ảnh đang xử lý.

Lý do sửa: người dùng nhìn thấy app đang làm việc, không tưởng app bị đứng.

### 5. Thêm cấu hình đường dẫn lưu

Single mode thêm button:

```text
Đường dẫn lưu
```

Batch mode thêm button:

```text
Đường dẫn lưu
```

Bấm vào sẽ mở `QFileDialog.getExistingDirectory()` để chọn folder output.

Sau đó:

- Single save sẽ lưu vào folder người dùng chọn.
- Batch save sẽ lưu vào folder người dùng chọn.

Default vẫn là:

```text
outputs/single_saved
outputs/batch_saved
```

### 6. Button được làm to hơn và căn giữa

Các button workflow được đưa vào helper:

```python
center_button_row()
apply_button_style()
```

Single mode có:

```text
Lưu ảnh
Edit
Đường dẫn lưu
Mở folder đã lưu
```

Batch mode có:

```text
Lưu ảnh
Cắt cả folder
Đường dẫn lưu
Mở folder đã lưu
```

Nút `Lưu ảnh` dùng style màu xanh để nổi bật hơn.

### 7. Thêm nút Edit cho single mode

Single mode thêm button:

```text
Edit
```

Hiện tại button này chỉ hiện message:

```text
Tính năng Edit/SAM sẽ được thêm ở bước sau.
```

Lý do: giữ vị trí UI sẵn cho bước SAM/manual correction sau này, nhưng chưa thêm logic chỉnh sửa.

### 8. Batch chỉ preview ảnh đầu tiên

Sau khi batch cắt xong nhiều ảnh, app chỉ preview ảnh đầu tiên:

```text
Preview ảnh đầu tiên trong batch
```

Toàn bộ prediction vẫn được giữ trong RAM để khi bấm `Lưu ảnh`, app lưu toàn bộ batch.

### 9. Bỏ QListWidget batch khỏi UI chính

Bản trước có list trạng thái từng ảnh. Bản này bỏ list đó để đúng yêu cầu gom UI thành một layout chính.

Thông tin batch hiện còn trong:

```text
status_label
progress bar
batch_log.csv sau khi lưu
```

---

## bg_engine.py

Giữ bản M10.1.

Logic quan trọng vẫn là:

```text
M6 raw alpha -> clamp theo trimap -> final alpha
```

Cụ thể:

```python
alpha[trimap == 0] = 0.0
alpha[trimap == 255] = 1.0
alpha[trimap == 128] = giữ output M6
```

Lý do giữ: debug trước đó cho thấy F2 mask đúng, trimap đúng, nhưng M6 raw làm background xám. Clamp giúp tránh lỗi cả ảnh bị mờ.

---

## Output folder

Nếu dùng default:

```text
outputs/single_saved/
outputs/batch_saved/
```

Nếu user chọn `Đường dẫn lưu`, output sẽ nằm trong folder đó.

Trong folder output hiện vẫn lưu:

```text
01_cutouts/
02_preview_checker/
03_preview_white/
99_debug/
```

Trong app chính thức sau này có thể đổi trong `app.py`:

```python
SAVE_DEBUG_OUTPUTS = False
```

Khi đó chỉ nên giữ output cutout chính.

---

## AniBGRmv.spec

Giữ logic include ONNX và `.onnx.data` nếu tồn tại.

Nếu bạn đổi tên model ONNX mới, cần sửa tiếp trong:

```text
app.py
AniBGRmv.spec
```

Cụ thể:

```python
F2_A1_ONNX_PATH = "models/...onnx"
M6_ONNX_PATH = "models/...onnx"
```

và danh sách `model_files` trong `AniBGRmv.spec`.

---

## Có train không?

Không.

---

## Ảnh hưởng khi chạy

- Mở app vẫn có trang chủ.
- Vào single/batch sẽ thấy một khung lớn duy nhất.
- Ban đầu khung hiện `Upload ảnh`.
- Upload/kéo-thả xong khung chuyển sang loading.
- Cắt xong khung chuyển thành preview.
- Bấm X sẽ xóa preview và quay lại upload.
- Bấm `Đường dẫn lưu` để chọn folder output.
- Bấm `Lưu ảnh` mới ghi file ra ổ đĩa.

---

## Dung lượng disk/Drive

Nặng nhất vẫn là:

```text
models/*.onnx
models/*.onnx.data nếu có
dist/
build/
outputs/ nếu lưu nhiều debug
```

Có thể xóa sau khi build/test:

```text
build/
dist/ cũ
outputs/verify/debug cũ
```

Không nên xóa:

```text
models/*.onnx
models/*.onnx.data nếu model đang phụ thuộc external data
```
