# CHANGELOG

# CHANGELOG - M10 UI + Save Patch

## Tổng quan

Bản này sửa app từ UI phẳng một màn hình sang UI có trang chủ và 2 workflow rõ ràng:

1. Cắt nền 1 hình
2. Cắt nhiều hình

Pipeline AI không đổi:

`image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> final alpha -> cutout PNG`

Không train model.

## app.py

### Thêm

- `HomePage`
  - Trang chọn tác vụ.
  - Có nút `Cắt nền 1 hình`, `Cắt nhiều hình`.
  - Có nút disabled `Cắt video (sau)`, `Xóa vật thể (sau)`.

- `SingleCutPage`
  - Workflow riêng cho 1 ảnh.
  - Upload area nằm giữa màn hình.
  - Hỗ trợ click để chọn ảnh.
  - Hỗ trợ kéo-thả.
  - Nếu kéo-thả nhiều ảnh thì chỉ lấy ảnh đầu tiên.
  - Chọn/thả ảnh xong tự động cắt.
  - Không tự động lưu.
  - Có nút `Lưu ảnh đã cắt`.

- `BatchCutPage`
  - Workflow riêng cho nhiều ảnh.
  - Upload area nằm giữa màn hình.
  - Hỗ trợ click để chọn nhiều ảnh.
  - Hỗ trợ kéo-thả nhiều ảnh/folder.
  - Có nút `Chọn folder và cắt cả folder`.
  - Chọn/thả xong tự động cắt.
  - Không tự động lưu.
  - Có nút `Lưu tất cả ảnh đã cắt`.

- `DropUploadArea`
  - Component dùng lại cho cả single và batch.
  - Tách logic kéo-thả khỏi page.

- Worker mới:
  - `SingleCutWorker`
  - `BatchCutWorker`
  - `BatchSaveWorker`

- Helper:
  - `collect_image_files_from_paths()`
  - `pil_to_pixmap()`

### Đổi

- Không còn nút `Tự động cắt nền` kiểu cũ.
- Sau upload, app tự động cắt.
- Save là hành động riêng.
- Output single và batch tách folder:
  - `outputs/single_saved/`
  - `outputs/batch_saved/`

### Xóa/loại bỏ logic cũ

- Bỏ workflow một màn hình gồm `Chọn ảnh`, `Tự động cắt nền`, `Batch nhiều ảnh`.
- Bỏ việc auto save ngay trong quá trình cắt từ UI.

## bg_engine.py

### Thêm

- `process_for_preview(image_path)`
  - Chạy model.
  - Tạo alpha/cutout/preview.
  - Giữ trong RAM.
  - Chưa ghi file.

- `save_prediction(prediction, output_dir, save_debug=True)`
  - Chỉ ghi file khi user bấm Save.
  - Lưu cutout, preview, debug theo folder rõ.

- `_unique_path()`
  - Tránh ghi đè khi batch có nhiều ảnh trùng tên.

### Đổi

- Engine không còn coi `process_one()` là API chính.
- Tách rõ:
  - AI inference
  - preview
  - save file

### Folder output hiện tại

```text
01_cutouts/
02_preview_checker/
03_preview_white/
99_debug/
  01_final_alpha/
  02_f2_auto_alpha_768/
  03_loose_trimap_768/
  04_m6_alpha_768/
```

## requirements.txt

Không đổi lớn. Runtime vẫn dùng:

```text
pyside6
pillow
opencv-python
numpy
pandas
onnxruntime
pyinstaller
```

## AniBGRmv.spec

### Đổi

- Tự include `.onnx.data` nếu file tồn tại.
- Exclude thêm `torchaudio`.

### Vì sao

Một số lần export ONNX tạo file ngoài:

```text
*.onnx.data
```

Nếu build exe mà thiếu file này thì app có thể lỗi khi load ONNX.

## tools/verify_onnx_pipeline.py

### Sửa

- Thêm root project vào `sys.path`.

### Vì sao

File verify nằm trong `tools/`, còn `bg_engine.py` nằm ngoài folder `tools`.

## Dung lượng Drive/disk

Nặng nhất vẫn là:

```text
models/f2a1_unet_mit_b1_768.onnx
models/m6_vitmatte_m6_768.onnx
models/*.onnx.data nếu có
dist/
build/
```

Có thể xóa sau khi build/test:

```text
build/
```

Có thể xóa `dist/` cũ nếu đã build bản mới và không cần giữ exe cũ.

Không nên xóa:

```text
models/*.onnx
models/*.onnx.data nếu ONNX đang phụ thuộc file data ngoài
```

## Có train không?

Không.

## Ảnh hưởng khi chạy

- Người dùng chọn workflow trước.
- Upload ảnh xong tự động cắt.
- Không ghi output cho tới khi bấm Save.
- Single và batch lưu ở 2 folder khác nhau.
- Debug output tách folder, dễ tắt sau này bằng `SAVE_DEBUG_OUTPUTS = False`.


# CHANGELOG - M10.1 Trimap Known Region Clamp

## Vấn đề

Debug cho thấy:

- `f2_auto_alpha_768`: đúng
- `loose_trimap_768`: đúng
- `m6_alpha_768`: background bị xám, foreground trắng
- `final_alpha`: giống M6 alpha rồi resize về size gốc

=> Lỗi không nằm ở F2_A1 và không nằm ở trimap. Lỗi nằm ở cách dùng output M6 làm final alpha.

## Sửa trong `bg_engine.py`

Thêm hàm:

```python
enforce_known_regions_from_trimap(alpha_01, trimap_u8)
```

Logic:

```python
alpha[trimap == 0] = 0.0
alpha[trimap == 255] = 1.0
alpha[trimap == 128] = giữ output M6
```

Thêm debug mới:

```text
99_debug/04_m6_alpha_768_clamped/
99_debug/05_m6_raw_alpha_768_before_clamp/
```

## Vì sao sửa

Trong trimap matting:

- 0 = chắc chắn background
- 255 = chắc chắn foreground
- 128 = vùng unknown cần model refine

M6 chỉ nên quyết định vùng unknown. Không nên để M6 làm background chắc chắn thành màu xám.

## Có train không?

Không.

## Ảnh hưởng khi chạy

- Vùng trimap đen sẽ trong suốt hoàn toàn.
- Vùng trimap trắng sẽ giữ lại hoàn toàn.
- M6 chỉ còn ảnh hưởng vùng unknown.
- Lỗi "cả ảnh bị mờ" sẽ hết nếu trimap đúng.


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
