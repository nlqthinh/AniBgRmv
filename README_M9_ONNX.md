# AniBGRmv M9 ONNX Patch

## Mục tiêu

Pipeline mặc định:

`image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> final alpha -> cutout PNG`

Không có fast mode / quality mode.

## Thứ tự dùng

### 1. Cài requirements để export model

Chỉ cần làm trên máy dev / Colab / máy build.

```bash
pip install -r requirements_export.txt
```

### 2. Đổi F2_A1 `.pth` sang ONNX

```bash
python tools/export_f2a1_to_onnx.py ^
  --pth models/F2_A1_mild_aug_from_FP3_768_unet_mit_b1_3k5_best.pth ^
  --out models/f2a1_unet_mit_b1_768.onnx ^
  --size 768
```

### 3. Đổi M6 `.pt` sang ONNX

```bash
python tools/export_m6_vitmatte_to_onnx.py ^
  --pt models/m6_vitmatte_tune_best.pt ^
  --out models/m6_vitmatte_m6_768.onnx ^
  --size 768
```

Nếu thấy `missing keys` hoặc `unexpected keys` quá nhiều, checkpoint M6 có format khác với script này. Khi đó cần sửa load checkpoint theo đúng notebook M6.

### 4. Test pipeline ONNX trước khi build exe

```bash
pip install -r requirements.txt

python tools/verify_onnx_pipeline.py ^
  --image test_images/your_image.png ^
  --out outputs/verify_onnx
```

### 5. Build exe

```bash
build_exe.bat
```

Output nằm ở:

```text
dist/AniBGRmv/AniBGRmv.exe
```

Nên phát hành cả folder `dist/AniBGRmv/` dạng `.zip`, không chỉ copy mỗi file exe.

## File nặng nhất

Nặng nhất là:

```text
models/f2a1_unet_mit_b1_768.onnx
models/m6_vitmatte_m6_768.onnx
```

Nếu dùng PyInstaller onedir, chúng nằm trong:

```text
dist/AniBGRmv/models/
```

Nếu dùng onefile, model sẽ bị nhét vào exe và mỗi lần chạy có thể phải giải nén ra temp, startup chậm hơn. Vì vậy bản patch này chọn onedir.

## Có train không?

Không. Patch này chỉ phục vụ inference/app integration/export/deploy. Không train F2_A1, không train M6.
