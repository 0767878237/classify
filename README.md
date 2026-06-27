# Dog vs Cat Classification with ResNet

## Tong quan

Du an nay phan loai anh cho va meo tu:

- `data/train`: `25,000` anh co nhan
- `data/test`: `12,500` anh khong co nhan

Pipeline da duoc thuc thi theo huong ban chot:

- Muc tieu chinh: `accuracy toi da`
- Tu dong dung `GPU` neu may co, neu khong se fallback sang `CPU`
- Output vua phu hop competition, vua trinh bay on voi doanh nghiep

Mo hinh hien tai dung transfer learning voi ResNet va da co san:

- `ResNet34` la config mac dinh de uu tien accuracy
- `ResNet18` la config nhe hon de chay tren CPU nhanh hon

## Cau truc du an

```text
classify/
|-- data/
|   |-- train/
|   `-- test/
|-- src/
|   |-- __init__.py
|   |-- datasets.py
|   |-- transforms.py
|   |-- model.py
|   |-- train.py
|   |-- infer.py
|   `-- utils.py
|-- configs/
|   |-- resnet18_cpu.yaml
|   `-- resnet34_accuracy.yaml
|-- artifacts/
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Flow thuc te

```text
1. Doc anh tu data/train
2. Tach train/valid theo stratified split
3. Augmentation + normalize theo ImageNet
4. Load ResNet pretrained
5. Freeze backbone trong vai epoch dau
6. Unfreeze va fine-tune de toi da hoa ket qua
7. Luu checkpoint tot nhat theo valid F1
8. Chay infer tren data/test
9. Xuat:
   - submission.csv
   - predictions.jsonl
```

## Thanh phan chinh

- `src/datasets.py`
  - Tach nhan tu ten file `cat.xxx.jpg` va `dog.xxx.jpg`
  - Tao dataset cho train/valid va infer

- `src/transforms.py`
  - Train transforms co augmentation vua phai
  - Eval transforms on dinh cho validation va inference

- `src/model.py`
  - Ho tro `resnet18` va `resnet34`
  - Thay head cuoi cho bai toan 2 lop

- `src/train.py`
  - Tu dong chon `cuda` neu co, neu khong dung `cpu`
  - Freeze/unfreeze fine-tune
  - Early stopping
  - Luu `artifacts/best.pt`

- `src/infer.py`
  - Load checkpoint tot nhat
  - Infer theo batch
  - Xuat 2 dinh dang ket qua

## Output cho doanh nghiep

Du an xuat 2 file:

### 1. `outputs/submission.csv`

Format gon, hop voi competition hoac workflow scoring:

```csv
id,label
1,0.998421
2,0.031552
```

Y nghia:

- `id`: id anh test
- `label`: xac suat la `dog`

### 2. `outputs/predictions.jsonl`

Format nay phu hop hon khi demo voi doanh nghiep, vi moi dong la mot record co du:

- id anh
- duong dan file
- nhan du doan
- confidence
- xac suat cho/meo
- co can review thu cong hay khong

Vi du:

```json
{"image_id":"1","file_path":"data/test/1.jpg","predicted_label":"dog","confidence":0.998421,"dog_probability":0.998421,"cat_probability":0.001579,"review_recommended":false}
```

Format nay de dua vao:

- dashboard
- API response
- luong duyet noi bo
- bao cao demo voi doanh nghiep

## Cau hinh hien co

### Accuracy-first

File: `configs/resnet34_accuracy.yaml`

- Model: `resnet34`
- Batch size: `8`
- Epochs: `24`
- Freeze warmup: `2`
- Early stopping: `5`
- Dung cho truong hop uu tien ket qua tot nhat co the

### CPU-friendly

File: `configs/resnet18_cpu.yaml`

- Model: `resnet18`
- Batch size: `16`
- Epochs: `18`
- Nhe hon, hop khi train tren CPU lau dai

## Cai dat

Neu moi truong `venv` hien tai cua ban dang loi, ban nen tao lai venv sach:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu may co nhieu Python, co the dung:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cach train

Mac dinh hien tai, `src/train.py` da uu tien config accuracy:

```powershell
python -m src.train
```

Hoac chi dinh ro config:

```powershell
python -m src.train --config configs/resnet34_accuracy.yaml
```

Neu muon train nhe hon tren CPU:

```powershell
python -m src.train --config configs/resnet18_cpu.yaml
```

## Cach infer

Sau khi da co `artifacts/best.pt`:

```powershell
python -m src.infer --checkpoint artifacts/best.pt
```

Ket qua se duoc luu vao:

- `outputs/submission.csv`
- `outputs/predictions.jsonl`

## Cach pipeline tu chon thiet bi

Code hien tai tu dong chon:

- `cuda` neu `torch.cuda.is_available() == True`
- `cpu` neu may khong co GPU

Ban khong can sua code de chay tren may hien tai khong co GPU.

## Chien luoc toi uu accuracy

Neu muon day accuracy len tiep, nen thu theo thu tu sau:

1. Train `resnet34_accuracy.yaml` truoc
2. Tang `epochs` neu valid metric van di len
3. Tang `tta_passes` trong config khi infer
4. Giam learning rate backbone them mot chut
5. Thu review cac anh predict sai tu confusion pattern

## File sinh ra sau khi train

- `artifacts/best.pt`: checkpoint tot nhat
- `artifacts/best_metrics.json`: metric tot nhat
- `outputs/training_history.csv`: lich su train theo epoch

## Ghi chu quan trong

- `train.py` dang luu best model theo `valid_f1`, vi metric nay can bang hon accuracy trong giai doan toi uu. Tuy nhien voi bo du lieu can bang nhu hien tai, khi `f1` tang thi accuracy thuong cung tang rat sat.
- Neu ban muon toi uu thang theo `valid_accuracy`, minh co the chinh lai trong 1 phut.
- `ResNet34` tren CPU se chay kha lau. Neu uu tien trai nghiem phat trien tren may ca nhan, hay chay `ResNet18` truoc de xac nhan pipeline.

## Buoc tiep theo de nen lam

Neu ban muon, minh co the lam tiep ngay 1 trong 3 huong:

1. Them script danh gia anh predict sai de xem cac truong hop mo hinh nham.
2. Them file `app.py` hoac API nho de demo cho doanh nghiep.
3. Them notebook EDA va bao cao metric truc quan.
