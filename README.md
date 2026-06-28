# Dog vs Cat Classification with ResNet

## Tong quan

Du an nay phan loai anh cho va meo tu:

- `data/train`: `25,000` anh co nhan
- `data/test`: `12,500` anh khong co nhan

Pipeline da duoc thuc thi theo huong ban chot:

Muc tieu chinh: accuracy toi da
Tu dong dung GPU neu may co, neu khong se fallback sang CPU
Output vua phu hop competition, vua trinh bay on voi doanh nghiep
Mo hinh hien tai dung transfer learning voi ResNet va da co san:

ResNet34 la config mac dinh de uu tien accuracy
ResNet18 la config nhe hon de chay tren CPU nhanh hon
Cau truc du an
classify/
|-- data/
|   |-- train/
|   `-- test/
|-- src/
|   |-- __init__.py
|   |-- datasets.py
|   |-- transforms.py
|   |-- model.py
|   |-- predictor.py
|   |-- train.py
|   |-- infer.py
|   `-- utils.py
|-- configs/
|   |-- resnet18_cpu.yaml
|   |-- resnet18_cpu_fast.yaml
|   |-- resnet18_cpu_ultrafast.yaml
|   `-- resnet34_accuracy.yaml
|-- artifacts/
|-- outputs/
|-- api.py
|-- app.py
|-- requirements.txt
`-- README.md
Flow thuc te
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
Thanh phan chinh
src/datasets.py

Tach nhan tu ten file cat.xxx.jpg va dog.xxx.jpg
Tao dataset cho train/valid va infer
src/transforms.py

Train transforms co augmentation vua phai
Eval transforms on dinh cho validation va inference
src/model.py

Ho tro resnet18 va resnet34
Thay head cuoi cho bai toan 2 lop
src/train.py

Tu dong chon cuda neu co, neu khong dung cpu
Freeze/unfreeze fine-tune
Early stopping
Luu artifacts/best.pt
src/infer.py

Load checkpoint tot nhat
Infer theo batch
Xuat 2 dinh dang ket qua
Output cho doanh nghiep
Du an xuat 2 file:

1. outputs/submission.csv
Format gon, hop voi competition hoac workflow scoring:

id,label
1,0.998421
2,0.031552
Y nghia:

id: id anh test
label: xac suat la dog
2. outputs/predictions.jsonl
Format nay phu hop hon khi demo voi doanh nghiep, vi moi dong la mot record co du:

id anh
duong dan file
nhan du doan
confidence
xac suat cho/meo
co can review thu cong hay khong
Vi du:

{"image_id":"1","file_path":"data/test/1.jpg","predicted_label":"dog","confidence":0.998421,"dog_probability":0.998421,"cat_probability":0.001579,"review_recommended":false}
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

### CPU-fast

File: `configs/resnet18_cpu_fast.yaml`

- Model: `resnet18`
- Batch size: `32`
- Image size: `160`
- Chi train classification head
- Validation thua hon de giam thoi gian moi epoch
- Nen dung dau tien neu may khong co GPU
- Da chot `num_workers: 0` theo benchmark tren may Windows cua ban

### CPU-ultrafast

File: `configs/resnet18_cpu_ultrafast.yaml`

- Model: `resnet18`
- Batch size: `64`
- Image size: `128`
- Chi train classification head
- Validation rat thua
- Dung khi can lap thu nhanh, debug pipeline, hoac test nhieu lan

## Cai dat

Neu moi truong `venv` hien tai cua ban dang loi, ban nen tao lai venv sach:

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Neu may co nhieu Python, co the dung:

py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Cach train
Mac dinh hien tai, src/train.py da uu tien config accuracy:

python -m src.train
Hoac chi dinh ro config:

python -m src.train --config configs/resnet34_accuracy.yaml
Neu muon train nhe hon tren CPU:

python -m src.train --config configs/resnet18_cpu.yaml
Neu muon uu tien toc do train tren CPU:

python -m src.train --config configs/resnet18_cpu_fast.yaml
Neu muon nhanh nhat co the tren CPU:

python -m src.train --config configs/resnet18_cpu_ultrafast.yaml
Cach infer
Sau khi da co artifacts/best.pt:

python -m src.infer --checkpoint artifacts/best.pt
Ket qua se duoc luu vao:

outputs/submission.csv
outputs/predictions.jsonl
Cach pipeline tu chon thiet bi
Code hien tai tu dong chon:

cuda neu torch.cuda.is_available() == True
cpu neu may khong co GPU
Ban khong can sua code de chay tren may hien tai khong co GPU.

Chien luoc toi uu accuracy
Neu muon day accuracy len tiep, nen thu theo thu tu sau:

1. Train `resnet34_accuracy.yaml` truoc
2. Tang `epochs` neu valid metric van di len
3. Tang `tta_passes` trong config khi infer
4. Giam learning rate backbone them mot chut
5. Thu review cac anh predict sai tu confusion pattern

## Toi uu toc do train

Minh da toi uu lai pipeline theo huong CPU-first:

- Giam `image_size` o che do nhanh
- Dung augmentation nhe hon
- Bo tinh metric train moi epoch khi khong can
- Cho phep chi train classification head
- Validation khong can chay moi epoch
- Chon `num_workers: 0` tren Windows khi benchmark thuc te cho thay worker startup dat hon loi ich

Neu may ban van cham, thu theo thu tu nay:

1. `python -m src.train --config configs/resnet18_cpu_fast.yaml`
2. `python -m src.train --config configs/resnet18_cpu_ultrafast.yaml`
3. `python -m src.train --config configs/resnet18_cpu.yaml`
4. `python -m src.train --config configs/resnet34_accuracy.yaml`

## File sinh ra sau khi train

- `artifacts/best.pt`: checkpoint tot nhat
- `artifacts/best_metrics.json`: metric tot nhat
- `outputs/training_history.csv`: lich su train theo epoch

## Ghi chu quan trong

- `train.py` dang luu best model theo `valid_f1`, vi metric nay can bang hon accuracy trong giai doan toi uu. Tuy nhien voi bo du lieu can bang nhu hien tai, khi `f1` tang thi accuracy thuong cung tang rat sat.
- Neu ban muon toi uu thang theo `valid_accuracy`, minh co the chinh lai trong 1 phut.
- `ResNet34` tren CPU se chay kha lau. Neu uu tien trai nghiem phat trien tren may ca nhan, hay chay `ResNet18` truoc de xac nhan pipeline.
- Neu 1 epoch truoc day mat khoang 30 phut, nguyen nhan chinh thuong la do `224x224` + augmentation nang + validation day du moi epoch tren CPU.
- Benchmark tren may ban cho thay `num_workers=0` cho tong thoi gian epoch tot hon `1` va `2`, du `train_sec_per_batch` co the nhinh hon mot chut, vi startup worker tren Windows qua ton.

## Buoc tiep theo de nen lam

Neu ban muon, minh co the lam tiep ngay 1 trong 3 huong:

1. Them script danh gia anh predict sai de xem cac truong hop mo hinh nham.
2. Them file `app.py` hoac API nho de demo cho doanh nghiep.
3. Them notebook EDA va bao cao metric truc quan.

## Demo API + Streamlit

Du an hien tai da co:

- `app.py`: Streamlit UI tieng Anh de demo ket qua
- `src/predictor.py`: shared inference helper de load model va predict truc tiep trong Streamlit
- `api.py`: FastAPI backend tuy chon, de dung khi ban muon tach rieng UI va backend sau nay

### Cach chay toi uu cho deploy Streamlit

```powershell
streamlit run app.py
```

App se:

- load `artifacts/best.pt` truc tiep
- cache model bang `st.cache_resource`
- khong can `127.0.0.1:8000`
- khong reload model moi lan upload anh

### Luong demo

1. Start Streamlit
2. Mo giao dien tren browser
3. Upload anh cho hoac meo moi
4. Bam `Run Prediction`
5. Xem:
   - predicted label
   - confidence
   - dog probability
   - cat probability
   - review recommendation

### Man hinh co san trong Streamlit

- `Model Overview`
  - hien best metric tu `artifacts/best_metrics.json`
  - hien chart tu `outputs/training_history.csv`

- `Predict New Image`
  - upload anh moi
  - predict truc tiep trong app

- `Batch Predictions`
  - doc `outputs/predictions.jsonl`
  - loc theo nhan
  - loc cac du doan can review
  - tai `submission.csv` va `predictions.jsonl`

### Khi nao moi can API rieng

Neu sau nay ban muon:

- mot frontend rieng
- mobile app goi predict
- he thong doanh nghiep goi qua HTTP

thi moi can chay them:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Con voi demo Streamlit deploy len cloud, cach toi uu va on dinh nhat la predict truc tiep trong `app.py`.
